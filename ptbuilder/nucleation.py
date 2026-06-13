from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Nucleation rate models
# ---------------------------------------------------------------------------

class NucleationRate(ABC):
    """
    Strategy for sampling bubble nucleation events.

    Returns arrays of positions and nucleation times for N bubbles inside a
    cubic box of side box_size.
    """

    @abstractmethod
    def sample(self, box_size: float, n_bubbles: int,
               rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            positions: (n_bubbles, 3) in [0, box_size)
            times:     (n_bubbles,)
        """
        ...


class UniformNucleation(NucleationRate):
    """Uniform (Poisson) nucleation over a fixed time window."""

    def __init__(self, t_start: float = 0., t_end: float = 1.):
        self.t_start = t_start
        self.t_end   = t_end

    def sample(self, box_size, n_bubbles, rng):
        positions = rng.uniform(0., box_size, (n_bubbles, 3))
        times     = rng.uniform(self.t_start, self.t_end, n_bubbles)
        return positions, times


class ExponentialNucleation(NucleationRate):
    """
    Exponential nucleation rate dN/dt ∝ exp(β t).
    Inverse-CDF sampling gives times distributed according to this rate.
    """

    def __init__(self, beta: float, t_start: float = 0., t_end: float = 1.):
        self.beta    = beta
        self.t_start = t_start
        self.t_end   = t_end

    def sample(self, box_size, n_bubbles, rng):
        positions = rng.uniform(0., box_size, (n_bubbles, 3))
        u         = rng.uniform(0., 1., n_bubbles)
        exp_start = np.exp(self.beta * self.t_start)
        exp_end   = np.exp(self.beta * self.t_end)
        times     = np.log(u * (exp_end - exp_start) + exp_start) / self.beta
        return positions, times


class FixedNucleation(NucleationRate):
    """Pre-specified positions and times; useful for unit tests and demos."""

    def __init__(self, positions: np.ndarray, times: np.ndarray):
        self._positions = np.asarray(positions, dtype=float)
        self._times     = np.asarray(times,     dtype=float)

    def sample(self, box_size, n_bubbles, rng):
        return self._positions[:n_bubbles], self._times[:n_bubbles]


# ---------------------------------------------------------------------------
# Bubble population
# ---------------------------------------------------------------------------

@dataclass
class BubbleEvent:
    index:    int            # ordinal (sorted by nucleation time)
    position: np.ndarray     # shape (3,)
    t_nuc:    float

    def __repr__(self):
        return (f"BubbleEvent(index={self.index}, "
                f"t_nuc={self.t_nuc:.4f}, pos={self.position})")


@dataclass
class CollisionEvent:
    i:      int    # index into BubbleEvent list
    j:      int
    t_coll: float  # collision time
    d:      float  # centre-to-centre separation
    gamma:  float  # Lorentz factor of wall at collision (from BubbleKinematics)

    def __repr__(self):
        return (f"CollisionEvent(i={self.i}, j={self.j}, "
                f"d={self.d:.4f}, t_coll={self.t_coll:.4f}, "
                f"gamma={self.gamma:.4f})")


class BubblePopulation:
    """
    A realisation of bubble nucleation events inside a periodic box.

    After constructing the events, all pairwise collision times are computed
    using BubbleKinematics.collision_time().  Only pairs whose collision
    occurs after both bubbles have nucleated are recorded.
    """

    def __init__(self, events: list, collisions: list, box_size: float,
                 kinematics=None):
        self.events     = events
        self.collisions = collisions
        self.box_size   = box_size
        self.kinematics = kinematics

    # -- factory -----------------------------------------------------------

    @classmethod
    def generate(cls, n_bubbles: int, nucleation: NucleationRate,
                 kinematics, box_size: float,
                 seed: int = 0) -> "BubblePopulation":
        """
        Sample n_bubbles events from `nucleation`, then compute all pairwise
        collisions using `kinematics` (a BubbleKinematics instance).
        """
        from .physics import BubbleKinematics
        if not isinstance(kinematics, BubbleKinematics):
            raise TypeError("kinematics must be a BubbleKinematics instance")

        rng = np.random.default_rng(seed)
        positions, times = nucleation.sample(box_size, n_bubbles, rng)

        order  = np.argsort(times)
        events = [
            BubbleEvent(index=idx, position=positions[i], t_nuc=times[i])
            for idx, i in enumerate(order)
        ]
        collisions = cls._find_collisions(events, kinematics, box_size)
        return cls(events, collisions, box_size, kinematics=kinematics)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _separation(x0: np.ndarray, x1: np.ndarray,
                    box_size: float) -> float:
        """Minimum-image convention distance."""
        d = np.abs(x0 - x1)
        d = np.where(d > box_size / 2., box_size - d, d)
        return float(np.sqrt(np.sum(d**2)))

    @classmethod
    def _find_collisions(cls, events, kinematics,
                         box_size: float) -> list:
        collisions = []
        n = len(events)
        for i in range(n):
            for j in range(i + 1, n):
                d = cls._separation(events[i].position,
                                    events[j].position, box_size)
                try:
                    t_c = kinematics.collision_time(
                        d, events[i].t_nuc, events[j].t_nuc
                    )
                    t_after = max(events[i].t_nuc, events[j].t_nuc)
                    if np.isfinite(t_c) and t_c > t_after:
                        gamma = float(kinematics.Gamma(t_c, events[i].t_nuc))
                        collisions.append(
                            CollisionEvent(i=i, j=j, t_coll=t_c,
                                           d=d, gamma=gamma)
                        )
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    pass
        return sorted(collisions, key=lambda c: c.t_coll)

    # -- queries -----------------------------------------------------------

    def n_bubbles(self) -> int:
        return len(self.events)

    def n_collisions(self) -> int:
        return len(self.collisions)

    def collision_gammas(self) -> np.ndarray:
        return np.array([c.gamma for c in self.collisions])

    def collision_times(self) -> np.ndarray:
        return np.array([c.t_coll for c in self.collisions])

    def collision_separations(self) -> np.ndarray:
        return np.array([c.d for c in self.collisions])
