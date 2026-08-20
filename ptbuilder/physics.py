from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from scipy import interpolate, optimize

try:
    from ptbuilder._potential import (
        Potential,
        Phi4Potential as _Phi4Potential,
        Phi4PiecewisePotential as _Phi4PiecewisePotential,
        PolynomialPotential as _PolynomialPotential,
    )
    _POTENTIAL_AVAILABLE = True
except ImportError:
    _POTENTIAL_AVAILABLE = False

    class Potential:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError(
                "ptbuilder._potential (C++ extension) is not compiled. "
                "Run `make` from the repo root to build it."
            )

    _Phi4Potential           = Potential
    _Phi4PiecewisePotential  = Potential
    _PolynomialPotential     = Potential


# ---------------------------------------------------------------------------
# Python mixin: numpy vectorization, HDF5 serialization, derived quantities
# ---------------------------------------------------------------------------

class _PotentialMixin:
    def V(self, phi):
        return np.vectorize(super().V)(np.asarray(phi, dtype=float))

    def dV(self, phi):
        return np.vectorize(super().dV)(np.asarray(phi, dtype=float))

    @property
    def delta_V(self) -> float:
        return float(super().V(self.phi_false) - super().V(self.phi_true))

    @property
    def name(self) -> str:
        param_str = "_".join(f"{k}{v:.6g}" for k, v in self.params.items())
        return f"{self.type_name}_{param_str}"

    def to_hdf5(self, group: h5py.Group) -> None:
        # Write as variable-length ASCII so C++ H5::PredType::C_S1 can read it.
        group.attrs.create("type", self.type_name,
                           dtype=h5py.string_dtype(encoding='ascii'))
        group.attrs["phi_false"] = float(self.phi_false)
        group.attrs["phi_true"]  = float(self.phi_true)
        for key, val in self.params.items():
            group.attrs[key] = float(val)


# ---------------------------------------------------------------------------
# Concrete potential classes (math lives in C++)
# ---------------------------------------------------------------------------

class Phi4Potential(_PotentialMixin, _Phi4Potential):
    pass

class Phi4PiecewisePotential(_PotentialMixin, _Phi4PiecewisePotential):
    pass

class PolynomialPotential(_PotentialMixin, _PolynomialPotential):
    pass

_POTENTIAL_REGISTRY = {
    "phi4":           Phi4Potential,
    "phi4_piecewise": Phi4PiecewisePotential,
    "polynomial":     PolynomialPotential,
}

def potential_from_hdf5(group: h5py.Group) -> Potential:
    ptype = str(group.attrs["type"])
    if ptype not in _POTENTIAL_REGISTRY:
        raise ValueError(f"Unknown potential type {ptype!r}. "
                         f"Known: {sorted(_POTENTIAL_REGISTRY)}")
    params = {k: float(v) for k, v in group.attrs.items()
              if k not in ("type", "phi_false", "phi_true")}
    return _POTENTIAL_REGISTRY[ptype](**params)


# ---------------------------------------------------------------------------
# Instanton profile
# ---------------------------------------------------------------------------

@dataclass
class InstantonProfile:
    """
    Stores the tunnelling (instanton) solution and derived bubble wall radii.

    Radii (rin_0, rmid_0, rout_0) are defined via the tanh contours of the
    field profile, consistent with the bubble_initialiser convention.
    """
    R:      np.ndarray   # radial coordinate array
    Phi:    np.ndarray   # field values along R
    rin_0:  float        # inner wall radius (tanh(-0.5) contour)
    rmid_0: float        # mid-wall radius   (half-amplitude)
    rout_0: float        # outer wall radius (tanh(+0.5) contour)
    interp: object       # callable r -> phi (scipy interp1d)

    @property
    def wall_width(self) -> float:
        return self.rout_0 - self.rin_0


# ---------------------------------------------------------------------------
# Bubble kinematics
# ---------------------------------------------------------------------------

class BubbleKinematics:
    """
    Kinematic quantities for an expanding bubble wall derived from an
    InstantonProfile.  All radii follow R(t) = sqrt(R0^2 + (t - t_init)^2).
    """

    def __init__(self, profile: InstantonProfile):
        self.profile = profile

    def R(self, t, t_init: float = 0., r: str = "mid") -> np.ndarray:
        """Bubble wall radius at time t for a bubble nucleated at t_init."""
        R0 = {"in": self.profile.rin_0,
              "mid": self.profile.rmid_0,
              "center": self.profile.rmid_0,
              "out": self.profile.rout_0}[r]
        return np.sqrt(R0**2 + np.maximum(0., t - t_init)**2)

    def Gamma(self, t, t_init: float = 0.) -> np.ndarray:
        """Lorentz factor of the bubble wall at time t."""
        w0 = self.profile.wall_width
        w  = self.R(t, t_init, "out") - self.R(t, t_init, "in")
        return w0 / w

    def collision_time(self, d: float,
                       t0_nuc: float = 0., t1_nuc: float = 0.,
                       collision_radius: str = "mid") -> float:
        """
        Analytical collision time for two bubbles with centre-to-centre
        separation d and nucleation times t0_nuc, t1_nuc.
        ``collision_radius`` selects which wall surface first touching defines
        the collision: ``"mid"`` (wall midpoint) or ``"out"`` (outer edge).
        """
        if collision_radius not in ("mid", "out"):
            raise ValueError("collision_radius must be 'mid' or 'out'")
        R0 = (self.profile.rmid_0 if collision_radius == "mid"
              else self.profile.rout_0)
        dt = t0_nuc - t1_nuc
        disc = (d**2 - dt**2) * (d**2 - 4*R0**2 - dt**2)
        num  = (d * np.sqrt(disc)
                + d**2*(t0_nuc + t1_nuc)
                - dt**2*(t0_nuc + t1_nuc))
        return num / (2.*(d**2 - dt**2))


# ---------------------------------------------------------------------------
# PhysicsModel: ties everything together
# ---------------------------------------------------------------------------

class PhysicsModel:
    """
    Central object for a phase transition model.

    Holds a Potential and lazily computes / disk-caches the instanton profile
    and bubble kinematics.  The results directory is automatically namespaced
    by model name so different lambda_bar values don't collide.
    """

    def __init__(self, potential: Potential, config=None,
                 instanton_alpha: int = 3):
        from .config import Config
        self.potential        = potential
        self.config           = config or Config()
        self.instanton_alpha  = instanton_alpha

    @property
    def name(self) -> str:
        return self.potential.name

    @property
    def results_dir(self) -> Path:
        d = self.config.results_dir / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- instanton (disk-cached) --------------------------------------------

    @cached_property
    def instanton(self) -> InstantonProfile:
        cache = self.results_dir / "instanton.h5"
        if cache.exists():
            return self._load_instanton(cache)
        profile = self._compute_instanton()
        self._save_instanton(profile, cache)
        return profile

    def _compute_instanton(self) -> InstantonProfile:
        from cosmoTransitions.tunneling1D import SingleFieldInstanton
        pot = self.potential
        kw  = dict(xtol=1e-5, phitol=1e-5)
        if hasattr(pot, "phi_esc"):
            kw["xguess"] = pot.phi_esc
        raw = SingleFieldInstanton(
            pot.phi_true, pot.phi_false, pot.V, pot.dV,
            alpha=self.instanton_alpha,
        ).findProfile(**kw)

        phi0    = raw.Phi[0]
        phi_in  = phi0 * (1 - np.tanh(-0.5)) / 2
        phi_out = phi0 * (1 - np.tanh( 0.5)) / 2
        phi_mid = phi0 * 0.5

        def _r(target):
            return optimize.brentq(
                lambda r: np.interp(r, raw.R, raw.Phi) - target,
                raw.R[0], raw.R[-1],
            )

        interp = interpolate.interp1d(
            raw.R, raw.Phi, kind="cubic",
            bounds_error=False,
            fill_value=(raw.Phi[0], raw.Phi[-1]),
        )
        return InstantonProfile(
            R=raw.R, Phi=raw.Phi,
            rin_0=_r(phi_in), rmid_0=_r(phi_mid), rout_0=_r(phi_out),
            interp=interp,
        )

    def _save_instanton(self, p: InstantonProfile, path: Path):
        with h5py.File(path, "w") as f:
            f.create_dataset("R",   data=p.R)
            f.create_dataset("Phi", data=p.Phi)
            f.attrs["rin_0"]  = p.rin_0
            f.attrs["rmid_0"] = p.rmid_0
            f.attrs["rout_0"] = p.rout_0
            # store potential so the cache is self-documenting
            self.potential.to_hdf5(f.require_group("potential"))

    def _load_instanton(self, path: Path) -> InstantonProfile:
        with h5py.File(path, "r") as f:
            R, Phi = f["R"][:], f["Phi"][:]
            rin_0  = float(f.attrs["rin_0"])
            rmid_0 = float(f.attrs["rmid_0"])
            rout_0 = float(f.attrs["rout_0"])
        interp = interpolate.interp1d(
            R, Phi, kind="cubic",
            bounds_error=False, fill_value=(Phi[0], Phi[-1]),
        )
        return InstantonProfile(R=R, Phi=Phi,
                                rin_0=rin_0, rmid_0=rmid_0, rout_0=rout_0,
                                interp=interp)

    # -- kinematics (derived from instanton) --------------------------------

    @cached_property
    def kinematics(self) -> BubbleKinematics:
        return BubbleKinematics(self.instanton)
