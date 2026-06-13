from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Optional

import h5py
import numpy as np
from scipy import interpolate, optimize


# ---------------------------------------------------------------------------
# Potential base class + registry
# ---------------------------------------------------------------------------

class Potential(ABC):
    """
    Abstract base class for scalar field potentials.

    Subclasses must define:
      - type_name (ClassVar[str])  -- unique string identifier
      - phi_false, phi_true (floats set in __init__)
      - V, dV, d2V
      - params (property returning dict of primary parameters)
      - _from_hdf5_params (classmethod to reconstruct from an HDF5 group)

    Serialization (to_hdf5 / from_hdf5) writes the type_name + params as
    HDF5 group attributes.  The matching C++ Potential::from_hdf5() in
    cpp/common/potential.h reads the same layout.
    """

    _registry: ClassVar[dict] = {}
    type_name: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "type_name" in cls.__dict__:
            Potential._registry[cls.type_name] = cls

    # -- abstract interface --------------------------------------------------

    @abstractmethod
    def V(self, phi: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def dV(self, phi: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def d2V(self, phi: np.ndarray) -> np.ndarray: ...

    @property
    @abstractmethod
    def params(self) -> dict:
        """Primary parameters that fully specify this potential instance."""
        ...

    @classmethod
    @abstractmethod
    def _from_hdf5_params(cls, group: h5py.Group) -> "Potential": ...

    # -- derived quantities --------------------------------------------------

    @property
    def delta_V(self) -> float:
        return float(self.V(self.phi_false) - self.V(self.phi_true))

    @property
    def name(self) -> str:
        param_str = "_".join(f"{k}{v:.6g}" for k, v in self.params.items())
        return f"{self.type_name}_{param_str}"

    # -- serialization -------------------------------------------------------

    def to_hdf5(self, group: h5py.Group) -> None:
        """Write potential to an HDF5 group (readable by C++ potential.h)."""
        group.attrs["type"]      = self.type_name
        group.attrs["phi_false"] = float(self.phi_false)
        group.attrs["phi_true"]  = float(self.phi_true)
        for key, val in self.params.items():
            group.attrs[key] = float(val)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "Potential":
        """Reconstruct a Potential from an HDF5 group written by to_hdf5."""
        ptype = str(group.attrs["type"])
        if ptype not in cls._registry:
            raise ValueError(
                f"Unknown potential type {ptype!r}. "
                f"Known types: {sorted(cls._registry)}"
            )
        return cls._registry[ptype]._from_hdf5_params(group)


# ---------------------------------------------------------------------------
# Phi^4 potential  (arXiv:2005.13537)
# ---------------------------------------------------------------------------

class Phi4Potential(Potential):
    """
    Renormalisable phi^4 potential parameterised by a single lambda_bar.
    False vacuum: phi=0,  true vacuum: phi=1.
    """

    type_name = "phi4"

    def __init__(self, lambda_bar: float):
        self.lambda_bar = lambda_bar
        up = 3.0 + np.sqrt(9.0 - 8.0 * lambda_bar)
        self.m2    = 0.5
        self.delta = -(3.0 + np.sqrt(9.0 - 8.0 * lambda_bar)) / (4.0 * lambda_bar)
        self.lam   = up**2 / (32.0 * lambda_bar)
        self.phi_false = 0.0
        self.phi_true  = 1.0

    @property
    def params(self) -> dict:
        return {"lambda_bar": self.lambda_bar}

    @classmethod
    def _from_hdf5_params(cls, group: h5py.Group) -> "Phi4Potential":
        return cls(float(group.attrs["lambda_bar"]))

    def V(self, phi):
        return self.m2*phi**2 + self.delta*phi**3 + self.lam*phi**4

    def dV(self, phi):
        return 2.*self.m2*phi + 3.*self.delta*phi**2 + 4.*self.lam*phi**3

    def d2V(self, phi):
        return 2.*self.m2 + 6.*self.delta*phi + 12.*self.lam*phi**2


# ---------------------------------------------------------------------------
# Phi^4 + piecewise extension ("bubble tails" model)
# ---------------------------------------------------------------------------

class Phi4PiecewisePotential(Potential):
    """
    Phi^4 for phi <= phi_esc, quadratic well for phi > phi_esc.
    The two pieces are joined smoothly via a cosmological-constant shift Lambda.
    """

    type_name = "phi4_piecewise"

    def __init__(self, lambda_bar: float, phi_esc: float,
                 eps: float, vbar: float):
        self.lambda_bar = lambda_bar
        self.phi_esc    = phi_esc
        self.eps        = eps
        self.vbar       = vbar

        up = 3.0 + np.sqrt(9.0 - 8.0 * lambda_bar)
        self.m2    = 0.5
        self.delta = -(3.0 + np.sqrt(9.0 - 8.0 * lambda_bar)) / (4.0 * lambda_bar)
        self.lam   = up**2 / (32.0 * lambda_bar)
        self.Lambda = (
            0.5*eps**2*vbar**2 - eps**2*vbar*phi_esc
            + 0.5*(eps**2 - 1.)*phi_esc**2
            - up*phi_esc**3 / (32.*lambda_bar) * (up*phi_esc - 8.)
        )
        self.phi_false = 0.0
        self.phi_true  = vbar

    @property
    def params(self) -> dict:
        return {
            "lambda_bar": self.lambda_bar,
            "phi_esc":    self.phi_esc,
            "eps":        self.eps,
            "vbar":       self.vbar,
        }

    @classmethod
    def _from_hdf5_params(cls, group: h5py.Group) -> "Phi4PiecewisePotential":
        return cls(
            float(group.attrs["lambda_bar"]),
            float(group.attrs["phi_esc"]),
            float(group.attrs["eps"]),
            float(group.attrs["vbar"]),
        )

    def V(self, phi):
        phi4 = self.m2*phi**2 + self.delta*phi**3 + self.lam*phi**4
        pw   = 0.5*self.eps**2*(phi - self.vbar)**2 - self.Lambda
        return np.where(phi <= self.phi_esc, phi4, pw)

    def dV(self, phi):
        phi4 = 2.*self.m2*phi + 3.*self.delta*phi**2 + 4.*self.lam*phi**3
        pw   = self.eps**2*(phi - self.vbar)
        return np.where(phi <= self.phi_esc, phi4, pw)

    def d2V(self, phi):
        phi4 = 2.*self.m2 + 6.*self.delta*phi + 12.*self.lam*phi**2
        pw   = np.full_like(np.asarray(phi, dtype=float), self.eps**2)
        return np.where(phi <= self.phi_esc, phi4, pw)


# ---------------------------------------------------------------------------
# Polynomial potential  (pot_type=0, old literature)
# ---------------------------------------------------------------------------

class PolynomialPotential(Potential):
    """
    Old-literature polynomial potential for validation (pot_type=0).

    V  = phi^4/4 - phi^3/3 + (1/9)*lambda_bar*phi^2
    dV = phi^3 - phi^2 + (2/9)*lambda_bar*phi

    False vacuum: phi=0.  True vacuum: phi=(1+sqrt(1-8*lambda_bar/9))/2.
    Valid for lambda_bar <= 9/8.
    """

    type_name = "polynomial"

    def __init__(self, lambda_bar: float):
        self.lambda_bar = lambda_bar
        disc = 1.0 - (8.0 / 9.0) * lambda_bar
        if disc < 0:
            raise ValueError(
                f"lambda_bar={lambda_bar} too large for PolynomialPotential "
                "(need lambda_bar <= 9/8)"
            )
        self.phi_false = 0.0
        self.phi_true  = (1.0 + np.sqrt(disc)) / 2.0

    @property
    def params(self) -> dict:
        return {"lambda_bar": self.lambda_bar}

    @classmethod
    def _from_hdf5_params(cls, group: h5py.Group) -> "PolynomialPotential":
        return cls(float(group.attrs["lambda_bar"]))

    def V(self, phi):
        lb = self.lambda_bar
        return phi**4 / 4. - phi**3 / 3. + (1./9.) * lb * phi**2

    def dV(self, phi):
        lb = self.lambda_bar
        return phi**3 - phi**2 + (2./9.) * lb * phi

    def d2V(self, phi):
        lb = self.lambda_bar
        return 3.*phi**2 - 2.*phi + (2./9.) * lb


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
                       t0_nuc: float = 0., t1_nuc: float = 0.) -> float:
        """
        Analytical collision time for two bubbles with centre-to-centre
        separation d and nucleation times t0_nuc, t1_nuc.
        Both bubbles are assumed to have the same rout_0.
        """
        R0 = self.profile.rmid_0
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
