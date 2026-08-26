"""
Initial conditions: write setup files for BubbleMaster and sledgehamr.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# 2D BubbleMaster setup
# ---------------------------------------------------------------------------

@dataclass
class BubbleMasterParams:
    """
    All grid / physics parameters needed to run one BubbleMaster simulation.

    ``collision_radius`` selects whether collision means first contact of the
    wall midpoint or outer edge.  Gamma is always the physical wall-thinning
    ratio and is made equal to the requested gamma at that contact time.
    """
    # resolution
    # Choose exactly one spatial-resolution convention.  ``wall_points``
    # resolves the Lorentz-contracted wall at collision and is therefore the
    # preferred choice for scans over gamma.  ``dz`` remains available for
    # reproducing older fixed-grid runs.
    dz:           Optional[float] = 0.005
    wall_points:  Optional[int]   = None
    n_w:          int   = 128
    n_k:          int   = 51
    how_often_ds: int   = 5
    baby_steps:   int   = 100

    # cutoff function
    cutoff_type:  int   = 0    # 0=Gaussian, 1=sin^2, 2=polynomial
    t_0_scal:     float = 1.0  # scales the cutoff decay width
    collision_radius: str = "mid"  # "mid" or "out"


def _mass_scale(potential) -> float:
    """
    Scalar mass at the top of the barrier from the phi^4 lambda_bar parametrisation.
    M = 1/2 * sqrt(3*Upsilon/lambda_bar - 8), where Upsilon = 3 + sqrt(9 - 8*lambda_bar).
    """
    lb = potential.params["lambda_bar"]
    up = 3.0 + np.sqrt(9.0 - 8.0 * lb)
    return 0.5 * np.sqrt(3.0 * up / lb - 8.0)


def _two_bubble_ic(profile, gamma: float, dz: float,
                   collision_radius: str = "mid"):
    """
    Build the initial field profile and z grid for two colliding bubbles.

    Returns (z, phi0, d, ds):
      - z     : 1-D array of z-grid values (half-grid, centre at origin)
      - phi0  : corresponding combined field phi = sqrt(phi1^2 + phi2^2)
      - d     : centre-to-centre bubble separation
      - ds    : step size in s (Milne arc-length)

    The separation d is chosen so that the wall Lorentz factor (wall-thinning
    ratio) equals gamma exactly when the selected wall surfaces first touch.
    """
    r_out = profile.rout_0
    r_in  = profile.rin_0
    w0    = r_out - r_in
    if collision_radius not in ("mid", "out"):
        raise ValueError("collision_radius must be 'mid' or 'out'")
    r_collision = (profile.rmid_0 if collision_radius == "mid" else r_out)

    def _gamma_at_t(t):
        return w0 / (np.sqrt(r_out**2 + t**2) - np.sqrt(r_in**2 + t**2))

    if gamma <= 1.0:
        t_coll = 0.0
    else:
        t_coll = brentq(lambda t: _gamma_at_t(t) - gamma,
                        0., gamma * (r_out + r_in))
    d  = 2.0 * np.sqrt(r_collision**2 + t_coll**2)
    ds = dz * 0.2

    # Extend profile arrays for interpolation out to d
    R   = np.append(profile.R,   [profile.R[-1] + (profile.R[1] - profile.R[0]), d])
    Phi = np.append(profile.Phi, [0.0, 0.0])

    # Right bubble centred at 0 (growing toward +d)
    phi1_interp = interp1d(R, Phi, kind="linear", bounds_error=False,
                           fill_value=(Phi[0], 0.0))

    # Left bubble centred at d (growing toward -d)
    R2   = -R[::-1] + d
    Phi2 = Phi[::-1]
    phi2_interp = interp1d(R2, Phi2, kind="linear", bounds_error=False,
                           fill_value=(0.0, Phi2[-1]))

    # z grid from r_in (inner radius) to d
    n_z = int(round(d / dz))
    if n_z % 2 == 0:
        n_z += 1
    z_raw = np.linspace(R[0], d, n_z)

    phi1 = phi1_interp(z_raw)
    phi2 = phi2_interp(z_raw)
    phi_comb = np.sqrt(phi1**2 + phi2**2)

    # Half-grid: take right half (from just past 0 to d/2)
    half_idx   = 1 + int((n_z - 1) / 2)
    z_half     = z_raw[1:half_idx]
    phi_half   = phi_comb[1:half_idx]

    # Left side: z_raw shifted to [-d..0] with left-bubble profile
    z_left  = z_raw - z_raw[-1]   # [-d+R[0], ..., 0]
    phi_left = phi2               # left bubble

    # Combine: left side + right half
    z_comb   = np.append(z_left, z_half)
    phi_comb = np.append(phi_left, phi_half)

    # Flip and shift so z runs from 0 (collision center) outward
    phi_comb = phi_comb[::-1]
    z_comb   = -z_comb[::-1] + d / 2.0

    # Add trailing zeros (vacuum region beyond the outer bubble wall)
    t_max_approx = 12.0 / 9.0 * d + 7.0 * (3.0 / 40.0 * d) + 0.5 * r_out
    extra_length = (d / 2.0 + t_max_approx + r_out) * 1.1 - z_comb[-1]
    extra_n      = int(round(extra_length / dz))
    if extra_n > 0:
        z_extra   = np.linspace(0., dz * extra_n, extra_n + 1) + z_comb[-1] + dz
        phi_extra = np.zeros(extra_n + 1)
        z_comb   = np.append(z_comb, z_extra)
        phi_comb = np.append(phi_comb, phi_extra)

    return z_comb, phi_comb, d, ds


def collision_wall_width(profile, gamma: float) -> float:
    """Return the outer-minus-inner wall width when its thinning is ``gamma``."""
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1")
    return float((profile.rout_0 - profile.rin_0) / gamma)


def _spatial_step(profile, gamma: float, params: BubbleMasterParams) -> float:
    """Resolve the requested fixed-dz or points-across-wall convention."""
    if params.wall_points is not None:
        if params.dz is not None:
            raise ValueError("set either wall_points or dz, not both")
        if params.wall_points <= 0:
            raise ValueError("wall_points must be positive")
        return collision_wall_width(profile, gamma) / params.wall_points
    if params.dz is None or params.dz <= 0:
        raise ValueError("dz must be positive when wall_points is not set")
    return float(params.dz)


def _cutoff_times(d: float, cutoff_type: int, t_0_scal: float,
                  t_min_global: float = 0.0):
    # Match reference: smax=1.2*d, t_cut=0.9*smax, t_0=(smax-t_cut)/4
    smax  = 1.2 * d
    t_cut = 0.9 * smax                          # = 1.08 * d
    t_0   = 0.25 * (smax - t_cut) * t_0_scal   # = 0.03 * d
    t_m   = t_cut + t_0 / 2.0
    if cutoff_type == 0:
        t_max = t_cut + 7.0 * t_0
    else:
        t_max = t_cut + t_0
    t_max = max(t_max, t_min_global)
    return t_0, t_cut, t_m, t_max


def write_2d_setup(model, gamma: float, times: np.ndarray,
                   path: Path, params: Optional[BubbleMasterParams] = None,
                   wlist: Optional[np.ndarray] = None):
    """
    Write a BubbleMaster setup HDF5 file for a single (model, gamma) pair.

    Parameters
    ----------
    model  : PhysicsModel (provides potential + instanton profile)
    gamma  : Lorentz factor for the initial bubble separation
    times  : 1-D array of integration time points
    path   : output path for the HDF5 file
    params : BubbleMasterParams (resolution/cutoff settings)
    """
    if params is None:
        params = BubbleMasterParams()

    profile = model.instanton
    p       = params

    dz_target = _spatial_step(profile, gamma, p)
    z, phi0, d, ds = _two_bubble_ic(
        profile, gamma, dz_target, p.collision_radius
    )

    t_min_global       = float(times[-1]) if len(times) > 0 else 0.
    t_0, t_cut, t_m, t_max = _cutoff_times(d, p.cutoff_type, p.t_0_scal,
                                            t_min_global)
    smax = 1.2 * d

    n_z_half = len(z)
    dz_act   = float(z[1] - z[0])
    M        = _mass_scale(model.potential)
    if wlist is None:
        wmin  = np.pi / ((n_z_half - 1) * dz_act) / 2.0
        wmax  = min(10.0 * M, np.pi / dz_act)
        wlist = np.geomspace(wmin, wmax, p.n_w)
    else:
        wlist = np.asarray(wlist, dtype=float)
        if wlist.ndim != 1 or len(wlist) < 2:
            raise ValueError("wlist must be a one-dimensional array of length >= 2")
        if np.any(wlist <= 0.0) or np.any(np.diff(wlist) <= 0.0):
            raise ValueError("wlist must be positive and strictly increasing")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        # Named scalar attributes (replaces the old flat header[16]).
        # Integers written as int32 to match C++ NATIVE_INT; h5py 3.x would
        # otherwise default to int64 which some HDF5 builds can't convert.
        f.attrs["n_z"]          = np.int32(n_z_half)
        f.attrs["n_w"]          = np.int32(len(wlist))
        f.attrs["n_k"]          = np.int32(p.n_k)
        f.attrs["n_t"]          = np.int32(len(times))
        f.attrs["ds"]           = float(ds)
        f.attrs["how_often_ds"] = np.int32(p.how_often_ds)
        f.attrs["baby_steps"]   = np.int32(p.baby_steps)
        f.attrs["d"]            = float(d)
        f.attrs["t_0"]          = float(t_0)
        f.attrs["t_cut"]        = float(t_cut)
        f.attrs["t_m"]          = float(t_m)
        f.attrs["t_max"]        = float(t_max)
        f.attrs["smax"]         = float(smax)
        f.attrs["cutoff_type"]  = np.int32(p.cutoff_type)
        f.attrs.create("collision_radius", p.collision_radius,
                       dtype=h5py.string_dtype(encoding="ascii"))
        f.attrs["collision_r0"] = float(
            profile.rmid_0 if p.collision_radius == "mid" else profile.rout_0
        )
        f.attrs["dz_target"] = float(dz_target)
        f.attrs["wall_points_target"] = (
            float(p.wall_points) if p.wall_points is not None else np.nan
        )
        f.attrs["collision_wall_width"] = collision_wall_width(profile, gamma)

        # Potential group (read by C++ Potential::from_hdf5)
        pot_grp = f.require_group("potential")
        model.potential.to_hdf5(pot_grp)

        # Arrays
        f.create_dataset("z",     data=z,     dtype="float64")
        f.create_dataset("phi0",  data=phi0,  dtype="float64")
        f.create_dataset("wlist", data=wlist, dtype="float64")
        f.create_dataset("times", data=np.asarray(times, dtype="float64"))


def extend_2d_setup_times(setup_path: Path, new_times: np.ndarray) -> None:
    """
    Extend an already-run row's setup.h5 IN PLACE to new, later cutoff
    times, by appending ``new_times`` to its existing ``times[]`` array.

    ``bubblemaster_gpu`` re-run on this SAME setup_path/output_dir pair
    afterwards auto-detects (by counting existing ``result_*.h5`` files
    against the new, larger ``n_t``) that this row now has more requested
    times than it has results for, and extends it using the plateau
    checkpoint embedded in its current last result file -- see main.cpp's
    auto-detection and ``checkpoint_io.h``'s Format A. There is deliberately
    no separate "continuation setup" file: reusing the same setup.h5/
    output_dir is what lets that auto-detection work.

    Every entry in ``new_times`` must be strictly greater than the setup's
    current last requested time -- the plateau checkpoint only has state to
    extend forward, never to fill in or redo an earlier cutoff.

    Only ``times``/``n_t`` change. Everything else -- critically including
    ``wlist`` and ``t_0``/``t_cut``/``t_m``/``t_max``/``smax`` -- is left
    completely untouched. ``wlist`` must stay the row's ORIGINAL frequency
    grid, not one recomputed for a possibly-different GAMMA_STAR_MAX (the
    GPU binary hard-checks this against the row's own last result file and
    refuses to run otherwise -- see main.cpp). ``t_0``/``t_cut``/``t_m``/
    ``t_max``/``smax`` are the row's own fixed reference values that every
    cutoff time's windowing is shifted relative to (see
    ``gpu_integrator.cu``'s ProcessBatch); recomputing them from the new,
    longer ``times`` array would break the plateau checkpoint's validity.
    """
    new_times = np.asarray(new_times, dtype=float)
    if new_times.ndim != 1 or len(new_times) < 1:
        raise ValueError("new_times must be a non-empty 1-D array")

    with h5py.File(setup_path, "r+") as f:
        old_times = f["times"][:]
        if len(old_times) == 0:
            raise ValueError(f"{setup_path} has an empty times[] array")
        old_last = float(old_times[-1])
        if np.any(new_times <= old_last):
            raise ValueError(
                f"new_times must all be strictly greater than {setup_path}'s "
                f"current last time ({old_last:.6f}); got "
                f"min(new_times)={new_times.min():.6f}"
            )
        if np.any(np.diff(new_times) <= 0):
            raise ValueError("new_times must be strictly increasing")

        full_times = np.concatenate([old_times, new_times])
        del f["times"]
        f.create_dataset("times", data=full_times, dtype="float64")
        f.attrs["n_t"] = np.int32(len(full_times))


# ---------------------------------------------------------------------------
# 3D sledgehamr bubble injection
# ---------------------------------------------------------------------------

def _evolve_1d_field(profile, potential, t_rel: float, dr: float = 0.01,
                     n_r: int = 2000) -> tuple:
    """
    Evolve the radial field profile from the instanton to time t_rel using a
    leapfrog scheme with the 3-point spherical Laplacian.

    Returns (r, phi, pi) at time t_rel.
    """
    r   = np.arange(n_r) * dr
    phi = profile.interp(r)
    pi  = np.zeros(n_r)

    def laplace(phi_arr):
        lap    = np.zeros(n_r)
        i      = np.arange(1, n_r - 1)
        lap[i] = ((i + 1) * phi_arr[i + 1]
                  - 2 * i * phi_arr[i]
                  + (i - 1) * phi_arr[i - 1]) / (i * dr**2)
        lap[0] = 6.0 * (phi_arr[1] - phi_arr[0]) / dr**2
        return lap

    if t_rel <= 0.:
        return r, phi, pi

    dt = dr * 0.5
    n_steps = max(int(round(t_rel / dt)), 1)
    dt = t_rel / n_steps

    dv = potential.dV  # accepts numpy arrays directly

    # Leapfrog: initial half-step for pi
    pi += 0.5 * dt * (laplace(phi) - dv(phi))

    for _ in range(n_steps):
        phi += dt * pi
        pi  += dt * (laplace(phi) - dv(phi))

    # Final half-step to bring pi to the same time level as phi
    pi -= 0.5 * dt * (laplace(phi) - dv(phi))

    return r, phi, pi


def _compute_refinement_levels(r, phi, dphi, dx_base: float,
                                max_levels: int = 3) -> np.ndarray:
    """
    Compute AMR refinement levels for a 1D radial profile.
    Returns integer array of length ~ n_points with level per point.
    """
    field_interp  = interp1d(r, phi,  kind="linear", fill_value=0., bounds_error=False)
    dfield_interp = interp1d(r, dphi, kind="linear", fill_value=0., bounds_error=False)

    # Start from base lattice, trim to non-negligible field
    x_vals = np.arange(len(r)) * dx_base
    mask   = np.abs(field_interp(x_vals)) > 1e-5
    x_vals = x_vals[mask]

    ref_levels = np.zeros(len(x_vals), dtype=int)
    dx = dx_base

    for level in range(max_levels):
        phi_at  = field_interp(x_vals)
        dphi_at = dfield_interp(x_vals)

        # CFL criterion for field
        x_p  = x_vals + dx / 2.
        phi_avg = (field_interp(x_p) + phi_at) / 2.
        cfl_phi = np.abs((phi_avg - phi_at) / (np.abs(phi_at) + 10.))

        # CFL criterion for time-derivative
        dphi_avg = (dfield_interp(x_p) + dphi_at) / 2.
        cfl_dphi = np.abs((dphi_avg - dphi_at) / (np.abs(dphi_at) + 1.))

        locs = np.union1d(
            np.where(cfl_phi [::2**level] > 1e-3)[0],
            np.where(cfl_dphi[::2**level] > 1e-3)[0],
        )
        ref_levels[locs] = level + 1

        if level < max_levels - 1:
            n_fine = int(2 * len(x_vals) - 1)
            dx /= 2.
            x_vals = np.arange(n_fine) * dx

    return ref_levels


def write_3d_bubbles(model, bubble_pop, t_inject: float,
                     lattice_N: int, path: Path,
                     max_ref_levels: int = 3):
    """
    Write a sledgehamr bubble injection HDF5 file.

    This is the FIXED schema that sledgehamr reads; do not modify the
    dataset names or Header layout.

    Parameters
    ----------
    model       : PhysicsModel (potential + instanton)
    bubble_pop  : BubblePopulation
    t_inject    : coordinate time at which to inject bubbles
    lattice_N   : base lattice size (N^3 grid)
    path        : output file path
    """
    profile  = model.instanton
    pot      = model.potential
    lb       = pot.params["lambda_bar"]
    L        = bubble_pop.box_size

    # Bubbles nucleated before t_inject
    events   = [e for e in bubble_pop.events if e.t_nuc <= t_inject]
    n_b      = len(events)
    locs     = np.array([e.position for e in events])  # (n_b, 3)

    # Use a single shared profile (profile id=0) for all bubbles.
    # Profile corresponds to the OLDEST bubble (longest evolution time).
    t_rel    = t_inject - min(e.t_nuc for e in events)
    dx_base  = L / lattice_N

    # Build fine-resolution profile arrays for the injection
    r, phi_arr, dphi_arr = _evolve_1d_field(profile, pot, t_rel, dr=dx_base / 10.)

    ref_lvls = _compute_refinement_levels(
        r, phi_arr, dphi_arr, dx_base, max_ref_levels
    )

    # Fine grid for the profile output (9 points between each coarse point)
    n_coarse  = len(ref_lvls)
    Nbins     = 9 * (n_coarse - 1) + n_coarse
    dx_fine   = dx_base / 10.
    r_fine    = np.arange(Nbins) * dx_fine

    # Project coarse refinement levels onto fine grid
    ref_fine  = np.zeros(Nbins, dtype=int)
    for j in range(n_coarse):
        ref_fine[10 * j: 10 * j + 6] = ref_lvls[j]
        if j > 0:
            ref_fine[10 * j - 4: 10 * j] = ref_lvls[j]

    phi_fine  = interp1d(r, phi_arr,  kind="linear",
                         fill_value=0., bounds_error=False)(r_fine)
    dphi_fine = interp1d(r, dphi_arr, kind="linear",
                         fill_value=0., bounds_error=False)(r_fine)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        # Fixed header: [lattice_N, L, n_bubbles, GammaStar, lambda_bar]
        kin = model.kinematics
        gamma_star = float(kin.Gamma(t_inject, min(e.t_nuc for e in events)))
        f.create_dataset("Header", data=np.array(
            [lattice_N, L, n_b, gamma_star, lb]
        ))

        # Bubble locations
        f.create_dataset("xlocs", data=locs[:, 0])
        f.create_dataset("ylocs", data=locs[:, 1])
        f.create_dataset("zlocs", data=locs[:, 2])
        f.create_dataset("t",
                         data=t_inject * np.ones(n_b))
        f.create_dataset("use_profile",
                         data=np.zeros(n_b, dtype=int))

        # Single shared profile (index 0)
        f.create_dataset("profile_header_0",
                         data=np.array([Nbins, 1.0 / dx_fine,
                                        dx_fine * (Nbins - 1),
                                        int(np.max(ref_fine))]))
        f.create_dataset("profile_level_0",  data=ref_fine)
        f.create_dataset("profile_Psi1_0",   data=phi_fine)
        f.create_dataset("profile_Pi1_0",    data=dphi_fine)

        zeros = np.zeros(Nbins)
        for comp in ("u_xx", "u_yy", "u_zz", "u_xy", "u_xz", "u_yz",
                     "du_xx", "du_yy", "du_zz", "du_xy", "du_xz", "du_yz"):
            f.create_dataset(f"profile_{comp}_0", data=zeros)
