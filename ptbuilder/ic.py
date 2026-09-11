"""
Initial conditions: write setup files for BubbleMaster and sledgehamr.
"""
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
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
                   collision_radius: str = "mid", t_horizon: float = 0.):
    """
    Build the initial field profile and z grid for two colliding bubbles.

    Returns (z, phi0, d, ds):
      - z     : 1-D array of z-grid values (half-grid, centre at origin)
      - phi0  : corresponding combined field phi = sqrt(phi1^2 + phi2^2)
      - d     : centre-to-centre bubble separation
      - ds    : step size in s (Milne arc-length)

    The separation d is chosen so that the wall Lorentz factor (wall-thinning
    ratio) equals gamma exactly when the selected wall surfaces first touch.

    ``t_horizon`` is the furthest time this row will actually be integrated
    to (e.g. a shared T_MAX forced across every row of a multi-gamma scan,
    which can be far later than this row's own natural cutoff for
    small-gamma/small-d rows). The trailing vacuum padding is sized to stay
    causally ahead of whichever of the row's own estimate or t_horizon is
    larger, so a row forced to run long doesn't have its wall/signal
    reflect off the domain edge partway through.
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

    # Add trailing zeros (vacuum region beyond the outer bubble wall).
    # Sized to stay causally ahead of the LARGER of this row's own natural
    # cutoff estimate and t_horizon (how far it will actually be run) --
    # using t_max_approx alone here would undersize the domain for any row
    # forced to run past its own natural timescale, letting its wall/signal
    # reflect off the domain edge before the run finishes.
    t_max_approx = 12.0 / 9.0 * d + 7.0 * (3.0 / 40.0 * d) + 0.5 * r_out
    t_max_approx = max(t_max_approx, t_horizon)
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
    t_min_global = float(times[-1]) if len(times) > 0 else 0.
    z, phi0, d, ds = _two_bubble_ic(
        profile, gamma, dz_target, p.collision_radius, t_horizon=t_min_global
    )

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
        f.attrs["gamma_ij"] = float(gamma)

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


def generate_surrogate(model, root: Path, tag_label: str,
                       gamma_star_min: float, gamma_star_max: float,
                       kR_min: float, kR_max: float, n_omega: int,
                       extend_from: Optional[Path] = None,
                       pair_gamma_factor: float = 2.5,
                       time_factor: float = 1.4,
                       omega_min_gamma_factor: float = 1.5,
                       dgamma_ij: float = 0.4,
                       dt: float = 2.0,
                       n_k: int = 51,
                       n_min: int = 128,
                       panels_per_oscillation: int = 40,
                       wall_points: int = 20,
                       how_often_ds: int = 5,
                       s_batch_size: int = 128,
                       cutoff_type: int = 0,
                       baby_steps: int = 100):
    """
    Build (or extend) a midpoint-convention BubbleMaster GPU scan: converts
    a physical (gamma_*, kR_*, N_omega) range into the actual per-row
    ``gamma_ij`` grid, each row's own ``times``/``wlist`` arrays, and writes
    every row's setup.h5 (via write_2d_setup/extend_2d_setup_times) plus a
    ``manifest.tsv`` listing them.

    ``model`` must already be built (``model.kinematics``/``model.instanton``/
    ``model.potential``, e.g. by loading an instanton profile and picking a
    potential) -- this function only turns a range into rows, it doesn't
    choose a physics model, and doesn't assume anything about what's inside
    ``model.potential`` (different potentials may have different params).
    ``tag_label`` is a plain, caller-chosen string identifying the model in
    the scan's directory name (e.g. ``"lb0.84"`` for a phi4 potential at
    lambda_bar=0.84) -- purely cosmetic, not read back from anywhere.

    extend_from=None builds a fresh scan under
    ``root/data/runtime_scan_{tag_label}_gs{gamma_star_min}-{gamma_star_max}
    _kR{kR_min}-{kR_max}_nw{n_omega}/``.

    extend_from=<existing runtime_scan_.../ directory> extends that scan's
    GAMMA_STAR_MAX to gamma_star_max instead (which must be greater than the
    old scan's own GAMMA_STAR_MAX, and at most omega_min_gamma_factor times
    it -- the frequency-grid headroom baked in when the old scan was built).
    gamma_star_min/kR_min/kR_max/n_omega/pair_gamma_factor/time_factor/
    omega_min_gamma_factor are then read back from the old scan's own attrs
    and OVERRIDE the arguments passed in here -- only gamma_star_max (the
    new target) is actually used from the arguments in that case. Existing
    rows get their times[] extended in place (wlist/geometry untouched, so
    their already-computed results -- and the plateau checkpoint embedded in
    each row's last result file -- remain valid seeds for the GPU binary's
    own auto-detection); newly-in-range rows are written fresh. The scan
    directory is renamed to match the new gamma_star_max.

    Returns a SimpleNamespace: root, tag, scan_root, setup_dir, output_root,
    manifest, manifest_rows (list of (index, gamma_ij, n_t, setup_path,
    name)), gamma_ij_grid, n_old, model, params, plus n_min/
    panels_per_oscillation/s_batch_size (needed to build a bubblemaster_gpu
    command line) and
    t_max/r_star_max/omega_min (for a walltime estimate).
    """
    root = Path(root)
    if not (0 < kR_min < kR_max):
        raise ValueError("Require 0 < kR_min < kR_max")
    if not (1.0 <= gamma_star_min <= gamma_star_max):
        raise ValueError("Require 1 <= gamma_star_min <= gamma_star_max")
    if n_omega < 2:
        raise ValueError("n_omega must be >= 2")

    kin = model.kinematics

    def time_at_gamma(gamma):
        if gamma <= 1.0:
            return 0.0
        return brentq(lambda t: float(kin.Gamma(t)) - gamma, 0.0, 1.0e7)

    def rstar_at_gamma(gamma):
        t = time_at_gamma(gamma)
        return 2.0 * float(kin.R(t, r="mid"))

    # --- If extending: read the old scan's actual rows straight from its
    # setups/*.h5 files (never from manifest.tsv, which is a write-only
    # convenience index for the SLURM script and can drift out of sync with
    # what's actually on disk -- see write_2d_setup's "gamma_ij" attr) and
    # its locked scan-definition attrs, overriding the arguments above with
    # them. gamma_ij is never recomputed from a fresh np.linspace over a
    # larger gamma_ij_max, which would silently shift EVERY row's gamma_ij,
    # not just add new ones.
    old_rows = None
    old_gamma_star_max = None
    if extend_from is not None:
        extend_from = Path(extend_from)
        old_rows = []
        for setup_path in (extend_from / "setups").glob("*.h5"):
            with h5py.File(setup_path, "r") as f:
                old_rows.append(SimpleNamespace(name=setup_path.stem,
                                                gamma_ij=float(f.attrs["gamma_ij"])))
        old_rows.sort(key=lambda r: r.gamma_ij)

        with h5py.File(extend_from / "setups" / f"{old_rows[-1].name}.h5", "r") as f:
            old_gamma_star_max     = float(f.attrs["gamma_star_max"])
            gamma_star_min         = float(f.attrs["gamma_star_min"])
            kR_min                 = float(f.attrs["kR_min"])
            kR_max                 = float(f.attrs["kR_max"])
            pair_gamma_factor      = float(f.attrs["pair_gamma_factor"])
            time_factor            = float(f.attrs["time_factor"])
            omega_min_gamma_factor = float(f.attrs["omega_min_gamma_factor"])
            n_omega                = int(f.attrs["n_w"])

        if gamma_star_max <= old_gamma_star_max:
            raise ValueError(
                f"gamma_star_max ({gamma_star_max}) must be greater than the scan being "
                f"extended's GAMMA_STAR_MAX ({old_gamma_star_max}) to extend it")
        if gamma_star_max > omega_min_gamma_factor * old_gamma_star_max:
            raise ValueError(
                f"gamma_star_max ({gamma_star_max}) exceeds the old scan's frequency "
                f"headroom ({omega_min_gamma_factor:g}x{old_gamma_star_max:g}="
                f"{omega_min_gamma_factor*old_gamma_star_max:.3f}) -- extending this far "
                f"would need new low-omega bins, which is not supported yet")

    gamma_ij_max = pair_gamma_factor * gamma_star_max
    r_star_max   = rstar_at_gamma(gamma_star_max)
    t_max        = time_factor * r_star_max

    # OMEGA_MIN is anchored to a gamma_star omega_min_gamma_factor times
    # larger than gamma_star_max (not gamma_star_max itself), so a future
    # scan that raises gamma_star_max by up to that factor can reuse this
    # scan's low-omega bins without hitting an empty range.
    r_star_omega_anchor = rstar_at_gamma(omega_min_gamma_factor * gamma_star_max)
    omega_min = kR_min / r_star_omega_anchor

    if old_rows is not None:
        old_gamma_ij_grid = np.array([r.gamma_ij for r in old_rows])
        last_old = old_gamma_ij_grid[-1]
        n_new = int(np.ceil((gamma_ij_max - last_old) / dgamma_ij))
        new_gamma_ij = last_old + dgamma_ij * np.arange(1, n_new + 1)
        gamma_ij_grid = np.concatenate([old_gamma_ij_grid, new_gamma_ij])
    else:
        n_gamma = int(np.ceil((gamma_ij_max - gamma_star_min) / dgamma_ij)) + 1
        gamma_ij_grid = np.linspace(gamma_star_min, gamma_ij_max, n_gamma)

    # --- Directory: folder name encodes the scan's physical inputs. ---
    tag = (f"{tag_label}_gs{gamma_star_min:g}-{gamma_star_max:g}"
           f"_kR{kR_min:g}-{kR_max:g}_nw{n_omega}")
    scan_root = root / "data" / f"runtime_scan_{tag}"

    n_old = 0
    if old_rows is not None:
        n_old = len(old_rows)
        if scan_root != extend_from:
            if scan_root.exists():
                raise FileExistsError(
                    f"{scan_root} already exists -- refusing to overwrite it while "
                    f"extending {extend_from.name}")
            extend_from.rename(scan_root)

    setup_dir   = scan_root / "setups"
    output_root = scan_root / "gpu"
    setup_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)

    params = BubbleMasterParams(
        dz=None, wall_points=wall_points, n_w=n_omega, n_k=n_k,
        how_often_ds=how_often_ds, baby_steps=baby_steps, cutoff_type=cutoff_type,
        t_0_scal=1.0, collision_radius="mid",
    )

    manifest_rows = []
    for index, gamma_ij in enumerate(gamma_ij_grid):
        # Plain sequential row names, not gamma_ij encoded into the name:
        # gamma_ij now lives in the row's own files (write_2d_setup's
        # "gamma_ij" attr, and every result_*.h5's own attr), so the name is
        # just an opaque on-disk id -- no precision-collision risk on a fine
        # grid, and gaps in the sequence make missing rows obvious at a glance.
        name = old_rows[index].name if index < n_old else f"row_{index:04d}"
        setup_path = setup_dir / f"{name}.h5"

        if index < n_old:
            # Existing row: extend its times[] in place to the new, larger
            # T_MAX -- wlist/geometry/t_cut/t_m/t_max/smax stay exactly as
            # they were, so the row's already-computed result_*.h5 files
            # (and the plateau checkpoint embedded in the last one) remain
            # valid seeds for the GPU binary's own auto-detection.
            with h5py.File(setup_path, "r") as f:
                old_last_time = float(f["times"][-1])
                old_n_t = int(f.attrs["n_t"])
            assert old_last_time < t_max, (
                f"{setup_path.name}: existing last time {old_last_time:.3f} already >= "
                f"new T_MAX={t_max:.3f} -- T_MAX should strictly grow with gamma_star_max")
            n_new_t = int(np.ceil((t_max - old_last_time) / dt))
            new_times = np.linspace(old_last_time, t_max, n_new_t + 1)[1:]
            extend_2d_setup_times(setup_path, new_times)
            n_t = old_n_t + len(new_times)
        else:
            # New row (gamma_ij beyond the old scan's range, or every row
            # for a from-scratch scan): fresh.
            t_first = time_at_gamma(gamma_ij)
            if t_first >= t_max:
                raise RuntimeError(f"Collision time exceeds T_MAX at gamma_ij={gamma_ij}")
            n_t = int(np.ceil((t_max - t_first) / dt)) + 1
            times = np.linspace(t_first, t_max, n_t)

            smallest_target = max(gamma_star_min, gamma_ij / pair_gamma_factor)
            omega_max = kR_max / rstar_at_gamma(smallest_target)
            if omega_max <= omega_min:
                raise RuntimeError(f"Empty omega range at gamma_ij={gamma_ij}")
            omega = np.geomspace(omega_min, omega_max, n_omega)

            write_2d_setup(model, gamma_ij, times, setup_path, params, wlist=omega)
            with h5py.File(setup_path, "a") as handle:
                handle.attrs["gamma_star_min"]         = gamma_star_min
                handle.attrs["gamma_star_max"]         = gamma_star_max
                handle.attrs["kR_min"]                 = kR_min
                handle.attrs["kR_max"]                 = kR_max
                handle.attrs["pair_gamma_factor"]      = pair_gamma_factor
                handle.attrs["time_factor"]            = time_factor
                handle.attrs["omega_min_gamma_factor"] = omega_min_gamma_factor
                handle.attrs["R_star_max"]              = r_star_max

        manifest_rows.append((index, gamma_ij, n_t, setup_path.relative_to(root), name))

    manifest = scan_root / "manifest.tsv"
    manifest.write_text(
        "index\tgamma_ij\tn_t\tsetup\tname\n" +
        "".join(f"{i}\t{g:.12g}\t{nt}\t{s}\t{n}\n" for i, g, nt, s, n in manifest_rows)
    )

    return SimpleNamespace(
        root=root, tag=tag, scan_root=scan_root, setup_dir=setup_dir,
        output_root=output_root, manifest=manifest, manifest_rows=manifest_rows,
        gamma_ij_grid=gamma_ij_grid, n_old=n_old, model=model, params=params,
        n_min=n_min, panels_per_oscillation=panels_per_oscillation, s_batch_size=s_batch_size,
        t_max=t_max, r_star_max=r_star_max, omega_min=omega_min,
    )


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
