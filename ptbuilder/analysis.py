"""
Bootstrap GW spectrum: combine 2D scan results with collision weights.
"""
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

_ASSETS_DIR = Path(__file__).parent / "assets"
_KEFF_PATH  = _ASSETS_DIR / "keff.npy"

# ---------------------------------------------------------------------------
# keff² correction (lattice momentum correction for sledgehamr output)
# ---------------------------------------------------------------------------

_keffsq_cache: Optional[dict] = None


def _compute_keff(dim: int) -> np.ndarray:
    """Compute mean |k| per integer shell for a cubic lattice of side `dim`."""
    kmax  = int(np.sqrt(3) / 2 * dim + 0.5) + 1
    k_sum = np.zeros(kmax)
    count = np.zeros(kmax)
    b = np.fft.fftfreq(dim) * dim
    c = np.fft.fftfreq(dim) * dim
    B2C2 = b[:, None] ** 2 + c[None, :] ** 2
    for a in range(dim // 2 + 1):
        ka   = a if a < dim // 2 else a - dim
        mult = 1.0 if (a == 0 or a == dim // 2) else 2.0
        k_mag = np.sqrt(ka * ka + B2C2)
        k_bin = np.floor(k_mag + 0.5).astype(np.int64)
        k_sum += np.bincount(k_bin.ravel(), weights=(mult * k_mag).ravel(),    minlength=kmax)
        count += np.bincount(k_bin.ravel(), weights=np.full(k_bin.size, mult), minlength=kmax)
    return k_sum / count


def apply_keffsq(k_int: np.ndarray) -> np.ndarray:
    """
    Return the keff² correction array for a sledgehamr k array.

    The box size N is inferred from the number of k bins. If the entry is not
    yet cached in ptbuilder/data/keff.npy it is computed on the fly and saved
    for future calls. Multiply k_int by sqrt(apply_keffsq(k_int)) to get the
    corrected k axis.
    """
    global _keffsq_cache
    if _keffsq_cache is None:
        _keffsq_cache = (np.load(_KEFF_PATH, allow_pickle=True).item()
                         if _KEFF_PATH.exists() else {})
    N = 2 ** int(round(np.log2(2 * (len(k_int) - 1) / np.sqrt(3))))
    if N not in _keffsq_cache:
        _keffsq_cache[N] = _compute_keff(N) ** 2
        np.save(_KEFF_PATH, _keffsq_cache)
    return _keffsq_cache[N]


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def sledgehamr_normalization(k_sq: np.ndarray, spectrum: np.ndarray,
                              L: float, zero_pad: float = 1.) -> tuple:
    """
    Apply sledgehamr normalization to convert raw output to dE/dlnk.

    Returns (k, dE_dlnk) arrays.
    """
    Leff = L * zero_pad
    kIR  = 2.0 * np.pi / Leff
    k    = np.sqrt(k_sq) * kIR
    r    = np.sqrt(k_sq)
    kL   = 2.0 * np.pi * r
    norm = 4.0 * kL * Leff**3
    return k, spectrum * norm


def bubblemaster_normalization(lambda_bar: float) -> float:
    """
    Overall prefactor relating BubbleMaster output to dE/dlnk in physical units.

    kconv = (1/3) * sqrt(2 * lambda_bar)
    norm  = 432 * sqrt(2*lambda_bar) / (3 + sqrt(9-8*lambda_bar))^4
    """
    kconv = 1.0 / 3.0 * np.sqrt(2.0 * lambda_bar)
    up    = 3.0 + np.sqrt(9.0 - 8.0 * lambda_bar)
    norm  = 432.0 * np.sqrt(2.0 * lambda_bar) / up**4
    return kconv, norm


# ---------------------------------------------------------------------------
# 2D scan interpolator
# ---------------------------------------------------------------------------

def build_scan_interpolator(scan_results: Dict[float, "ScanResult"],
                             k_out: Optional[np.ndarray] = None):
    """
    Build a RegularGridInterpolator over (gamma, time) -> spectrum.

    Parameters
    ----------
    scan_results : dict from run_scan()
    k_out        : optional common k grid to resample spectra onto

    Returns
    -------
    interp_s : RegularGridInterpolator mapping (gamma, t) -> spectrum [n_k]
    gammas   : sorted gamma array
    times    : time array (from first result)
    k_out    : common k grid used for interpolation
    """
    from .scan import ScanResult

    gammas = np.array(sorted(scan_results.keys()))
    times  = next(iter(scan_results.values())).times

    # Collect spectra; resample onto k_out if provided
    first_w = next(iter(scan_results.values())).w
    if k_out is None:
        k_out = first_w

    Ng, Nt, Nk = len(gammas), len(times), len(k_out)
    spec_grid = np.zeros((Ng, Nt, Nk))

    for ig, gamma in enumerate(gammas):
        res = scan_results[gamma]
        for it in range(Nt):
            spec_grid[ig, it] = np.interp(k_out, res.w, res.spectrum[it],
                                           left=0., right=0.)

    interp_s = RegularGridInterpolator(
        (gammas, times), spec_grid, bounds_error=False, fill_value=0.
    )
    return interp_s, gammas, times, k_out


# ---------------------------------------------------------------------------
# Weights I/O
# ---------------------------------------------------------------------------

def load_weights(path: Path, n_t: int) -> tuple:
    """
    Load the collision weights written by the weights binary.

    Returns
    -------
    weights : (n_collisions, n_t) array of geometric weights
    pair_i  : (n_collisions,) integer index of first bubble in each pair
    pair_j  : (n_collisions,) integer index of second bubble in each pair
    gammas  : (n_collisions,) Lorentz factor per pair, or None if not in file
    """
    with h5py.File(path, "r") as f:
        if "collision_radius" not in f.attrs:
            raise ValueError(
                f"{path} has no collision_radius metadata; regenerate its "
                "weights with the midpoint convention"
            )
        if str(f.attrs["collision_radius"]) != "mid":
            raise ValueError(
                f"{path} does not use the required midpoint convention"
            )
        flat   = f["weights"][:]
        pair_i = f["pair_i"][:].astype(int)
        pair_j = f["pair_j"][:].astype(int)
        gammas = f["gamma"][:] if "gamma" in f else None
    n_coll = len(flat) // n_t
    return flat.reshape(n_coll, n_t), pair_i, pair_j, gammas


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_spectrum(scan_results: Dict[float, "ScanResult"],
                       bubble_pop,
                       weights: np.ndarray,
                       pair_i: np.ndarray,
                       pair_j: np.ndarray,
                       t_range: Optional[tuple] = None,
                       k_out: Optional[np.ndarray] = None,
                       lambda_bar: Optional[float] = None,
                       collision_radius: str = "mid") -> tuple:
    """
    Bootstrap the 3D GW spectrum from the 2D scan + collision weights.

    For each colliding pair (i, j) identified by the weights binary:
      - Compute gamma at the analytical collision time via bubble_pop.kinematics.
      - Interpolate the 2D scan spectrum at that gamma.
      - Accumulate: sum_t w[pair, t] * ΔE(gamma, t).

    Parameters
    ----------
    scan_results : dict gamma -> ScanResult from run_scan()
    bubble_pop   : BubblePopulation (must have been generated with kinematics stored)
    weights      : (n_collisions, n_t) array  \\
    pair_i       : (n_collisions,) bubble indices > from load_weights()
    pair_j       : (n_collisions,) bubble indices /
    t_range      : (t_min, t_max) slice of times to use (defaults to full range)
    k_out        : output k grid (defaults to common w grid)
    lambda_bar   : if given, applies BubbleMaster normalization
    collision_radius : wall surface defining collision (``"mid"`` or ``"out"``)

    Returns
    -------
    k_out : 1-D array of wavenumbers
    spec  : 1-D array of dE/dlnk (same length as k_out)
    """
    if bubble_pop.kinematics is None:
        raise ValueError(
            "bubble_pop.kinematics is None. "
            "Generate the population with BubblePopulation.generate()."
        )

    interp_s, gammas, times, k_out = build_scan_interpolator(
        scan_results, k_out
    )

    if t_range is not None:
        locs = np.where((times >= t_range[0]) & (times < t_range[1]))[0]
    else:
        locs = np.arange(len(times))

    t_comp = times[locs]
    if len(t_comp) < 2:
        raise ValueError("t_range covers fewer than 2 time points.")

    n_coll  = len(pair_i)
    events  = bubble_pop.events
    kin     = bubble_pop.kinematics
    combined = np.zeros(len(k_out))

    for idx in range(n_coll):
        bi, bj = int(pair_i[idx]), int(pair_j[idx])
        d = bubble_pop._separation(events[bi].position,
                                   events[bj].position,
                                   bubble_pop.box_size)
        try:
            t_c      = kin.collision_time(
                d, events[bi].t_nuc, events[bj].t_nuc,
                collision_radius=collision_radius,
            )
            gamma_ij = float(kin.Gamma(t_c, events[bi].t_nuc))
        except Exception:
            continue

        g = float(np.clip(gamma_ij, gammas.min(), gammas.max()))
        w = weights[idx][locs]

        # Evaluate spectrum at all time points for this gamma at once
        pts   = np.column_stack([np.full(len(t_comp), g), t_comp])  # (n_t, 2)
        specs = interp_s(pts)      # (n_t, n_k)
        # Accumulate w[t] * ΔE(gamma, t) for each time interval
        delta_specs = np.diff(specs, axis=0)           # (n_t-1, n_k)
        combined += (w[:len(t_comp) - 1, None] * delta_specs).sum(axis=0)

    if lambda_bar is not None:
        _, norm = bubblemaster_normalization(lambda_bar)
        combined *= norm

    return k_out, combined


# ---------------------------------------------------------------------------
# Weights input writer
# ---------------------------------------------------------------------------

def write_weights_input(path: Path, positions: np.ndarray, t: np.ndarray,
                         kinematics, L: float, t_init: float = 0.,
                         collision_radius: str = "mid") -> None:
    """
    Write the input HDF5 expected by the cpp/weights binary.

    R(t) is computed for the selected collision surface (``"mid"`` or
    ``"out"``).  The convention and its initial radius are stored in the file
    so the C++ calculation uses precisely the same geometry.
    rout_0 and rin_0 are stored as attributes so the binary can compute
    the Lorentz factor gamma analytically for each colliding pair.

    Parameters
    ----------
    path       : output file path
    positions  : (n_b, 3) bubble positions in [0, L)
    t          : time array (n_t,)
    kinematics : BubbleKinematics instance
    L          : box side length
    t_init     : nucleation time (default 0)
    collision_radius : wall surface defining collision (``"mid"`` or ``"out"``)
    """
    positions = np.asarray(positions, dtype=float)
    t         = np.asarray(t,         dtype=float)
    n_b, n_t  = len(positions), len(t)
    if collision_radius not in ("mid", "out"):
        raise ValueError("collision_radius must be 'mid' or 'out'")
    R         = kinematics.R(t, t_init=t_init, r=collision_radius)
    rout_0    = float(kinematics.profile.rout_0)
    rin_0     = float(kinematics.profile.rin_0)
    rmid_0    = float(kinematics.profile.rmid_0)
    collision_r0 = rmid_0 if collision_radius == "mid" else rout_0

    with h5py.File(path, "w") as f:
        f.attrs["L"]      = float(L)
        f.attrs["n_t"]    = int(n_t)
        f.attrs["n_b"]    = int(n_b)
        f.attrs["rout_0"] = rout_0
        f.attrs["rin_0"]  = rin_0
        f.attrs["rmid_0"] = rmid_0
        f.attrs.create("collision_radius", collision_radius,
                       dtype=h5py.string_dtype(encoding="ascii"))
        f.attrs["collision_r0"] = collision_r0
        f.create_dataset("t",     data=t)
        f.create_dataset("R",     data=np.asarray(R, dtype=float))
        f.create_dataset("xlocs", data=positions[:, 0])
        f.create_dataset("ylocs", data=positions[:, 1])
        f.create_dataset("zlocs", data=positions[:, 2])


# ---------------------------------------------------------------------------
# Convenience: load + combine sledgehamr spectrum across time slices
# ---------------------------------------------------------------------------

def sledgehamr_spectrum_range(output, t_min: float, t_max: float,
                               L: float, zero_pad: float = 1.,
                               agg: str = "mean") -> tuple:
    """
    Aggregate pySledgehamr GW spectrum slices between t_min and t_max.

    Parameters
    ----------
    output    : pySledgehamr Output object
    t_min/max : time window
    L, zero_pad : box params for normalization
    agg       : "mean", "min", "max"

    Returns (k, dE_dlnk_agg).
    """
    all_k    = []
    all_spec = []
    for sl in range(output.GetNumberOfSnapshots()):
        data = output.GetGravitationalWaveSpectrum(sl)
        if data["t"] < t_min or data["t"] > t_max:
            continue
        k, s = sledgehamr_normalization(data["k_sq"], data["spectrum"],
                                        L, zero_pad)
        all_k.append(k)
        all_spec.append(s)

    if not all_spec:
        raise ValueError(f"No snapshots found in [{t_min}, {t_max}].")

    all_spec = np.array(all_spec)
    k = all_k[0]
    if   agg == "mean": return k, all_spec.mean(axis=0)
    elif agg == "min":  return k, all_spec.min(axis=0)
    elif agg == "max":  return k, all_spec.max(axis=0)
    else: raise ValueError(f"Unknown agg={agg!r}. Use 'mean', 'min', or 'max'.")
