"""
Bootstrap GW spectrum: combine 2D scan results with collision weights.
"""
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator


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
    """
    with h5py.File(path, "r") as f:
        flat   = f["weights"][:]
        pair_i = f["pair_i"][:].astype(int)
        pair_j = f["pair_j"][:].astype(int)
    n_coll = len(flat) // n_t
    return flat.reshape(n_coll, n_t), pair_i, pair_j


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
                       lambda_bar: Optional[float] = None) -> tuple:
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
            t_c      = kin.collision_time(d, events[bi].t_nuc, events[bj].t_nuc)
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
