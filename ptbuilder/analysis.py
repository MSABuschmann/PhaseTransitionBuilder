"""
Bootstrap GW spectrum: combine 2D scan results with collision weights.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    gammas  : (n_collisions,) Lorentz factor per pair, or None if not in file
    """
    with h5py.File(path, "r") as f:
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
# Weights input writer
# ---------------------------------------------------------------------------

def write_weights_input(path: Path, positions: np.ndarray, t: np.ndarray,
                         kinematics, L: float, t_init: float = 0.) -> None:
    """
    Write the input HDF5 expected by the cpp/weights binary.

    R(t) is computed using the outer wall radius (rout_0), consistent with
    the collision_time / Gamma definitions throughout the codebase.
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
    """
    positions = np.asarray(positions, dtype=float)
    t         = np.asarray(t,         dtype=float)
    n_b, n_t  = len(positions), len(t)
    R         = kinematics.R(t, t_init=t_init, r='out')
    rout_0    = float(kinematics.profile.rout_0)
    rin_0     = float(kinematics.profile.rin_0)

    with h5py.File(path, "w") as f:
        f.attrs["L"]      = float(L)
        f.attrs["n_t"]    = int(n_t)
        f.attrs["n_b"]    = int(n_b)
        f.attrs["rout_0"] = rout_0
        f.attrs["rin_0"]  = rin_0
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


# ---------------------------------------------------------------------------
# Amplitude I/O  (requires --save-amplitude BubbleMaster output)
# ---------------------------------------------------------------------------

def load_amplitude_scan(output_dir: Path) -> dict:
    """
    Read all result_*.h5 files written with --save-amplitude and assemble
    the complex amplitude scan.

    Returns
    -------
    dict with keys:
      'w'       : (n_w,)          frequency grid
      'k'       : (n_k,)          cos(theta) grid, linspace(0,1,n_k)
      'spectrum': (n_t, n_w)      direction-integrated power spectrum
      'amp_re'  : (n_t, n_w, n_k) Re A(w, cos_theta) at each time step
      'amp_im'  : (n_t, n_w, n_k) Im A(w, cos_theta)
    """
    output_dir = Path(output_dir)
    result_files = sorted(output_dir.glob("result_*.h5"),
                          key=lambda p: int(p.stem.split("_")[1]))
    if not result_files:
        raise FileNotFoundError(f"No result_*.h5 files in {output_dir}")

    w = k = None
    spec_list, re_list, im_list = [], [], []
    for fpath in result_files:
        with h5py.File(fpath, "r") as f:
            if "amp_re" not in f:
                raise KeyError(
                    f"{fpath} has no 'amp_re' dataset — "
                    "was BubbleMaster run with --save-amplitude?"
                )
            if w is None:
                w = f["w"][:]
                k = f["k"][:]
            spec_list.append(f["spectrum"][:])
            re_list.append(f["amp_re"][:])   # (n_w, n_k)
            im_list.append(f["amp_im"][:])

    return {
        "w":        w,
        "k":        k,
        "spectrum": np.array(spec_list),    # (n_t, n_w)
        "amp_re":   np.array(re_list),      # (n_t, n_w, n_k)
        "amp_im":   np.array(im_list),      # (n_t, n_w, n_k)
    }


def check_amplitude_consistency(amp_scan: dict, rtol: float = 1e-4) -> bool:
    """
    Self-consistency check: verify that squaring the stored amplitude and
    direction-integrating recovers the stored spectrum exactly.

    Uses the same quadrature as BubbleMaster:
      spectrum[i_w] = sum_k fk * dk * (re^2 + im^2) * w^3 * 2*pi
    where fk=1 at the endpoints and fk=2 for interior points.

    Returns True if all values agree within rtol, else raises AssertionError.
    """
    w       = amp_scan["w"]          # (n_w,)
    k       = amp_scan["k"]          # (n_k,)
    spec    = amp_scan["spectrum"]   # (n_t, n_w)
    amp_re  = amp_scan["amp_re"]     # (n_t, n_w, n_k)
    amp_im  = amp_scan["amp_im"]     # (n_t, n_w, n_k)

    n_k = len(k)
    dk  = k[1] - k[0]
    fk  = np.full(n_k, 2.0)
    fk[0] = fk[-1] = 1.0

    # direction-integrate: (n_t, n_w)
    power = (amp_re**2 + amp_im**2)          # (n_t, n_w, n_k)
    spec_recon = (power * fk[None, None, :]).sum(axis=-1) * dk * w[None, :]**3 * 2 * np.pi

    max_err = np.abs(spec_recon - spec).max()
    max_val = np.abs(spec).max()
    rel_err = max_err / max_val if max_val > 0 else max_err

    if rel_err > rtol:
        raise AssertionError(
            f"Amplitude self-consistency FAILED: max relative error = {rel_err:.2e} "
            f"(threshold {rtol:.2e}). Check C++ output."
        )
    print(f"Amplitude self-consistency OK: max rel error = {rel_err:.2e}")
    return True


# ---------------------------------------------------------------------------
# Coherent reconstruction
# ---------------------------------------------------------------------------

def coherent_reconstruct(
    amp_scan_grid: dict,
    gammas_grid:   np.ndarray,
    scan_times:    np.ndarray,
    weights:       np.ndarray,
    weights_times: np.ndarray,
    gammas_pairs:  np.ndarray,
    pair_axes:     np.ndarray,
    pair_centers:  np.ndarray,
    lambda_bar:    Optional[float] = None,
    n_theta:       int = 64,
    n_phi:         int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coherent GW reconstruction accounting for inter-bubble interference.

    For each pair p, compute the effective weighted complex amplitude:
      A_p_eff(w, k) = sum_t w_p(t) * delta_amp(gamma_p, t, w, k)
    then coherently sum over pairs for each sphere direction hat_k, and
    integrate |A_total|^2 over the full sphere.

    Parameters
    ----------
    amp_scan_grid  : output of build_amplitude_scan_grid()
    gammas_grid    : (n_gamma,) scan gamma values
    scan_times     : (n_t,) BM scan time points
    weights        : (n_pairs, n_t_w) pair weight time series
    weights_times  : (n_t_w,) time axis for weights
    gammas_pairs   : (n_pairs,) Lorentz factor for each pair
    pair_axes      : (n_pairs, 3) collision axis unit vectors (normalised)
    pair_centers   : (n_pairs, 3) collision midpoint positions
    lambda_bar     : if given, apply BubbleMaster normalisation
    n_theta, n_phi : sphere quadrature resolution

    Returns
    -------
    w     : (n_w,) frequency array
    P_coh : (n_w,) coherent GW power spectrum
    """
    w       = amp_scan_grid["w"]             # (n_w,)
    k       = amp_scan_grid["k"]             # (n_k,)
    amp_re  = amp_scan_grid["amp_re"]        # (n_gamma, n_t, n_w, n_k)
    amp_im  = amp_scan_grid["amp_im"]

    n_pairs = len(gammas_pairs)
    n_w     = len(w)
    n_k     = len(k)
    n_t     = len(scan_times)

    # ------------------------------------------------------------------
    # 1. For each pair: weighted sum of incremental complex amplitude
    # ------------------------------------------------------------------
    A_eff_re = np.zeros((n_pairs, n_w, n_k))
    A_eff_im = np.zeros((n_pairs, n_w, n_k))

    for p in range(n_pairs):
        gamma_p = float(np.clip(gammas_pairs[p], gammas_grid.min(), gammas_grid.max()))
        # Interpolate amplitude grid at this gamma (linear in log-gamma space)
        ig  = int(np.searchsorted(gammas_grid, gamma_p, side="right")) - 1
        ig  = int(np.clip(ig, 0, len(gammas_grid) - 2))
        t   = (gamma_p - gammas_grid[ig]) / (gammas_grid[ig + 1] - gammas_grid[ig])
        ar  = (1 - t) * amp_re[ig] + t * amp_re[ig + 1]   # (n_t, n_w, n_k)
        ai  = (1 - t) * amp_im[ig] + t * amp_im[ig + 1]

        # Weights resampled onto scan_times
        wp = np.interp(scan_times, weights_times, weights[p])  # (n_t,)

        # Incremental amplitude delta_amp[t] = amp[t] - amp[t-1]
        dar = np.diff(ar, axis=0, prepend=0.)  # (n_t, n_w, n_k)
        dai = np.diff(ai, axis=0, prepend=0.)

        A_eff_re[p] = (wp[:, None, None] * dar).sum(axis=0)
        A_eff_im[p] = (wp[:, None, None] * dai).sum(axis=0)

    # ------------------------------------------------------------------
    # 2. Sphere quadrature: Gauss–Legendre in cos(theta), uniform in phi
    # ------------------------------------------------------------------
    from numpy.polynomial.legendre import leggauss
    cos_nodes, gl_weights = leggauss(n_theta)          # in [-1, 1]
    phi_vals = np.linspace(0., 2. * np.pi, n_phi, endpoint=False)
    d_phi    = 2. * np.pi / n_phi

    # hat_k unit vectors: (n_theta * n_phi, 3)
    CT, PH = np.meshgrid(cos_nodes, phi_vals, indexing="ij")   # (n_theta, n_phi)
    ST     = np.sqrt(np.maximum(1. - CT**2, 0.))
    hat_k  = np.stack([ST * np.cos(PH), ST * np.sin(PH), CT], axis=-1)
    hat_k  = hat_k.reshape(-1, 3)                              # (N_sphere, 3)
    d_omega = np.outer(gl_weights, np.full(n_phi, d_phi)).ravel()  # (N_sphere,)

    # Collision axes and centres: (n_pairs, 3)
    hat_n = np.asarray(pair_axes,   dtype=float)
    x_p   = np.asarray(pair_centers, dtype=float)

    # cos(theta_p) for all (pair, sphere_dir) combinations: (n_pairs, N_sphere)
    cos_th = np.abs(hat_n @ hat_k.T)          # (n_pairs, N_sphere), clip to [0,1]
    cos_th = np.clip(cos_th, 0., 1.)

    # ------------------------------------------------------------------
    # 3. Direction-integrate |sum_p A_p * exp(i phi_p)|^2
    # ------------------------------------------------------------------
    P_coh = np.zeros(n_w)

    # Process one sphere direction at a time to keep memory bounded.
    # Vectorising over pairs and frequencies; looping over directions.
    for j in range(len(hat_k)):
        A_tot_re = np.zeros(n_w)
        A_tot_im = np.zeros(n_w)

        for p in range(n_pairs):
            ct = cos_th[p, j]
            # Interpolate amplitude at this cos_theta for all frequencies
            # A_eff_re[p, n_w, n_k] -> (n_w,) at cos_theta = ct
            Ap_re = np.array([np.interp(ct, k, A_eff_re[p, iw]) for iw in range(n_w)])
            Ap_im = np.array([np.interp(ct, k, A_eff_im[p, iw]) for iw in range(n_w)])

            # Position phase: w * hat_k · x_p
            phase     = w * float(hat_k[j] @ x_p[p])
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)

            A_tot_re += Ap_re * cos_phase - Ap_im * sin_phase
            A_tot_im += Ap_re * sin_phase + Ap_im * cos_phase

        P_coh += (A_tot_re**2 + A_tot_im**2) * w**3 * 2. * np.pi * d_omega[j]

    if lambda_bar is not None:
        _, norm = bubblemaster_normalization(lambda_bar)
        P_coh *= norm

    return w, P_coh


def build_amplitude_scan_grid(
    scan_results_amplitude: dict,
    gammas_grid: np.ndarray,
) -> dict:
    """
    Stack per-gamma amplitude scan results into 4-D grids ready for
    coherent_reconstruct().

    Parameters
    ----------
    scan_results_amplitude : dict mapping gamma -> load_amplitude_scan() output
    gammas_grid            : (n_gamma,) sorted gamma values (keys of above dict)

    Returns
    -------
    dict with 'w', 'k', 'amp_re'[n_gamma, n_t, n_w, n_k], 'amp_im'[...]
    """
    gammas_sorted = np.sort(gammas_grid)
    first = scan_results_amplitude[float(gammas_sorted[0])]
    n_t, n_w, n_k = first["amp_re"].shape

    amp_re_grid = np.zeros((len(gammas_sorted), n_t, n_w, n_k))
    amp_im_grid = np.zeros_like(amp_re_grid)

    for ig, gamma in enumerate(gammas_sorted):
        res = scan_results_amplitude[float(gamma)]
        amp_re_grid[ig] = res["amp_re"]
        amp_im_grid[ig] = res["amp_im"]

    return {
        "w":      first["w"],
        "k":      first["k"],
        "amp_re": amp_re_grid,
        "amp_im": amp_im_grid,
    }
