"""
Bootstrap GW spectrum: combine 2D scan results with collision weights.
"""
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

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


def rho_vac_bar(lambda_bar: float) -> float:
    """False-minus-true vacuum energy in BubbleMaster dimensionless units."""
    upsilon = 3.0 + np.sqrt(9.0 - 8.0 * lambda_bar)
    return -(0.5 - upsilon / (4.0 * lambda_bar) + upsilon**2 / (32.0 * lambda_bar))


def to_chw(dE_dlnk: np.ndarray, rstar: float, lambda_bar: float,
          volume: float) -> np.ndarray:
    """dE/dlnk -> the CHW dimensionless quantity
    [H_*R_*Omega_vac]^-2 dOmega_gw/dlnk (Cutting, Hindmarsh & Weir)."""
    rho_vac = rho_vac_bar(lambda_bar)
    return (3.0 / (8.0 * np.pi * rho_vac**2 * rstar**2 * volume)) * dE_dlnk


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
# Non-rectangular runtime scan: query at an arbitrary (gamma_ij, t)
#
# Unlike the rectangular scan above (every row sharing the same times[] and
# k grid, so RegularGridInterpolator applies directly), a runtime scan (as
# built by ptbuilder.ic.generate_surrogate) gives each gamma_ij row its own
# times[] (starting at that pair's own collision time) and its own wlist
# (narrower at higher gamma_ij) -- there's no shared grid to hand to
# RegularGridInterpolator. query_spectrum uses the same domain-aware
# interpolation already validated for the N=64 reconstruction comparison
# (16_n64_reconstruction_scan_comparison.ipynb's new_interp_fn), simplified
# to a single-point query instead of the full weighted-N-bubble
# reconstruction.
# ---------------------------------------------------------------------------

def load_scan(scan_root: Path) -> SimpleNamespace:
    """
    Load a runtime scan (as built by ptbuilder.ic.generate_surrogate) fully
    into memory -- every row's own times[]/wlist[]/spectrum[n_t, n_w] -- so
    query_spectrum() can be called repeatedly with no further disk I/O.
    These scans are small (tens of rows x tens of times x ~32 frequencies),
    so eager loading upfront is cheap and simpler than per-row lazy caching.

    Reads ONLY each row's own result_*.h5 files under scan_root/gpu/ -- never
    manifest.tsv (a write-only convenience index for the SLURM script, not a
    source of truth here) or setup.h5. Every result_*.h5 carries its own "t"
    and "gamma_ij" attributes and "w"/"spectrum" datasets, so a row is fully
    reconstructed from whatever result files actually exist for it on disk;
    a row directory with no result files yet is simply skipped, and a row's
    time axis is exactly however many result files it actually has, in
    contrast to trusting a possibly-stale expected count from elsewhere.
    """
    scan_root = Path(scan_root)
    output_root = scan_root / "gpu"

    rows = []
    for row_dir in sorted(output_root.iterdir()):
        if not row_dir.is_dir():
            continue
        result_paths = sorted(row_dir.glob("result_*.h5"))
        if not result_paths:
            continue

        ts, gammas, ws, specs = [], [], [], []
        for path in result_paths:
            with h5py.File(path, "r") as f:
                ts.append(float(f.attrs["t"]))
                gammas.append(float(f.attrs["gamma_ij"]))
                ws.append(f["w"][:])
                specs.append(f["spectrum"][:])

        gamma_ij = gammas[0]
        if any(abs(g - gamma_ij) > 1e-9 * max(1.0, abs(gamma_ij)) for g in gammas):
            raise RuntimeError(
                f"{row_dir.name}: gamma_ij differs across its own result files "
                f"({sorted(set(gammas))})")
        wlist = ws[0]
        if any(w.shape != wlist.shape or np.any(np.abs(w - wlist) > 1e-9) for w in ws):
            raise RuntimeError(
                f"{row_dir.name}: w (frequency grid) differs across its own result files")

        order = np.argsort(ts)
        rows.append(SimpleNamespace(
            gamma_ij=gamma_ij, times=np.asarray(ts)[order], wlist=wlist,
            spectrum=np.asarray(specs)[order]))

    if not rows:
        raise FileNotFoundError(f"No rows with any result_*.h5 files found under {output_root}")

    rows.sort(key=lambda r: r.gamma_ij)
    gamma_min = rows[0].gamma_ij
    gamma_max = rows[-1].gamma_ij
    t_max = max(r.times[-1] for r in rows)
    print(f"Loaded scan {scan_root.name}: {len(rows)} rows, "
          f"gamma_ij in [{gamma_min:.3f}, {gamma_max:.3f}], T_MAX={t_max:.3f}")

    return SimpleNamespace(root=scan_root.parent.parent, scan_root=scan_root, rows=rows,
                           gamma_min=gamma_min, gamma_max=gamma_max, t_max=t_max)


def load_scan_manifest(scan_dir: Path) -> SimpleNamespace:
    """
    Load an older, manifest.tsv-indexed runtime scan into the same shape
    load_scan() returns (SimpleNamespace with .rows/.gamma_min/.gamma_max/
    .t_max), so query_spectrum() and reconstruct_pair_spectrum() work on
    either. Unlike load_scan()'s result_*.h5 files (which carry their own
    t/gamma_ij attributes), this format's result_*.h5 files carry neither --
    times and the frequency grid come from each row's own setup.h5 instead,
    indexed via manifest.tsv.
    """
    scan_dir  = Path(scan_dir)
    repo_root = scan_dir.parent.parent   # scan_dir = repo_root/'data'/<scan name>
    manifest_lines = (scan_dir / "manifest.tsv").read_text().strip().split("\n")[1:]
    rows = []
    for line in manifest_lines:
        index, gamma, n_t, setup_rel, name = line.split("\t")
        setup_path = repo_root / setup_rel
        outdir = scan_dir / "gpu" / name
        files = sorted(outdir.glob("result_*.h5"), key=lambda p: int(p.stem.split("_")[1]))
        if not files:
            continue
        with h5py.File(setup_path) as f:
            times_row = f["times"][:]
            w_row     = f["wlist"][:]
        spectrum_row = np.zeros((len(files), len(w_row)))
        for it, fp in enumerate(files):
            with h5py.File(fp) as f:
                spectrum_row[it] = f["spectrum"][:]
        rows.append(SimpleNamespace(gamma_ij=float(gamma), times=times_row[:len(files)],
                                    wlist=w_row, spectrum=spectrum_row))
    if not rows:
        raise FileNotFoundError(f"No rows with any result_*.h5 files found under {scan_dir}")
    rows.sort(key=lambda r: r.gamma_ij)
    gamma_min, gamma_max = rows[0].gamma_ij, rows[-1].gamma_ij
    t_max = max(r.times[-1] for r in rows)
    print(f"Loaded scan {scan_dir.name}: {len(rows)} rows, "
         f"gamma_ij in [{gamma_min:.3f}, {gamma_max:.3f}], T_MAX={t_max:.3f}")
    return SimpleNamespace(root=scan_dir.parent, scan_root=scan_dir, rows=rows,
                           gamma_min=gamma_min, gamma_max=gamma_max, t_max=t_max)


def _interp_row_at_t(row: SimpleNamespace, t: float) -> np.ndarray:
    """Log-linear interpolation over one row's own times[] -- floored (in
    log-space) below its own t_first (no collision yet in this row), clipped
    to its last computed value above its own last time."""
    floor = max(row.spectrum.max() * 1e-12, 1e-300)
    log_spec = np.log(np.maximum(row.spectrum, floor))
    f = interp1d(row.times, log_spec, axis=0, kind="linear", bounds_error=False,
                fill_value=(np.log(floor), log_spec[-1]))
    return np.exp(f(t))


def query_spectrum(scan_data: SimpleNamespace, gamma_ij: float, t: float,
                   no_throw: bool = False) -> tuple:
    """
    Query a non-rectangular runtime scan's GW spectrum at an arbitrary
    (gamma_ij, t) via interpolation:
      - log-linear across the two bracketing gamma_ij rows in scan_data.rows
        (if gamma_ij exactly matches a row, that row's own wlist/spectrum,
        no interpolation needed)
      - log-linear across each bracketing row's own times[] (see
        _interp_row_at_t) -- t below a row's own collision time floors to
        that row's own floor value (a real, physical pre-collision zero,
        not missing data)

    scan_data is whatever load_scan() returned -- pure in-memory, no disk
    I/O here, so cheap to call repeatedly (e.g. for several gamma_ij targets
    against the same scan).

    (gamma_ij, t) outside what the scan actually covers -- gamma_ij outside
    [scan_data.gamma_min, scan_data.gamma_max], or t > scan_data.t_max -- is
    missing data, not a physical zero, so by default this raises ValueError.
    Pass no_throw=True (for a reconstruction pass that needs to keep going
    and separately tally how much weight fell outside the scan's coverage)
    to get an all-zero spectrum back instead.

    Returns (k, spectrum, out_of_range). out_of_range is always False unless
    no_throw=True let the query through. k is the lower-gamma_ij bracketing
    row's own wlist, TRUNCATED to omega_max_hi = min(row_lo.wlist[-1],
    row_hi.wlist[-1]) -- omega_max shrinks with gamma_ij, so the higher
    row's own coverage is usually the narrower one. Never extrapolated past
    what BOTH bracketing rows actually computed: combining a real value from
    one row with a zero-padded "row simply doesn't reach here" value from
    the other, in log-space, would otherwise crash the result toward the
    floor right at that edge -- not a physical feature, just missing data
    pretending to be zero (caught by comparing a real extended-vs-direct
    scan: the last couple of points were orders of magnitude below their
    neighbors).
    """
    rows = scan_data.rows

    if gamma_ij < scan_data.gamma_min or gamma_ij > scan_data.gamma_max or t > scan_data.t_max:
        if not no_throw:
            raise ValueError(
                f"Requested (gamma_ij={gamma_ij:g}, t={t:g}) is outside this scan's "
                f"covered range (gamma_ij in [{scan_data.gamma_min:g}, "
                f"{scan_data.gamma_max:g}], t <= {scan_data.t_max:g}) -- pass "
                f"no_throw=True to get a zero spectrum instead")
        ref_row = rows[0] if gamma_ij <= scan_data.gamma_min else rows[-1]
        return ref_row.wlist, np.zeros_like(ref_row.wlist), True

    gammas = np.array([r.gamma_ij for r in rows])
    g = float(gamma_ij)
    ig_hi = int(np.searchsorted(gammas, g))
    ig_hi = min(max(ig_hi, 1), len(rows) - 1)
    ig_lo = ig_hi - 1
    if gammas[ig_hi] == g:
        ig_lo = ig_hi

    row_lo = rows[ig_lo]
    spec_lo_full = _interp_row_at_t(row_lo, t)

    if ig_lo == ig_hi:
        return row_lo.wlist, spec_lo_full, False

    row_hi = rows[ig_hi]
    spec_hi_full = _interp_row_at_t(row_hi, t)

    k_shared_max = min(row_lo.wlist[-1], row_hi.wlist[-1])
    mask = row_lo.wlist <= k_shared_max
    k = row_lo.wlist[mask]
    spec_lo = spec_lo_full[mask]
    spec_hi = np.interp(k, row_hi.wlist, spec_hi_full, left=0., right=0.)

    frac = (g - gammas[ig_lo]) / (gammas[ig_hi] - gammas[ig_lo])
    floor = max(spec_lo.max(), spec_hi.max()) * 1e-12
    floor = max(floor, 1e-300)
    log_lo = np.log(np.maximum(spec_lo, floor))
    log_hi = np.log(np.maximum(spec_hi, floor))
    spectrum = np.exp(log_lo + frac * (log_hi - log_lo))
    return k, spectrum, False


def _loglog_interp(x_new: np.ndarray, x_old: np.ndarray,
                   y_old: np.ndarray) -> np.ndarray:
    """Linear interpolation in log(x) vs log(y) -- the appropriate choice
    for a log-spaced k grid and a spectrum that varies over many decades
    (matches the log-space convention already used for the time and
    gamma_ij axes; plain np.interp on raw values would treat the log-spaced
    k grid as if it were linear). Zero outside [x_old[0], x_old[-1]]."""
    floor = max(y_old.max(), 1e-300) * 1e-12
    logy = np.log(np.maximum(y_old, floor))
    logx_old = np.log(x_old)
    logx_new = np.log(np.maximum(x_new, x_old[0]))
    out = np.exp(np.interp(logx_new, logx_old, logy, left=np.log(floor), right=logy[-1]))
    return np.where((x_new >= x_old[0]) & (x_new <= x_old[-1]), out, 0.)


def build_runtime_scan_interpolator(scan_data: SimpleNamespace,
                                    k_out: np.ndarray):
    """
    Build a (gamma_ij, t) -> spectrum[k_out] interpolator for a
    non-rectangular runtime scan (load_scan()/load_scan_manifest()), for
    repeated reconstruct_pair_spectrum() calls against the same scan.

    Every row is regridded onto the shared k_out grid *first* (log-log,
    see _loglog_interp), so that the subsequent interpolation across
    gamma_ij (log-linear, between the two bracketing rows) and across each
    row's own times (log-linear) always compares values already living on
    the same k axis. This is deliberate -- interpolating across gamma_ij
    on each row's own native k-grid and regridding to k_out only at the
    end is a different (and ~10x slower, since it cannot be precomputed
    once and reused) calculation, not an equivalent one; regrid-first was
    checked to be within ~6% of regrid-last even with consistent log-log
    k-interpolation, so the two are not interchangeable, and this is the
    faster of the two.

    Returns (interp_fn, gammas) -- interp_fn(pts) takes pts = [[gamma, t],
    ...] and returns spectrum[k_out] per row, gammas is the sorted array of
    tabulated gamma_ij values (for clipping queries to the covered range).
    """
    rows = scan_data.rows
    gammas = np.array([r.gamma_ij for r in rows])
    row_log_interp = []
    for r in rows:
        spec_k = np.zeros((len(r.times), len(k_out)))
        for it in range(len(r.times)):
            spec_k[it] = _loglog_interp(k_out, r.wlist, r.spectrum[it])
        floor = max(spec_k.max(), 1e-300) * 1e-12
        log_spec = np.log(np.maximum(spec_k, floor))
        row_log_interp.append(interp1d(
            r.times, log_spec, axis=0, kind="linear", bounds_error=False,
            fill_value=(np.log(floor), log_spec[-1]),
        ))

    def interp_fn(pts):
        g_arr, t_arr = pts[:, 0], pts[:, 1]
        out = np.empty((len(pts), len(k_out)))
        for g_val in np.unique(g_arr):
            sel = g_arr == g_val
            g_clip = float(np.clip(g_val, gammas.min(), gammas.max()))
            ig_hi = int(np.searchsorted(gammas, g_clip))
            ig_hi = min(max(ig_hi, 1), len(gammas) - 1)
            ig_lo = ig_hi - 1
            g_lo, g_hi = gammas[ig_lo], gammas[ig_hi]
            frac = 0.0 if g_hi == g_lo else (g_clip - g_lo) / (g_hi - g_lo)
            log_lo = row_log_interp[ig_lo](t_arr[sel])
            log_hi = row_log_interp[ig_hi](t_arr[sel])
            out[sel] = np.exp(log_lo + frac * (log_hi - log_lo))
        return out

    return interp_fn, gammas


def reconstruct_pair_spectrum(gamma_ij: float, weights_t: np.ndarray,
                              weights_times: np.ndarray, t_max: float,
                              interp_fn, times_bounds: np.ndarray,
                              gammas: np.ndarray) -> np.ndarray:
    """
    Reconstruct one colliding pair's dE/dlnk(k_out), integrating its
    collision weight against interp_fn's cumulative spectrum over
    [times_bounds[0], min(t_max, times_bounds[-1])].

    interp_fn, gammas: from build_runtime_scan_interpolator(scan_data, k_out)
        -- build once per (scan, k_out), reuse across every pair.
    times_bounds: [earliest row start time, scan_data.t_max].

    If t_max exceeds the scan's own coverage (times_bounds[-1]), the query
    is silently clamped there -- warns with the fraction of this pair's own
    collision weight (by integrated area, not sample count) that falls past
    the clamp and is therefore dropped, so an insufficient scan surfaces
    itself instead of requiring a separate manual check.
    """
    g = float(np.clip(gamma_ij, gammas.min(), gammas.max()))
    t_hi = min(t_max, times_bounds[-1])

    if t_max > times_bounds[-1]:
        full_mask = (weights_times >= times_bounds[0]) & (weights_times <= t_max)
        if full_mask.sum() >= 2:
            wt_full, tt_full = weights_t[full_mask], weights_times[full_mask]
            total_weight = np.trapz(wt_full, tt_full)
            if total_weight > 0:
                used = tt_full <= t_hi
                used_weight = np.trapz(wt_full[used], tt_full[used]) if used.sum() >= 2 else 0.
                discarded_frac = 1. - used_weight / total_weight
                if discarded_frac > 1e-3:
                    warnings.warn(
                        f"reconstruct_pair_spectrum: gamma_ij={gamma_ij:.3f} queried to "
                        f"t_max={t_max:.2f}, but the scan only covers t<={times_bounds[-1]:.2f} "
                        f"-- {discarded_frac:.1%} of this pair's weight (by integrated area) "
                        f"lies past the scan's coverage and is silently dropped.",
                        stacklevel=2,
                    )

    mask = (weights_times >= times_bounds[0]) & (weights_times <= t_hi)
    t_comp = weights_times[mask]
    if len(t_comp) == 0 or t_comp[-1] < t_hi:
        t_comp = np.append(t_comp, t_hi)
    w = np.interp(t_comp, weights_times, weights_t)
    pts = np.column_stack([np.full(len(t_comp), g), t_comp])
    specs = interp_fn(pts)
    delta = np.diff(specs, axis=0)
    return (w[:len(t_comp) - 1, None] * delta).sum(axis=0)


def reconstruct_pair_spectrum_series(gamma_ij: float, weights_t: np.ndarray,
                                     weights_times: np.ndarray, t_query: np.ndarray,
                                     interp_fn, times_bounds: np.ndarray,
                                     gammas: np.ndarray) -> np.ndarray:
    """
    Same result as calling reconstruct_pair_spectrum(..., t_max, ...) once
    per entry of t_query, but O(n_weights) total instead of
    O(n_weights * len(t_query)): integrates the cumulative w(t)*dspec once
    over weights_times, then linearly interpolates that cumulative curve
    onto t_query. weights_times is dense (hundreds-to-thousands of points)
    relative to any realistic t_query, so this is not a separate
    approximation -- just reading off the same cumulative integral at many
    points instead of recomputing it from scratch at each one.

    Returns array of shape (len(t_query), len(k_out)).
    """
    g = float(np.clip(gamma_ij, gammas.min(), gammas.max()))
    t_lo, t_scan_hi = times_bounds[0], times_bounds[-1]
    t_query = np.asarray(t_query, dtype=float)

    mask = (weights_times >= t_lo) & (weights_times <= t_scan_hi)
    t_comp = weights_times[mask]
    if len(t_comp) == 0 or t_comp[-1] < t_scan_hi:
        t_comp = np.append(t_comp, t_scan_hi)
    w = np.interp(t_comp, weights_times, weights_t)
    pts = np.column_stack([np.full(len(t_comp), g), t_comp])
    specs = interp_fn(pts)
    delta = np.diff(specs, axis=0)
    contrib = w[:len(t_comp) - 1, None] * delta
    cum = np.vstack([np.zeros((1, specs.shape[1])), np.cumsum(contrib, axis=0)])

    t_max_q = float(t_query.max()) if len(t_query) else t_lo
    if t_max_q > t_scan_hi:
        full_mask = (weights_times >= t_lo) & (weights_times <= t_max_q)
        if full_mask.sum() >= 2:
            wt_full, tt_full = weights_t[full_mask], weights_times[full_mask]
            total_weight = np.trapz(wt_full, tt_full)
            if total_weight > 0:
                used = tt_full <= t_scan_hi
                used_weight = np.trapz(wt_full[used], tt_full[used]) if used.sum() >= 2 else 0.
                discarded_frac = 1. - used_weight / total_weight
                if discarded_frac > 1e-3:
                    warnings.warn(
                        f"reconstruct_pair_spectrum_series: gamma_ij={gamma_ij:.3f} queried up to "
                        f"t_max={t_max_q:.2f}, but the scan only covers t<={t_scan_hi:.2f} "
                        f"-- {discarded_frac:.1%} of this pair's weight (by integrated area) "
                        f"lies past the scan's coverage and is silently dropped.",
                        stacklevel=2,
                    )

    t_clamped = np.clip(t_query, t_lo, t_scan_hi)
    idx = np.clip(np.searchsorted(t_comp, t_clamped, side='right') - 1, 0, len(t_comp) - 2)
    t0, t1 = t_comp[idx], t_comp[idx + 1]
    frac = np.where(t1 > t0, (t_clamped - t0) / np.where(t1 > t0, t1 - t0, 1.), 0.)
    out = cum[idx] + frac[:, None] * (cum[idx + 1] - cum[idx])
    out[t_query < t_lo] = 0.
    return out


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
                f"{path} has no collision_radius metadata; regenerate or "
                "relabel it explicitly before use"
            )
        if str(f.attrs["collision_radius"]) not in ("mid", "out", "in"):
            raise ValueError(
                f"{path} has unrecognized collision_radius="
                f"{f.attrs['collision_radius']!r}; must be 'mid', 'out', or 'in'"
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
    if collision_radius not in ("mid", "out", "in"):
        raise ValueError("collision_radius must be 'mid', 'out', or 'in'")
    R         = kinematics.R(t, t_init=t_init, r=collision_radius)
    rout_0    = float(kinematics.profile.rout_0)
    rin_0     = float(kinematics.profile.rin_0)
    rmid_0    = float(kinematics.profile.rmid_0)
    collision_r0 = {"mid": rmid_0, "out": rout_0, "in": rin_0}[collision_radius]

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
