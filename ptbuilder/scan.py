"""
Scan runner: iterate over (gamma, t_cut) parameter space using BubbleMaster.
Results are cached to disk so each (model, gamma) pair is run at most once.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from .ic import BubbleMasterParams, write_2d_setup
from .sim2d import run_bubblemaster


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScanConfig:
    """Parameters that define the (gamma, time) scan grid."""
    gammas:       np.ndarray          # 1-D array of γ values
    times:        np.ndarray          # 1-D array of integration time points
    bm_params:    BubbleMasterParams = field(
                      default_factory=BubbleMasterParams)

    def __post_init__(self):
        self.gammas = np.asarray(self.gammas, dtype=float)
        self.times  = np.asarray(self.times,  dtype=float)


@dataclass
class ScanResult:
    """
    Holds the BubbleMaster output for one (model, gamma) run.

    Attributes
    ----------
    gamma     : γ parameter
    times     : integration time points (n_t,)
    w         : frequency array (n_w,)
    spectrum  : cumulative GW energy at each time (n_t, n_w)
                = E(<= times[i])  in BubbleMaster units
    dEdt      : instantaneous rate  (n_t, n_w)
                = finite-difference derivative of spectrum w.r.t. time
    k         : cos(theta) grid (n_k,), or None if this run didn't save
                amplitude (i.e. was run without --save-amplitude)
    amp_re    : Re A(w, cos_theta) at each time, (n_t, n_w, n_k), or None
    amp_im    : Im A(w, cos_theta) at each time, (n_t, n_w, n_k), or None
    """
    gamma:    float
    times:    np.ndarray
    w:        np.ndarray
    spectrum: np.ndarray   # (n_t, n_w)
    dEdt:     np.ndarray   # (n_t, n_w)
    k:        Optional[np.ndarray] = None   # (n_k,)
    amp_re:   Optional[np.ndarray] = None   # (n_t, n_w, n_k)
    amp_im:   Optional[np.ndarray] = None   # (n_t, n_w, n_k)

    def to_hdf5(self, group: h5py.Group):
        group.attrs["gamma"] = self.gamma
        group.create_dataset("times",    data=self.times)
        group.create_dataset("w",        data=self.w)
        group.create_dataset("spectrum", data=self.spectrum)
        group.create_dataset("dEdt",     data=self.dEdt)
        if self.amp_re is not None:
            group.create_dataset("k",      data=self.k)
            group.create_dataset("amp_re", data=self.amp_re)
            group.create_dataset("amp_im", data=self.amp_im)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "ScanResult":
        has_amp = "amp_re" in group
        return cls(
            gamma    = float(group.attrs["gamma"]),
            times    = group["times"][:],
            w        = group["w"][:],
            spectrum = group["spectrum"][:],
            dEdt     = group["dEdt"][:],
            k        = group["k"][:]      if has_amp else None,
            amp_re   = group["amp_re"][:] if has_amp else None,
            amp_im   = group["amp_im"][:] if has_amp else None,
        )


# ---------------------------------------------------------------------------
# Scan runner
# ---------------------------------------------------------------------------

def run_scan(model, scan_config: ScanConfig, config=None,
             force: bool = False, save_amplitude: bool = False) -> Dict[float, ScanResult]:
    """
    Run or load a full (gamma, time) scan for `model`.

    For each γ in scan_config.gammas:
      - Check whether a cached result exists in model.results_dir/scan.h5
      - If not (or if force=True), write a BubbleMaster setup, run the binary,
        read results, compute dEdt, and save to cache.

    save_amplitude : if True, pass --save-amplitude to bubblemaster so each
        run also stores Re/Im A(w, cos_theta); ScanResult.amp_re/amp_im/k
        get populated (and scan_cache.h5 grows accordingly -- angle-resolved
        amplitude is ~n_k times larger than the angle-integrated spectrum).

    Returns a dict mapping γ → ScanResult.
    """
    from .config import Config
    cfg = config or Config()

    scan_dir  = model.results_dir / "scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    cache_path = model.results_dir / "scan_cache.h5"

    results: Dict[float, ScanResult] = {}

    # Load existing cache
    if cache_path.exists() and not force:
        results = load_scan(cache_path)

    for gamma in scan_config.gammas:
        key = float(gamma)
        if key in results and not force:
            continue

        # Write BubbleMaster setup
        setup_path  = scan_dir / f"setup_gamma{gamma:.6g}.h5"
        output_dir  = scan_dir / f"out_gamma{gamma:.6g}"
        output_dir.mkdir(parents=True, exist_ok=True)

        write_2d_setup(
            model   = model,
            gamma   = gamma,
            times   = scan_config.times,
            path    = setup_path,
            params  = scan_config.bm_params,
        )

        # Run bubblemaster
        run_bubblemaster(setup_path, output_dir, cfg, save_amplitude=save_amplitude)

        # Read results
        result = _read_bubblemaster_output(output_dir, gamma, scan_config.times)
        results[key] = result

    # Save / update cache
    _save_scan_cache(results, cache_path)

    return results


def load_bm_result(output_dir: Path, gamma: float,
                   times: np.ndarray) -> ScanResult:
    """Load a single BubbleMaster run from its output directory."""
    return _read_bubblemaster_output(Path(output_dir), gamma, times)


def load_scan(cache_path: Path) -> Dict[float, ScanResult]:
    """Load a previously saved scan cache."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}
    results = {}
    with h5py.File(cache_path, "r") as f:
        for key in f:
            results[float(key)] = ScanResult.from_hdf5(f[key])
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_bubblemaster_output(output_dir: Path, gamma: float,
                              times: np.ndarray) -> ScanResult:
    """
    Read the per-time-step HDF5 files written by the new bubblemaster binary
    and assemble into a ScanResult.

    Expected layout in output_dir:
      result_0000.h5, result_0001.h5, ...
    Each file has datasets "w" [n_w] and "spectrum" [n_w]; if the run used
    --save-amplitude, also "k" [n_k], "amp_re" [n_w, n_k], "amp_im" [n_w, n_k]
    -- auto-detected here (no separate flag needed) and stacked into
    ScanResult.amp_re/amp_im if present.
    """
    output_dir = Path(output_dir)
    result_files = sorted(output_dir.glob("result_*.h5"),
                          key=lambda p: int(p.stem.split("_")[1]))

    if not result_files:
        raise FileNotFoundError(
            f"No result_*.h5 files found in {output_dir}. "
            "Did bubblemaster run successfully?"
        )

    spectra  = []
    amp_res  = []
    amp_ims  = []
    w = None
    k = None
    has_amp = None
    for fpath in result_files:
        with h5py.File(fpath, "r") as f:
            w_i    = f["w"][:]
            spec_i = f["spectrum"][:]
            file_has_amp = "amp_re" in f
            if has_amp is None:
                has_amp = file_has_amp
            elif file_has_amp != has_amp:
                raise ValueError(
                    f"{fpath} has inconsistent amplitude data vs. earlier "
                    f"result files in {output_dir} -- was --save-amplitude "
                    f"used for only part of this run?"
                )
            if has_amp:
                if k is None:
                    k = f["k"][:]
                amp_res.append(f["amp_re"][:])
                amp_ims.append(f["amp_im"][:])
        if w is None:
            w = w_i
        spectra.append(spec_i)

    spectrum = np.array(spectra)   # (n_t, n_w)

    # dEdt via finite differences (forward difference, first order)
    dt   = np.diff(np.concatenate([[0.], times]))
    dEdt = np.zeros_like(spectrum)
    dEdt[0] = spectrum[0] / dt[0] if dt[0] > 0 else 0.
    for i in range(1, len(times)):
        dt_i       = times[i] - times[i - 1]
        dEdt[i]    = (spectrum[i] - spectrum[i - 1]) / dt_i if dt_i > 0 else 0.

    return ScanResult(
        gamma    = gamma,
        times    = times,
        w        = w,
        spectrum = spectrum,
        dEdt     = dEdt,
        k        = k               if has_amp else None,
        amp_re   = np.array(amp_res) if has_amp else None,   # (n_t, n_w, n_k)
        amp_im   = np.array(amp_ims) if has_amp else None,   # (n_t, n_w, n_k)
    )


def _save_scan_cache(results: Dict[float, ScanResult], path: Path):
    path = Path(path)
    # Merge with existing cache (keep existing entries not in results)
    existing = {}
    if path.exists():
        existing = load_scan(path)
    merged = {**existing, **results}

    with h5py.File(path, "w") as f:
        for gamma, res in merged.items():
            grp = f.require_group(str(gamma))
            res.to_hdf5(grp)
