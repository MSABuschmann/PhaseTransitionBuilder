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
    """
    gamma:    float
    times:    np.ndarray
    w:        np.ndarray
    spectrum: np.ndarray   # (n_t, n_w)
    dEdt:     np.ndarray   # (n_t, n_w)

    def to_hdf5(self, group: h5py.Group):
        group.attrs["gamma"] = self.gamma
        group.create_dataset("times",    data=self.times)
        group.create_dataset("w",        data=self.w)
        group.create_dataset("spectrum", data=self.spectrum)
        group.create_dataset("dEdt",     data=self.dEdt)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "ScanResult":
        return cls(
            gamma    = float(group.attrs["gamma"]),
            times    = group["times"][:],
            w        = group["w"][:],
            spectrum = group["spectrum"][:],
            dEdt     = group["dEdt"][:],
        )


# ---------------------------------------------------------------------------
# Scan runner
# ---------------------------------------------------------------------------

def run_scan(model, scan_config: ScanConfig, config=None,
             force: bool = False) -> Dict[float, ScanResult]:
    """
    Run or load a full (gamma, time) scan for `model`.

    For each γ in scan_config.gammas:
      - Check whether a cached result exists in model.results_dir/scan.h5
      - If not (or if force=True), write a BubbleMaster setup, run the binary,
        read results, compute dEdt, and save to cache.

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
        run_bubblemaster(setup_path, output_dir, cfg)

        # Read results
        result = _read_bubblemaster_output(output_dir, gamma, scan_config.times)
        results[key] = result

    # Save / update cache
    _save_scan_cache(results, cache_path)

    return results


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
    Each file has datasets "w" [n_w] and "spectrum" [n_w].
    """
    output_dir = Path(output_dir)
    result_files = sorted(output_dir.glob("result_*.h5"),
                          key=lambda p: int(p.stem.split("_")[1]))

    if not result_files:
        raise FileNotFoundError(
            f"No result_*.h5 files found in {output_dir}. "
            "Did bubblemaster run successfully?"
        )

    spectra = []
    w = None
    for fpath in result_files:
        with h5py.File(fpath, "r") as f:
            w_i    = f["w"][:]
            spec_i = f["spectrum"][:]
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
