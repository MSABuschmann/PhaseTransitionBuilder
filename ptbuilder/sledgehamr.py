"""
Thin wrapper around pySledgehamr for reading 3D simulation output.
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _import_pysledgehamr(config=None):
    try:
        import pySledgehamr as sledgehamr
        return sledgehamr
    except ImportError:
        pass

    # Try the path from config, or the default sibling directory
    candidates = []
    if config is not None:
        candidates.append(Path(config.pysledgehamr_path))
    candidates.append(Path(__file__).parent.parent.parent / "sledgehamr")

    for p in candidates:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
        try:
            import pySledgehamr as sledgehamr
            return sledgehamr
        except ImportError:
            continue

    raise ImportError(
        "pySledgehamr not found. Set config.pysledgehamr_path to the sledgehamr repo."
    )


def load_output(path: str, config=None):
    """Load a sledgehamr output directory. Returns a pySledgehamr Output object."""
    sledgehamr = _import_pysledgehamr(config)
    return sledgehamr.Output(str(path))


def get_gw_spectrum(output, snapshot: int, L: float,
                    zero_pad: float = 1.) -> tuple:
    """
    Read and normalize the GW spectrum from one snapshot index.

    Returns (k, dE_dlnk, t).
    """
    from .analysis import sledgehamr_normalization
    data = output.GetGravitationalWaveSpectrum(snapshot)
    k, s = sledgehamr_normalization(data["k_sq"], data["spectrum"], L, zero_pad)
    return k, s, float(data["t"])


def get_final_spectrum(output, L: float, zero_pad: float = 1.,
                       n_avg: int = 5) -> tuple:
    """
    Return the GW spectrum averaged over the last n_avg snapshots.

    Returns (k, dE_dlnk_mean, dE_dlnk_std).
    """
    from .analysis import sledgehamr_normalization

    times  = output.GetTimesOfGravitationalWaveSpectra()
    n_total = len(times)
    start   = max(0, n_total - n_avg)

    all_s = []
    k_ref = None
    for sl in range(start, n_total):
        data = output.GetGravitationalWaveSpectrum(sl)
        k, s = sledgehamr_normalization(data["k_sq"], data["spectrum"], L, zero_pad)
        if k_ref is None:
            k_ref = k
        all_s.append(s)

    all_s = np.array(all_s)
    return k_ref, all_s.mean(axis=0), all_s.std(axis=0)


def get_spectrum_at_time(output, t_target: float, L: float,
                         zero_pad: float = 1.) -> tuple:
    """
    Return the GW spectrum from the snapshot closest to t_target.

    Returns (k, dE_dlnk, t_actual).
    """
    times = np.array(output.GetTimesOfGravitationalWaveSpectra())
    idx   = int(np.argmin(np.abs(times - t_target)))
    return get_gw_spectrum(output, idx, L, zero_pad)
