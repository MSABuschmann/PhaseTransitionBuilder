"""
Thin wrapper around pySledgehamr for reading 3D simulation output.
"""
from pathlib import Path
from typing import Optional

import numpy as np


def load_output(path: str):
    """
    Load a sledgehamr output directory via pySledgehamr.

    Returns the pySledgehamr Output object.
    Raises ImportError with a helpful message if pySledgehamr is not installed.
    """
    try:
        import pySledgehamr as sledgehamr
    except ImportError as e:
        raise ImportError(
            "pySledgehamr is not installed.  It is part of the sledgehamr "
            "project; see the sledgehamr repository for installation instructions."
        ) from e
    return sledgehamr.Output(str(path))


def get_gw_spectrum(output, snapshot: int, L: float,
                    zero_pad: float = 1.) -> tuple:
    """
    Read and normalize the GW spectrum from one pySledgehamr snapshot.

    Returns (k, dE_dlnk, t).
    """
    from .analysis import sledgehamr_normalization
    data = output.GetGravitationalWaveSpectrum(snapshot)
    k, s = sledgehamr_normalization(data["k_sq"], data["spectrum"],
                                    L, zero_pad)
    return k, s, float(data["t"])


def get_final_spectrum(output, L: float, zero_pad: float = 1.,
                       n_avg: int = 5) -> tuple:
    """
    Return the GW spectrum averaged over the last n_avg snapshots.

    Returns (k, dE_dlnk_mean, dE_dlnk_std).
    """
    from .analysis import sledgehamr_normalization

    n_total = output.GetNumberOfSnapshots()
    start   = max(0, n_total - n_avg)

    all_s = []
    k_ref = None
    for sl in range(start, n_total):
        data = output.GetGravitationalWaveSpectrum(sl)
        k, s = sledgehamr_normalization(data["k_sq"], data["spectrum"],
                                        L, zero_pad)
        if k_ref is None:
            k_ref = k
        all_s.append(s)

    all_s = np.array(all_s)
    return k_ref, all_s.mean(axis=0), all_s.std(axis=0)
