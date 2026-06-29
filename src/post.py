"""
:module: src/post.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Post-processing of gradiometry products: quality masking,
          NaN-aware median filtering, and product export.

Mirrors the right-hand column of the Davis et al. (2026, Fig. 6) workflow:
suppress edge artifacts -> phase velocity -> stack -> median filter.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.utils import PathLike

logger = logging.getLogger(__name__)


def quality_mask(
    nch: int,
    offsets: Sequence[np.ndarray],
    *,
    edge_frac: float = 0.1,
    r_min_m: float = 0.0,
) -> np.ndarray:
    """
    Boolean validity mask over channels for a stacked I-FDG product.

    Two effects make channels unreliable:
    - ARRAY EDGES: the spatial FFT assumes periodicity; even with channel
      tapering, the outer ``edge_frac`` of the aperture is biased.
    - NEAR-SOURCE: within r_min of a virtual source the field is near-field
      (the Hankel singularity) and the masked window is degenerate. A channel
      is flagged only if it is near-source in EVERY contributing VSG, since
      the VSG stack averages that bias away when other sources cover it.

    :param nch: Number of channels.
    :param offsets: Per-VSG signed offset arrays (each (nch,)).
    :param edge_frac: Fraction of channels masked at each array end.
    :param r_min_m: Near-source exclusion radius (m).
    :return: (nch,) boolean, True = valid.
    """
    valid = np.ones(nch, dtype=bool)
    n_edge = int(np.floor(float(edge_frac) * nch))
    if n_edge > 0:
        valid[:n_edge] = False
        valid[nch - n_edge:] = False

    if r_min_m > 0 and len(offsets) > 0:
        near_everywhere = np.ones(nch, dtype=bool)
        for off in offsets:
            near_everywhere &= np.abs(np.asarray(off)) < float(r_min_m)
        valid &= ~near_everywhere
    return valid


def median_filter_1d(a: np.ndarray, size: int, *, axis: int = 0) -> np.ndarray:
    """
    NaN-aware running median along one axis (numpy-only; uses
    ``scipy.ndimage`` semantics of edge replication via padding).

    Used to suppress residual numerical outliers after the VSG stack
    (Davis et al. 2026, Fig. 6, final step).

    :param a: Input array (NaNs pass through where the whole window is NaN).
    :param size: Window length (samples); forced odd.
    :param axis: Axis to filter along.
    :return: Filtered array, same shape and dtype float64.
    """
    a = np.asarray(a, dtype=np.float64)
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    if size == 1 or a.shape[axis] < 2:
        return a.copy()

    a_m = np.moveaxis(a, axis, -1)
    half = size // 2
    pad = [(0, 0)] * a_m.ndim
    pad[-1] = (half, half)
    a_pad = np.pad(a_m, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(a_pad, size, axis=-1)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        out = np.nanmedian(win, axis=-1)
    return np.moveaxis(out, -1, axis)


def save_products(
    out_path: PathLike,
    *,
    s2: np.ndarray,
    vel: np.ndarray,
    freqs: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
    meta: Mapping[str, Any],
) -> Path:
    """
    Write the gradiometry products to a single compressed ``.npz``.

    Keys:
        s2        : complex64 (nch, nfreq) stacked I-FDG sloth (eq. 10)
        vel       : float32  (nch, nfreq) phase velocity, NaN where invalid
        freqs     : float64  (nfreq,) Hz
        positions : float64  (nch,) absolute along-fiber positions (m)
        valid     : bool     (nch,) channel quality mask
        meta_json : str      JSON-encoded run metadata (config echo)

    :return: The resolved output path.
    """
    p = Path(out_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        s2=np.asarray(s2, dtype=np.complex64),
        vel=np.asarray(vel, dtype=np.float32),
        freqs=np.asarray(freqs, dtype=np.float64),
        positions=np.asarray(positions, dtype=np.float64),
        valid=np.asarray(valid, dtype=bool),
        meta_json=json.dumps(dict(meta), default=str),
    )
    logger.info("Saved gradiometry products: %s", p)
    return p
