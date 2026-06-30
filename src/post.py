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


def sloth_quality_metrics(
    positions: np.ndarray,
    freqs: np.ndarray,
    s2: np.ndarray,
    *,
    v_min: float = 120.0,
    v_max: float = 900.0,
    fk_ref: tuple[np.ndarray, np.ndarray] | None = None,
    coverage: np.ndarray | None = None,
) -> dict:
    """
    Assess a stacked I-FDG sloth field and return interpretable metrics.

    The sloth is converted to phase velocity ``c = 1/sqrt(Re s2)`` (eq. 11)
    and clipped to the physical window ``[v_min, v_max]``; pixels outside the
    window are counted as "non-physical" (white) rather than averaged in.

    Metrics returned
        white_pct          : % of (position, freq) pixels with no physical
                             velocity (Re s2 <= 0 or outside the window). Lower
                             is better; it is the direct measure of how much of
                             the section is uninterpretable.
        median_c, c_q25/75 : robust phase-velocity centre and IQR (m/s).
        lateral_coherence  : mean lag-1 spatial autocorrelation of c along the
                             fibre, averaged over frequency. ~0 is noise-like;
                             -> 1 means a laterally smooth / coherent section.
                             This is the key "is the image real?" number.
        c_of_f             : median phase velocity per frequency -- the
                             gradiometry dispersion curve (m/s).
        fkpeak_rms_mps     : RMS misfit of c_of_f against an independent
                             FK-peak dispersion ``fk_ref=(freqs_ref, c_ref)``;
                             cross-checks the absolute velocity scale.
        coverage_mean      : mean contributing gathers per pixel, if a
                             ``coverage`` array (same shape as s2) is provided.

    :param positions: (nch,) along-fibre coordinates (m). Kept for symmetry
        with the plotting API; not used in the scalar metrics.
    :param freqs: (nfreq,) frequency axis (Hz).
    :param s2: (nch, nfreq) complex stacked sloth.
    :param v_min: Lower physical phase-velocity bound (m/s).
    :param v_max: Upper physical phase-velocity bound (m/s).
    :param fk_ref: Optional ``(freqs_ref, c_ref)`` FK-peak dispersion to
        compare against.
    :param coverage: Optional (nch, nfreq) count of contributing gathers.
    :return: dict of metrics (NaN where undefined).
    """
    import warnings

    re = np.real(np.asarray(s2))
    v = np.full(re.shape, np.nan, dtype=np.float64)
    pos = re > 0
    v[pos] = 1.0 / np.sqrt(re[pos])
    v[(v < float(v_min)) | (v > float(v_max))] = np.nan

    n_pix = max(int(v.size), 1)
    white_pct = 100.0 * float(np.count_nonzero(~np.isfinite(v))) / n_pix
    finite = v[np.isfinite(v)]
    if finite.size:
        median_c = float(np.median(finite))
        q25, q75 = (float(x) for x in np.percentile(finite, [25, 75]))
    else:
        median_c = q25 = q75 = float("nan")

    # Lateral coherence: lag-1 along-fibre autocorrelation, per frequency.
    cohs = []
    for j in range(v.shape[1]):
        a, b = v[:-1, j], v[1:, j]
        m = np.isfinite(a) & np.isfinite(b)
        if int(m.sum()) > 5 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
            cohs.append(float(np.corrcoef(a[m], b[m])[0, 1]))
    lateral_coherence = float(np.mean(cohs)) if cohs else float("nan")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        c_of_f = np.nanmedian(v, axis=0)

    out: dict[str, Any] = {
        "white_pct": white_pct,
        "median_c": median_c,
        "c_q25": q25,
        "c_q75": q75,
        "lateral_coherence": lateral_coherence,
        "c_of_f": c_of_f,
        "n_pos": int(v.shape[0]),
        "n_freq": int(v.shape[1]),
    }
    if fk_ref is not None:
        fr, cr = (np.asarray(x, dtype=np.float64) for x in fk_ref)
        order = np.argsort(fr)
        cref_i = np.interp(
            np.asarray(freqs, dtype=np.float64), fr[order], cr[order],
            left=np.nan, right=np.nan,
        )
        good = np.isfinite(c_of_f) & np.isfinite(cref_i)
        out["fkpeak_rms_mps"] = (
            float(np.sqrt(np.mean((c_of_f[good] - cref_i[good]) ** 2)))
            if np.any(good) else float("nan")
        )
    if coverage is not None:
        cov = np.asarray(coverage, dtype=np.float64)
        out["coverage_mean"] = float(np.nanmean(cov)) if cov.size else float("nan")
        # Split the white pixels into coverage gaps vs. non-physical physics:
        #   uncovered  -> no gather reached this (x, f): an image gap, not a fault
        #   nonphysical -> covered, but Re s2 <= 0 / out of band / quality-masked
        covered = cov > 0
        out["uncovered_pct"] = 100.0 * float(np.count_nonzero(~covered)) / n_pix
        nonphys = covered & ~np.isfinite(v)
        out["nonphysical_pct"] = (
            100.0 * float(np.count_nonzero(nonphys)) / max(int(covered.sum()), 1)
        )
    return out


def print_quality_report(
    metrics: Mapping[str, Any], *, freqs: np.ndarray | None = None
) -> None:
    """
    Pretty-print :func:`sloth_quality_metrics` output for notebook QC.

    :param metrics: dict returned by :func:`sloth_quality_metrics`.
    :param freqs: If given, also tabulate the median dispersion curve c(f).
    """
    m = metrics
    print("I-FDG sloth - quality report")
    print(f"  grid                 : {m['n_pos']} positions x {m['n_freq']} freqs")
    print(f"  non-physical (white) : {m['white_pct']:.1f}%   (lower is better)")
    print(
        f"  phase velocity       : median {m['median_c']:.0f} m/s "
        f"(IQR {m['c_q25']:.0f}-{m['c_q75']:.0f})"
    )
    print(
        f"  lateral coherence    : {m['lateral_coherence']:+.2f}   "
        f"(lag-1 spatial autocorr; ~0 noise-like, ->1 coherent)"
    )
    if "fkpeak_rms_mps" in m:
        print(f"  vs FK-peak dispersion: RMS {m['fkpeak_rms_mps']:.0f} m/s")
    if "coverage_mean" in m:
        print(f"  coverage             : {m['coverage_mean']:.1f} gathers/pixel (mean)")
    if "uncovered_pct" in m:
        print(f"  coverage gaps        : {m['uncovered_pct']:.1f}% of grid uncovered (image gaps)")
        print(f"  non-physical (covered): {m['nonphysical_pct']:.1f}% of covered pixels")
    if freqs is not None and np.ndim(m.get("c_of_f")) == 1:
        print("  dispersion c(f):")
        for f, c in zip(np.asarray(freqs), np.asarray(m["c_of_f"])):
            if np.isfinite(c):
                print(f"      {float(f):5.1f} Hz : {float(c):5.0f} m/s")
