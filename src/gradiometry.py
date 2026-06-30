"""
:module: src/gradiometry.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Wavefield gradiometry solvers: TDG, FDG, and interferometric
          frequency-domain gradiometry (I-FDG).

All solvers estimate the SLOTH s^2(x) = 1/c^2(x) (squared phase slowness)
of a single surface-wave mode obeying the 2-D scalar wave equation
[lap - s^2 d^2/dt^2] u = 0  (de Ridder & Curtis 2017; Davis et al. 2026).

Equation map (Davis et al. 2026):
    eq. 2  -> :func:`tdg_sloth_pointwise`
    eq. 3  -> :func:`tdg_sloth`            (least-squares over time)
    eq. 5  -> :func:`fdg_sloth`            (per frequency)
    eq. 6  -> :func:`fdg_sloth_stack`      (stack over frequencies)
    eq. 9  -> :func:`fdg_sloth` applied to a VSG spectrum (I-FDG per VSG)
    eq. 10 -> :func:`ifdg_stack`           (stack over virtual sources)

The I-FDG sloth keeps its frequency argument -- this is the method's whole
point: stacking happens over VIRTUAL SOURCES, not over frequencies, so the
depth sensitivity carried by frequency (via the surface-wave kernel) is
retained.

The raw estimates are COMPLEX: for an exact Helmholtz field the imaginary
part vanishes (it carries the transport/amplitude equation residual), so
its magnitude is a useful quality diagnostic. Convert to phase velocity
with :func:`sloth_to_velocity`, which takes the real part and masks
non-physical values.
"""
from __future__ import annotations

import logging
import warnings
from typing import Sequence

import numpy as np

from src.utils import get_vs_number, cosine_taper
from src.laplacian import laplacian_fiber

logger = logging.getLogger(__name__)

#: Relative floor applied to |V|^2 denominators to avoid 0/0 at dead pixels.
_EPS_REL = 1e-12

def temporal_fft(
    data: np.ndarray,
    fs: float,
    *,
    f_min: float | None = None,
    f_max: float | None = None,
    axis: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    FFT a (possibly masked) time/lag-domain gather over the lag axis and
    keep the usable surface-wave band [f_min, f_max].

    :param data: real array with the lag axis along ``axis``.
    :param fs: Sampling rate (Hz).
    :param f_min: Lower band edge (Hz). None keeps everything above DC.
    :param f_max: Upper band edge (Hz). None keeps everything to Nyquist.
    :param axis: Lag axis.
    :return: (V, freqs): complex spectrum restricted to the band, and the
             frequency axis (Hz). The band excludes f = 0 (the (i w)^2
             denominator vanishes there).
    """
    data = np.asarray(data)
    n = data.shape[axis]
    V = np.fft.rfft(data, axis=axis)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))

    lo = 0.0 if f_min is None else float(f_min)
    hi = freqs[-1] if f_max is None else float(f_max)
    sel = (freqs > max(lo, 0.0)) & (freqs <= hi) & (freqs > 0.0)
    if not np.any(sel):
        raise ValueError(f"temporal_fft: empty band [{f_min}, {f_max}] Hz "
                         f"(axis has df={freqs[1]:.4f} Hz).")
    V = np.take(V, np.where(sel)[0], axis=axis)
    return V, freqs[sel]


# ==============================================================
# 1. Time-domain gradiometry (TDG)
# ==============================================================
def _spectral_acceleration(u: np.ndarray, fs: float, *, lag_axis: int = -1) -> np.ndarray:
    """
    Second time-derivative via the FFT: rfft -> x (i 2 pi f)^2 -> irfft.

    A 2nd-order finite difference (np.gradient twice) carries a relative
    bias of (omega dt)^2 / 12 -- already ~8% at 8 Hz with fs = 50 Hz --
    which propagates one-to-one into the TDG sloth. The spectral derivative
    is exact for band-limited input (the masked, tapered gather is), in the
    same spirit as the pseudospectral Laplacian.
    """
    u = np.asarray(u, dtype=np.float64)
    n = u.shape[lag_axis]
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    shape = [1] * u.ndim
    shape[lag_axis] = freqs.size
    mult = (1j * 2.0 * np.pi * freqs.reshape(shape)) ** 2
    return np.fft.irfft(np.fft.rfft(u, axis=lag_axis) * mult, n=n, axis=lag_axis)


def tdg_sloth_pointwise(u: np.ndarray, lap_u: np.ndarray, fs: float,
                        *, lag_axis: int = -1) -> np.ndarray:
    """
    Raw pointwise TDG estimate (Davis et al. 2026, eq. 2):
        s2(x, t) = lap u / d2u/dt2.
    Unstable wherever the acceleration crosses zero; provided for
    completeness and diagnostics. Prefer :func:`tdg_sloth`.
    """
    acc = _spectral_acceleration(u, fs, lag_axis=lag_axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.real(lap_u) / acc


def tdg_sloth(u: np.ndarray, lap_u: np.ndarray, fs: float,
              *, lag_axis: int = -1) -> np.ndarray:
    """
    Least-squares TDG sloth (Davis et al. 2026, eq. 3; Cao et al. 2020):

        s2(x) = sum_t [d2u/dt2 * lap u] / sum_t |d2u/dt2|^2.

    The time stack stabilizes the division (sum of squares in the
    denominator) at the cost of the frequency/depth dependence. The
    acceleration is computed spectrally (see :func:`_spectral_acceleration`)
    to avoid the finite-difference bias of the textbook implementation.

    :param u: real gather (..., nlag) with the lag axis last by default.
    :param lap_u: Laplacian of u (same shape; may be complex -- the real
                  part is used, consistent with a real wavefield).
    :param fs: Sampling rate (Hz).
    :return: real sloth estimate with the lag axis reduced away.
    """
    acc = _spectral_acceleration(u, fs, lag_axis=lag_axis)
    lap = np.real(np.asarray(lap_u))
    num = np.sum(acc * lap, axis=lag_axis)
    den = np.sum(acc * acc, axis=lag_axis)
    den = den + _EPS_REL * (np.max(den) if den.size else 1.0)
    return num / den


# ==============================================================
# 2. Frequency-domain gradiometry (FDG / I-FDG per VSG)
# ==============================================================
def fdg_sloth(V: np.ndarray, lap_V: np.ndarray, freqs: np.ndarray,
              *, freq_axis: int = -1) -> np.ndarray:
    """
    Frequency-domain sloth estimate (Davis et al. 2026, eq. 5; applied to a
    VSG spectrum this is exactly the per-VSG I-FDG estimate, their eq. 9):

        s2(x, w) = V*(x, w) lap V(x, w) / [ (i w)^2 |V(x, w)|^2 ].

    Multiplying by V* (instead of dividing by V) puts the real, non-negative
    |V|^2 in the denominator, so the estimate is stable and stackable.

    :param V: complex spectrum (..., nfreq).
    :param lap_V: spatial Laplacian of V (same shape).
    :param freqs: (nfreq,) frequency axis in Hz (must exclude 0).
    :param freq_axis: Axis carrying frequency.
    :return: complex sloth (..., nfreq). Real part = sloth; imaginary part
             ~ transport-equation residual (quality diagnostic).
    """
    V = np.asarray(V)
    lap_V = np.asarray(lap_V)
    freqs = np.asarray(freqs, dtype=np.float64)
    if np.any(freqs == 0.0):
        raise ValueError("fdg_sloth: frequency axis must exclude 0 Hz.")

    shape = [1] * V.ndim
    shape[freq_axis] = freqs.size
    iw2 = (1j * 2.0 * np.pi * freqs.reshape(shape)) ** 2   # = -(w^2)

    p2 = np.abs(V) ** 2
    floor = _EPS_REL * (np.max(p2) if p2.size else 1.0)
    return (np.conj(V) * lap_V) / (iw2 * (p2 + floor))


def fdg_sloth_stack(V: np.ndarray, lap_V: np.ndarray, freqs: np.ndarray,
                    *, freq_axis: int = -1) -> np.ndarray:
    """
    Frequency-stacked FDG sloth (Davis et al. 2026, eq. 6): the mean of
    :func:`fdg_sloth` over the band. Boosts SNR but discards the frequency
    (hence depth) dependence -- the limitation I-FDG removes.
    """
    return np.mean(fdg_sloth(V, lap_V, freqs, freq_axis=freq_axis), axis=freq_axis)


# ==============================================================
# 3. Interferometric FDG: stacking over virtual sources
# ==============================================================
def ifdg_stack(
    per_vsg_sloth: Sequence[np.ndarray],
    weights: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """
    I-FDG stack over virtual sources (Davis et al. 2026, eq. 10):

        s2(x, w) = (1/Ns) * sum_s  s2(x, x_s, w),

    i.e. the average of the per-VSG ratios -- NOT a ratio of averages. SNR
    grows with the number of virtual sources while the frequency axis (and
    with it depth sensitivity) is preserved.

    Optional per-VSG channel weights generalize eq. 10 to a weighted mean.
    The key application is NEAR-SOURCE EXCLUSION: within ~a wavelength of
    its own virtual source, every VSG is near-field (the Helmholtz
    point-source singularity sits at r = 0) and its sloth estimate is
    biased; weighting those channels to zero per VSG -- while keeping them
    where OTHER sources see them in the far field -- removes the dominant
    error of the plain stack (an order of magnitude in the synthetic
    benchmark, see src/eval.py).

    :param per_vsg_sloth: Sequence of identically shaped complex (nch, nfreq)
                          per-VSG estimates from :func:`fdg_sloth`.
    :param weights: Optional sequence of (nch,) or (nch, nfreq) non-negative
                    weights, one per VSG. Channels with zero total weight
                    come back as NaN.
    :return: complex (nch, nfreq) stacked sloth.
    """
    if len(per_vsg_sloth) == 0:
        raise ValueError("ifdg_stack: no per-VSG estimates supplied.")
    shapes = {np.asarray(a).shape for a in per_vsg_sloth}
    if len(shapes) != 1:
        raise ValueError(f"ifdg_stack: inconsistent shapes {sorted(shapes)}")
    shape = shapes.pop()

    if weights is None:
        acc = np.zeros(shape, dtype=np.complex128)
        for a in per_vsg_sloth:
            acc += np.asarray(a, dtype=np.complex128)
        return acc / float(len(per_vsg_sloth))

    if len(weights) != len(per_vsg_sloth):
        raise ValueError("ifdg_stack: need one weight array per VSG.")
    num = np.zeros(shape, dtype=np.complex128)
    cnt = np.zeros(shape, dtype=np.float64)
    for a, w in zip(per_vsg_sloth, weights):
        w = np.asarray(w, dtype=np.float64)
        if w.ndim == 1:
            w = w[:, None]
        wb = np.broadcast_to(w, shape)
        num += np.asarray(a, dtype=np.complex128) * wb
        cnt += wb
    out = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex128)
    ok = cnt > 0
    out[ok] = num[ok] / cnt[ok]
    return out


# ==============================================================
# 4. Sloth -> phase velocity
# ==============================================================
def sloth_to_velocity(
    s2: np.ndarray,
    *,
    v_min: float | None = None,
    v_max: float | None = None,
) -> np.ndarray:
    """
    Convert a (complex) sloth estimate to phase velocity (Davis et al. 2026):

        V(x, w) = [ Re s2(x, w) ]^(-1/2),

    with non-physical samples set to NaN: Re s2 <= 0 (negative squared
    slowness) and, optionally, velocities outside [v_min, v_max].

    :param s2: complex or real sloth array.
    :param v_min: Optional lower plausibility bound (m/s).
    :param v_max: Optional upper plausibility bound (m/s).
    :return: real velocity array (same shape), NaN where invalid.
    """
    s2r = np.real(np.asarray(s2)).astype(np.float64)
    v = np.full(s2r.shape, np.nan, dtype=np.float64)
    ok = s2r > 0.0
    v[ok] = 1.0 / np.sqrt(s2r[ok])
    if v_min is not None:
        v[v < float(v_min)] = np.nan
    if v_max is not None:
        v[v > float(v_max)] = np.nan
    return v


# ==============================================================
# 5. FK-peak dispersion (calibration / mode-centering reference)
# ==============================================================
def fk_peak_velocity(
    data: np.ndarray,
    dt: float,
    dx: float,
    *,
    f_min: float,
    f_max: float,
    v_min: float = 120.0,
    v_max: float = 900.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apparent phase velocity of the dominant (fundamental) energy per frequency,
    read off the 2-D FK spectrum: at each frequency take the wavenumber of peak
    power within the physical fan [v_min, v_max] and return v = f / |k|.

    This is an INDEPENDENT, ridge-following dispersion estimate (not gradiometry).
    Two uses: (1) a target to calibrate the Laplacian ``k_cutoff`` against, and
    (2) a centre velocity for a narrowband fundamental-mode FK extract. Because
    the FK peak tracks the strongest ridge, it is robust to the high-k tail that
    biases the pointwise k^2-weighted sloth low.

    :param data: (nch, nlag) real VSG (channels x lag).
    :param dt: Lag sampling (s).
    :param dx: Channel spacing (m).
    :param f_min: Lower frequency bound (Hz).
    :param f_max: Upper frequency bound (Hz).
    :param v_min: Slow edge of the search fan (m/s).
    :param v_max: Fast edge of the search fan (m/s).
    :return: (freqs, c_peak) -- 1-D arrays over the in-band positive
             frequencies; ``c_peak`` is NaN where no in-fan energy exists.
    """
    data = np.asarray(data, dtype=np.float64)
    nx, nt = data.shape
    P = np.abs(np.fft.fftshift(np.fft.fft2(data))) ** 2
    k = np.fft.fftshift(np.fft.fftfreq(nx, d=float(dx)))   # cycles/m
    f = np.fft.fftshift(np.fft.fftfreq(nt, d=float(dt)))   # Hz
    in_band = np.where((f >= float(f_min)) & (f <= float(f_max)) & (f > 0.0))[0]
    freqs = f[in_band]
    c_peak = np.full(freqs.size, np.nan, dtype=np.float64)
    for i, jf in enumerate(in_band):
        with np.errstate(divide="ignore"):
            v = np.where(k != 0.0, f[jf] / k, np.inf)
        good = (np.abs(v) >= v_min) & (np.abs(v) <= v_max)
        if good.any():
            row = np.where(good, P[:, jf], 0.0)
            c_peak[i] = abs(f[jf] / k[int(np.argmax(row))])
    return freqs, c_peak


# ==============================================================
# 6. Stack sloth
# ==============================================================
def stack_sloth(files, *, f_min, f_max, k_cutoff=None, r_exclude_m=150.0,
                aperture_m=350.0, channel_taper_alpha=0.3, include_curvature=True,
                stack_method='median', quality_max=1.0):
    """Per-VSG I-FDG sloth (eq. 9) -> position-aligned stack (eq. 10).
    
    Gathers are centred on their own VS and differ in size, so they are aligned
    on the absolute-channel grid (VS index + row - zero-offset row). s1 and s2
    gathers of the same VS land on the same positions, so passing both simply
    doubles the samples that feed the robust stack.

    NOTE: Memory-optimized implementation. Evaluates spatial overlaps densely to
    avoid constructing a mostly-NaN (n_sources, n_grid, n_freq) matrix that would
    cause Out-Of-Memory crashes on large 10,000+ channel DAS arrays.

    aperture_m   : trim each gather to |offset| <= aperture_m before the
                   Laplacian (both limbs kept), then channel-taper the kept
                   window. ~350 m is the sweet spot here (coherence peaks, gaps 0):
                   a full limb on both sides (all truncated) and the far offsets
                   are low-SNR/scattered (high-k bias); <~250 m raises the
                   aperture floor f_min ~ 3c/aperture into the band. None keeps
                   the full window.
    channel_taper_alpha : Tukey fraction applied across channels after the
                   aperture clip so the channel-axis FFT in the pseudospectral
                   Laplacian sees no step at the trimmed edge (it assumes
                   periodicity). 0 disables.
    stack_method : 'median' (robust -- rejects nodal/outlier pixels; default)
                   or 'mean' (Davis eq. 10).
    quality_max  : if set, mask pixels where |Im/Re| of the stacked sloth (the
                   transport-equation residual) exceeds it -- a reliability flag.
    Returns (positions_m, freqs, s2, coverage), coverage = #gathers/pixel.
    """
    per, starts, offsets = [], [], []
    freqs, dx0 = None, None
    
    for f in files:
        a = np.load(f)
        d, lg, off = a['data'], a['lag'], a['offset']
        
        # Cache dx0 so we don't have to hit the disk again later
        if dx0 is None and len(off) > 1:
            dx0 = float(abs(off[1] - off[0]))
            
        if aperture_m is not None:                      
            keep_ap = np.abs(off) <= aperture_m
            d, off = d[keep_ap], off[keep_ap]
            
        if d.shape[0] < 5:                              
            continue
            
        if channel_taper_alpha:                         
            d = d * cosine_taper(d.shape[0], channel_taper_alpha)[:, None]
            
        fs = 1.0 / (lg[1] - lg[0])
        dxi = float(abs(off[1] - off[0]))
        
        V, fr = temporal_fft(d, fs, f_min=f_min, f_max=f_max, axis=-1)
        lapV = laplacian_fiber(V, dxi, offset=off,
                               include_curvature=include_curvature, k_cutoff=k_cutoff)
        
        per.append(fdg_sloth(V, lapV, fr, freq_axis=-1))
        starts.append(get_vs_number(f) - int(np.argmin(np.abs(off))))
        offsets.append(off)
        freqs = fr
        
    if not per:
        raise ValueError("stack_sloth: No valid gathers found to stack.")
        
    gmin = min(starts)
    gmax = max(s + p.shape[0] for s, p in zip(starts, per))
    ng, nf = gmax - gmin, freqs.size
    
    # -------------------------------------------------------------------------
    # MEMORY OPTIMIZATION: List of Lists
    # Instead of allocating a 160 GB `(n_sources, ng, nf)` matrix filled with NaNs, 
    # we construct a sparse map of only the overlapping traces per absolute grid point.
    # -------------------------------------------------------------------------
    grid_data = [[] for _ in range(ng)]
    
    for s2i, st, off in zip(per, starts, offsets):
        for local_idx in range(s2i.shape[0]):
            if np.abs(off[local_idx]) >= r_exclude_m:
                abs_grid_idx = st + local_idx - gmin
                grid_data[abs_grid_idx].append(s2i[local_idx, :])
                
    s2 = np.full((ng, nf), np.nan + 0j, dtype=np.complex128)
    coverage = np.zeros(ng, dtype=int)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for i in range(ng):
            if not grid_data[i]:
                continue
                
            arr = np.array(grid_data[i]) # Shape: (n_overlapping_sources, nf)
            coverage[i] = len(arr)
            
            if stack_method == 'median':
                s2[i] = np.nanmedian(arr.real, axis=0) + 1j * np.nanmedian(arr.imag, axis=0)
            else:
                s2[i] = np.nanmean(arr, axis=0)
                
    if quality_max is not None:
        q = np.abs(s2.imag) / (np.abs(s2.real) + 1e-30)
        s2[q > quality_max] = np.nan
        
    if dx0 is None:
        dx0 = 1.0 # Safe fallback
        
    positions = (gmin + np.arange(ng)) * dx0
    return positions, freqs, s2, coverage