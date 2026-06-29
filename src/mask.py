"""
:module: src/mask.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Torus (moveout-annulus) masking of virtual source gathers.

Gradiometry assumes a SINGLE surface-wave mode (the 2-D scalar wave
equation holds mode by mode). A raw VSG additionally contains guided-P
energy, higher-mode surface waves, acausal scattering, and correlation
noise. Following Davis et al. (2026, Sec. 3.2), everything except the
fundamental-mode direct arrival is muted with a "torus filter": because the
surface wave expands outward with increasing |lag|, the keep-region in
(offset, lag) space is an annulus that expands with lag (a torus in map
view; a cone in the 1-D fiber section).

Keep-window for a channel at distance r from the virtual source:

    tau in [ r / v_outer + t_outer ,  r / v_inner + t_inner ]

with raised-cosine ramps of ``taper_sec`` on both edges. Davis et al. used
v_inner = v_outer = 0.83 km/s, t_inner = +5 s, t_outer = -5 s for Scholte
waves -- i.e. a constant-velocity moveout corridor of half-width 5 s.
"""
from __future__ import annotations

import logging

import os
import numpy as np

from typing import List, Union
from tqdm import tqdm

from src.utils import fk_filter

logger = logging.getLogger(__name__)


def _edge_ramp(t: np.ndarray, t0: float, taper: float, rising: bool) -> np.ndarray:
    """Raised-cosine ramp: 0 before/after t0 -+ taper, 1 on the keep side."""
    if taper <= 0:
        return (t >= t0).astype(np.float64) if rising else (t <= t0).astype(np.float64)
    x = (t - t0) / taper
    if not rising:
        x = -x
    ramp = 0.5 * (1.0 + np.sin(np.pi * np.clip(x, -0.5, 0.5)))
    return ramp


def torus_mask(
    lag: np.ndarray,
    offset: np.ndarray,
    *,
    v_inner: float,
    v_outer: float,
    t_inner: float,
    t_outer: float,
    taper_sec: float = 1.0,
    causal_only: bool = True,
) -> np.ndarray:
    """
    Build the (nch, nlag) torus-mask weights for one VSG.

    :param lag: (nlag,) lag axis in seconds (symmetric about 0).
    :param offset: (nch,) signed offsets from the virtual source (m).
    :param v_inner: Velocity (m/s) of the *late* (inner-annulus) mute edge.
    :param v_outer: Velocity (m/s) of the *early* (outer-annulus) mute edge.
    :param t_inner: Time pad (s) added to the late edge (r/v_inner + t_inner).
    :param t_outer: Time pad (s) added to the early edge (r/v_outer + t_outer);
                    typically negative (Davis et al.: -5 s).
    :param taper_sec: Raised-cosine ramp length (s) at each mute edge.
    :param causal_only: If True (Davis et al.), keep only tau > 0; otherwise
                        the window is mirrored onto the acausal side as well.
    :return: (nch, nlag) float64 weights in [0, 1].

    Notes:
    - With v_inner = v_outer = v, the window is a constant-width corridor
      around the moveout tau = r/v, exactly the Davis et al. torus.
    - The mask is purely geometric; apply it multiplicatively to the VSG
      *before* the temporal FFT.
    """
    if v_inner <= 0 or v_outer <= 0:
        raise ValueError("torus_mask: velocities must be > 0.")
    lag = np.asarray(lag, dtype=np.float64)
    r = np.abs(np.asarray(offset, dtype=np.float64))[:, None]     # (nch, 1)
    t = lag[None, :]                                              # (1, nlag)

    t_early = r / float(v_outer) + float(t_outer)   # window start (causal side)
    t_late = r / float(v_inner) + float(t_inner)    # window end
    if np.any(t_late <= t_early):
        bad = np.sum(t_late <= t_early)
        logger.warning(
            "torus_mask: %d channels have an empty keep-window "
            "(check v/t parameters).", int(bad),
        )

    # Causal-side corridor: rising ramp at the early edge, falling at the late.
    w_causal = (_edge_ramp(t - t_early, 0.0, taper_sec, rising=True)
                * _edge_ramp(t - t_late, 0.0, taper_sec, rising=False))

    if causal_only:
        w = w_causal * (t >= 0)
    else:
        # Mirror the corridor onto the acausal side (tau -> -tau).
        w_acausal = (_edge_ramp(-t - t_early, 0.0, taper_sec, rising=True)
                     * _edge_ramp(-t - t_late, 0.0, taper_sec, rising=False))
        w = np.maximum(w_causal * (t >= 0), w_acausal * (t <= 0))

    return np.ascontiguousarray(np.broadcast_to(w, (r.shape[0], lag.size)))


def channel_taper_weights(nch: int, alpha: float) -> np.ndarray:
    """
    Tukey taper along the CHANNEL axis, applied before the spatial FFT so
    the pseudospectral Laplacian does not see a discontinuity at the array
    ends (the spatial FFT assumes periodicity; cf. Davis et al. 2026,
    'suppress edge artifacts').

    :param nch: Number of channels.
    :param alpha: Tukey shape parameter (fraction tapered, e.g. 0.2).
    :return: (nch,) weights.
    """
    from src.utils import cosine_taper
    return cosine_taper(nch, alpha)


def apply_mask(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Multiplicative mask with shape checking; returns float64."""
    data = np.asarray(data, dtype=np.float64)
    if data.shape != weights.shape:
        raise ValueError(f"apply_mask: shape mismatch {data.shape} vs {weights.shape}")
    return data * weights


def apply_torus(
    data: np.ndarray,
    lag: np.ndarray,
    offset: np.ndarray,
    **torus_kwargs,
) -> np.ndarray:
    """
    Convenience one-call torus mask: ``apply_mask(data, torus_mask(...))``.

    Handy for inline experimentation in a notebook on a single loaded gather;
    for batch file->file processing use :func:`torus_batch_processor`.

    :param data: (nch, nlag) VSG amplitudes.
    :param lag: (nlag,) lag axis (s).
    :param offset: (nch,) signed offsets from the virtual source (m).
    :param torus_kwargs: forwarded to :func:`torus_mask` (v_inner, v_outer,
                         t_inner, t_outer, taper_sec, causal_only).
    :return: (nch, nlag) masked gather (float64).
    """
    return apply_mask(np.asarray(data, dtype=np.float64),
                      torus_mask(lag, offset, **torus_kwargs))


def torus_batch_processor(
    file_list: List[str],
    out_dir: str,
    *,
    v_inner: float = 300.0,
    v_outer: float = 600.0,
    t_inner: float = 0.25,
    t_outer: float = -0.25,
    taper_sec: float = 0.15,
    causal_only: bool = True,
    channel_taper_alpha: float = 0.0,
    range_m: float | None = None,
    max_lag: float | None = None,
) -> None:
    """
    Batch torus-mask wrapper -- the Davis et al. (2026) counterpart to
    :func:`fast_batch_processor`, with the SAME file->file interface so the
    notebook can call it and re-plot the result identically.

    **How this differs from** :func:`fast_batch_processor` (they are NOT the
    same operation): ``fast_batch_processor`` isolates the mode in the
    frequency-wavenumber domain via an f-k velocity-fan extract
    (:func:`src.utils.fk_filter`) followed by spatial/time-cone mutes;
    ``torus_batch_processor`` applies only the time-domain moveout-corridor
    weight :func:`torus_mask` (keep apparent velocity in [v_inner, v_outer]
    per trace). The torus is softer and mode-by-velocity in the time domain;
    the f-k fan separates dispersive/overlapping modes more sharply. Use the
    torus when you want the gradiometry-faithful Davis operator; use the f-k
    pipeline when guided-P / higher modes must be suppressed by velocity.

    Saves ``{name}_torus_{v_inner}_{v_outer}.npz`` carrying ``data``, ``lag``,
    ``offset`` (plus the torus parameters), matching the keys that
    :func:`src.plots.plot_vsg` / ``plot_fk`` expect.

    :param file_list: Paths to the input ``.npz`` VSGs (data, lag, offset).
    :param out_dir: Output directory (created if missing).
    :param v_inner: Slow (late) corridor-edge velocity (m/s).
    :param v_outer: Fast (early) corridor-edge velocity (m/s).
    :param t_inner: Time pad (s) on the late edge (r/v_inner + t_inner).
    :param t_outer: Time pad (s) on the early edge (r/v_outer + t_outer).
    :param taper_sec: Raised-cosine ramp length (s) at each corridor edge.
    :param causal_only: Keep only tau > 0 (Davis convention). Default True.
    :param channel_taper_alpha: If > 0, Tukey-taper the channel axis after
                                masking (limits spatial-FFT wraparound).
    :param range_m: If set, zero traces with |offset| > range_m.
    :param max_lag: If set, crop the lag axis to |lag| <= max_lag before saving.
    :return: None (writes one ``.npz`` per input file).
    """
    os.makedirs(out_dir, exist_ok=True)
    for path in tqdm(file_list, desc="Torus masking NCFs"):
        try:
            arc = np.load(path)
            data, lag, offset = arc["data"], arc["lag"], arc["offset"]
            w = torus_mask(lag, offset, v_inner=v_inner, v_outer=v_outer,
                           t_inner=t_inner, t_outer=t_outer,
                           taper_sec=taper_sec, causal_only=causal_only)
            out = apply_mask(data, w)
            if channel_taper_alpha and channel_taper_alpha > 0:
                out = out * channel_taper_weights(out.shape[0], channel_taper_alpha)[:, None]
            if range_m is not None:
                out[np.abs(offset) > float(range_m), :] = 0.0
            if max_lag is not None:
                keep = np.abs(lag) <= float(max_lag)
                lag, out = lag[keep], out[:, keep]
            name, ext = os.path.splitext(os.path.basename(path))
            save_path = os.path.join(out_dir, f"{name}_torus_{int(v_inner)}_{int(v_outer)}{ext}")
            np.savez_compressed(
                save_path, data=out, lag=lag, offset=offset,
                v_inner=v_inner, v_outer=v_outer, t_inner=t_inner,
                t_outer=t_outer, taper_sec=taper_sec,
            )
        except Exception as e:
            print(f"\nError processing {os.path.basename(path)}: {e}")


# ==============================================================
# Legacy code
# ==============================================================
def apply_inner_spatial_mute(data: np.ndarray, offset: np.ndarray, cut_m: float = 100.0, taper_m: float = 50.0) -> np.ndarray:
    """
    Zeros out the near-offset traces and applies a smooth cosine taper to the transition 
    zone to prevent spectral ringing (Gibbs phenomenon) in the f-k domain.
    
    :param data: 2D numpy array of seismic data.
    :param offset: 1D array of spatial offsets.
    :param cut_m: The absolute distance (in meters) to hard-mute.
    :param taper_m: The distance (in meters) over which to fade the data back in.
    :returns: Muted and tapered 2D numpy array.
    """
    dist = np.abs(offset)
    weights = np.ones_like(dist)
    weights[dist <= cut_m] = 0.0
    
    transition = (dist > cut_m) & (dist < cut_m + taper_m)
    norm_dist = (dist[transition] - cut_m) / taper_m
    weights[transition] = 0.5 * (1 - np.cos(np.pi * norm_dist))
    
    return data * weights[:, np.newaxis]

def apply_time_distance_mute(
    data: np.ndarray, 
    offset: np.ndarray, 
    lag: np.ndarray, 
    vmin: float, 
    vmax_time: float,            
    buffer_start_s: float = 0.2, 
    buffer_end_s: float = 1.0, 
    top_flat_m: float = 100.0
) -> np.ndarray:
    """
    Zeros out data outside the physical travel-time cone.
    
    Uses independent buffers for the fast leading edge and the slow trailing coda 
    to preserve both the early arrivals and the ringing tail of the wave packets.

    :param data: 2D array containing the seismic data (e.g., cross-correlations).
    :param offset: 1D array of absolute or relative spatial offsets in meters.
    :param lag: 1D array of time lags in seconds.
    :param vmin: Minimum expected phase velocity in m/s (defines the trailing edge).
    :param vmax_time: Maximum group velocity in m/s (defines the leading edge).
    :param buffer_start_s: Time buffer in seconds to keep before the fastest wave arrival.
    :param buffer_end_s: Time buffer in seconds to keep after the slowest wave arrival.
    :param top_flat_m: Distance in meters from the source where the upper mute remains flat.
    :returns: A muted 2D array where data outside the calculated time-cone is set to zero.
    """
    nx, nt = data.shape
    abs_offset = np.abs(offset)
    mask = np.zeros_like(data)
    
    for i in range(nx):
        # 1. Top wedge (Fast leading edge) with a flat top
        # This keeps effective_x at 0 until we pass your threshold
        effective_x = max(0, abs_offset[i] - top_flat_m)
        t_start = max(0, (effective_x / vmax_time) - buffer_start_s)   
        
        # 2. Bottom wedge (Slow trailing coda)
        t_end = (abs_offset[i] / vmin) + buffer_end_s
        
        mask[i, (lag >= t_start) & (lag <= t_end)] = 1.0
        
    return data * mask

def apply_spatial_taper(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Applies a Tukey (tapered cosine) window to the spatial axis of the data to gently 
    fade the far-offset edges to zero.
    
    :param data: 2D numpy array of seismic data.
    :param alpha: Fraction of the window inside the cosine tapered region (0 to 1).
    :returns: Tapered 2D numpy array.
    """
    if alpha <= 0:
        return data
    nx = data.shape[0]

    from scipy.signal.windows import tukey
    window = tukey(nx, alpha=alpha)
    return data * window[:, np.newaxis]


def fast_batch_processor(
    file_list: List[str],
    out_dir: str,
    vmin: float = 150.0,
    vmax: float = 2000.0,
    vmax_time: float | None = None,
    pos_offset: float = 100.0,
    inner_taper: float = 50.0,
    range_m: float = 4000.0,
    sigma: float = 1.0, 
    buffer_start_s: float = 0.2, 
    buffer_end_s: float = 1.5, 
    top_flat_m: float = 100.0,
    max_lag: float | None = None
) -> None:
    """
    High-speed, headless pipeline execution for Noise Correlation Functions (NCFs).
    
    Strips out all plotting logic to maximize processing speed. This function reads raw NCF 
    data, applies a Gaussian-smoothed f-k filter, cleans the arrays using a cascaded 
    series of spatial and temporal mutes/tapers, and optionally crops edge artifacts. 
    The resulting arrays are safely saved to disk.

    :param file_list: List of absolute or relative file paths to the raw `.npz` files.
    :param out_dir: Destination directory where the processed `.npz` files will be saved.
    :param vmin: Minimum expected phase velocity in m/s. Used for both the f-k filter 
                 boundaries and the time-distance travel cone mute. Defaults to 150.0.
    :param vmax: Maximum expected phase velocity in m/s (F-K Phase Velocity). Defaults to 2000.0.
    :param vmax_time: Maximum group velocity in m/s to use for the time-distance mute. If None, defaults to `vmax`.
    :param pos_offset: Absolute distance in meters from the virtual source to completely 
                       zero out (hard mute). Prevents near-offset artifacting. Defaults to 100.0.
    :param inner_taper: Distance in meters over which to smoothly fade the data back in 
                        after the `pos_offset` cut, preventing spectral ringing. Defaults to 50.0.
    :param range_m: Maximum absolute distance in meters to retain. Traces beyond this 
                    offset are completely zeroed out. Defaults to 4000.0.
    :param sigma: Standard deviation for the F-K filter's Gaussian smoothing. Higher 
                  values create smoother filter edges. Defaults to 1.0.
    :param buffer_start_s: Time buffer in seconds to keep before the fastest wave arrival in the time-distance mute. Defaults to 0.2s.
    :param buffer_end_s: Time buffer in seconds to keep after the slowest wave arrival in the time-distance mute. Defaults to 1.5s.
    :param top_flat_m: Distance in meters from the source where the upper mute remains flat, preserving early arrivals. Defaults to 100.0m.
    :param max_lag: Optional maximum absolute lag time (s) to retain. Physically crops the 
                    data arrays after processing to remove edge artifacts before saving.
    :returns: None. The function saves processed `.npz` files directly to `out_dir` with 
              the naming convention: `[original_name]_fk_[vmin]_[vmax].npz`.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Handle decoupled velocity logic
    if vmax_time is None:
        vmax_time = vmax
        
    for path in tqdm(file_list, desc="Processing & Saving NCFs"):
        try:
            # 2. Load Data
            archive = np.load(path)
            data = archive['data']   
            lag = archive['lag']
            offset = archive['offset']
            dt = lag[1] - lag[0]
            dx = np.abs(offset[1] - offset[0])
            
            # 3. Stage 1: F-K Filter (Uses Phase Velocity: vmax)
            data_clean = fk_filter(
                data, dt=dt, dx=dx, vmin=vmin, vmax=vmax,
                mode="extract", direction="both", smooth="gaussian", sigma=sigma
            )

            # 4. Stage 2: Cascaded Mutes and Tapers
            dist = np.abs(offset)
            data_clean = apply_inner_spatial_mute(data_clean, offset, cut_m=pos_offset, taper_m=inner_taper)
            data_clean[dist > range_m, :] = 0.0
            
            # Use Group Velocity (vmax_time) and Asymmetric Buffers for the Time Cone!
            data_clean = apply_time_distance_mute(
                data_clean, offset, lag, vmin, vmax_time, 
                buffer_start_s=buffer_start_s, buffer_end_s=buffer_end_s, top_flat_m=top_flat_m
            )
            data_clean = apply_spatial_taper(data_clean, alpha=0.1)

            # Crop Edge Artifacts 
            if max_lag is not None:
                lag_mask = np.abs(lag) <= max_lag
                lag = lag[lag_mask]
                data_clean = data_clean[:, lag_mask]

            # 5. Save with exact requested naming convention
            filename = os.path.basename(path)
            name_base, ext = os.path.splitext(filename)
            new_filename = f"{name_base}_fk_{int(vmin)}_{int(vmax)}{ext}"
            save_path = os.path.join(out_dir, new_filename)
            
            np.savez_compressed(
                save_path,
                data=data_clean,
                lag=lag,        
                offset=offset,
                vmin=vmin,
                vmax=vmax,
                vmax_time=vmax_time,   
                pos_offset=pos_offset
            )
            
        except Exception as e:
            print(f"\nError processing {os.path.basename(path)}: {e}")