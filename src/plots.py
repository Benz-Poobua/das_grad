"""
:module: src/plots.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Plotting utilities for DAS interferometric frequency-domain
          gradiometry (I-FDG): virtual source gathers, the sloth and
          phase-velocity frequency-distance panels (Davis et al. 2026, Figs
          7-9), and dispersion cross-checks against das_ani picks.

Style mirrors das_ani/src/plots.py and src/inv.py (Sphinx reST docstrings,
pcolormesh with 'gouraud' shading, turbo/seismic colormaps, 300 dpi saves).
The module is import-light: numpy + matplotlib only.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from src.utils import fk_transform, parse_ncf_stack_filename

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]

# ===========================================================================
# Unified Plotting Configuration
# ===========================================================================
_PLOT_CONFIG = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "font.size": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": [10, 5],
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
}
rcParams.update(_PLOT_CONFIG)


def _save(fig: plt.Figure, save_path: Optional[PathLike]) -> None:
    """
    Save a figure to disk with standard bounding and facecolor settings.
    
    :param fig: The Matplotlib figure to save.
    :type fig: plt.Figure
    :param save_path: The destination path for the saved figure.
    :type save_path: Optional[PathLike]
    """
    if save_path:
        p = Path(save_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        logger.info("Saved figure -> %s", p)


# ==============================================================
# 1. Virtual source gather (QC)
# ==============================================================
def plot_vsg(
    files: Sequence[PathLike],
    VS: Union[str, int],
    *,
    unit: str = "m",
    clip: Optional[float] = 0.05,
    pclip: Optional[float] = None,
    cmap: str = "seismic",
    range_m: float = 4000.0,
    clip_lim: bool = True,
    view_side: Literal["both", "left", "right"] = "both",
    pos_offset: float = 0.0,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show_cbar: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a static Noise Cross-Correlation Function (NCF) gather for a single Virtual Source.

    :param files: List of file paths to the pre-processed NCF numpy archives (.npz).
    :type files: Sequence[PathLike]
    :param VS: Virtual Source number to plot (e.g., 5 or "005").
    :type VS: Union[str, int]
    :param unit: Spatial distance unit for the x-axis ("m" or "km"), by default "m".
    :type unit: str
    :param clip: Absolute amplitude limit for colorbar scaling, by default 0.05.
    :type clip: Optional[float]
    :param pclip: Percentile for dynamic amplitude clipping (e.g., 99.0), overrides `clip`.
    :type pclip: Optional[float]
    :param cmap: Matplotlib colormap to use, by default "seismic".
    :type cmap: str
    :param range_m: Maximum spatial distance in meters to display, by default 4000.0.
    :type range_m: float
    :param clip_lim: If True, strictly limits x-axis bounds based on `range_m`, by default True.
    :type clip_lim: bool
    :param view_side: Determines which side of the gather to display, by default "both".
    :type view_side: Literal["both", "left", "right"]
    :param pos_offset: Spatial exclusion offset from the source to clip near-field noise, by default 0.0.
    :type pos_offset: float
    :param figsize: Custom figure dimensions (width, height) in inches.
    :type figsize: Optional[Tuple[float, float]]
    :param title: Custom plot title. If None, auto-generates from metadata.
    :type title: Optional[str]
    :param show_cbar: Toggle visibility of the colorbar, by default True.
    :type show_cbar: bool
    :returns: The constructed Matplotlib Figure and Axes objects.
    :rtype: Tuple[plt.Figure, plt.Axes]
    """
    if not files:
        raise ValueError("Provided file list is empty!")

    view_side_clean = view_side.lower().strip()
    unit_clean = unit.lower().strip()
    dist_scale = 1000.0 if unit_clean == "km" else 1.0
    plot_range = range_m / dist_scale
    plot_offset = pos_offset / dist_scale

    target_path = None
    target_date, target_window, target_vs_str, target_xmode = "", "", "", ""

    for p in files:
        date, vs_str, window, xmode = parse_ncf_stack_filename(str(p))
        if int(vs_str) == int(VS):
            target_path = p
            target_date, target_window, target_vs_str, target_xmode = date, window, vs_str, xmode
            break

    if target_path is None:
        raise FileNotFoundError(f"Could not find a file matching VS={VS} in the provided file list.")

    archive = np.load(target_path)
    current_offset = archive["offset"] / dist_scale
    lag_axis = archive["lag"]
    data = archive["data"].T

    if pclip is not None:
        c0 = float(np.percentile(np.abs(data), pclip))
    else:
        c0 = float(clip if clip is not None else 1.0)

    if view_side_clean == "both":
        left_bound, right_bound = -plot_range, plot_range
    elif view_side_clean == "right":
        left_bound, right_bound = plot_offset, plot_range
    else:
        left_bound, right_bound = -plot_range, -plot_offset

    if figsize is None:
        figsize = (8, 6) if clip_lim else (10, 6)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.invert_yaxis()
    ax.set_xlabel(f"Offset from Virtual Source ({unit_clean})")
    ax.set_ylabel("Lag time (s)")

    if clip_lim:
        ax.set_xlim(left_bound, right_bound)

    mesh = ax.pcolormesh(current_offset, lag_axis, data, shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0)
    ax.axvline(x=0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)

    if show_cbar:
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label("Correlation amplitude")

    if title is None:
        title = f"NCF Gather (VS={target_vs_str} | {target_date} | {target_xmode})"
    ax.set_title(title)

    return fig, ax

# ==============================================================
# 2. FK 
# ==============================================================

def animate_vsg(
    files: Sequence[PathLike],
    *,
    unit: str = "m",
    clip: Optional[float] = 0.05,
    pclip: Optional[float] = None,
    cmap: str = "seismic",
    range_m: float = 4000.0,
    clip_lim: bool = True,
    view_side: Literal["both", "left", "right"] = "both",
    pos_offset: float = 0.0,
    interval_ms: int = 200,
    save_vs: Optional[Sequence[int]] = None,
    save_dir: PathLike = "./saved_figures",
    save_fmt: str = "png",
    save_dpi: int = 300,
) -> FuncAnimation:
    """
    Animate spatially and temporally pre-processed NCF gathers.

    :param files: List of file paths to the pre-processed NCF numpy archives.
    :type files: Sequence[PathLike]
    :param unit: Spatial distance unit ("m" or "km"), by default "m".
    :type unit: str
    :param clip: Absolute amplitude limit for scaling, by default 0.05.
    :type clip: Optional[float]
    :param pclip: Percentile for dynamic amplitude clipping across early frames.
    :type pclip: Optional[float]
    :param cmap: Matplotlib colormap, by default "seismic".
    :type cmap: str
    :param range_m: Maximum spatial distance in meters to display, by default 4000.0.
    :type range_m: float
    :param clip_lim: Strictly limit x-axis bounds, by default True.
    :type clip_lim: bool
    :param view_side: Side of the gather to display, by default "both".
    :type view_side: Literal["both", "left", "right"]
    :param pos_offset: Near-source exclusion offset, by default 0.0.
    :type pos_offset: float
    :param interval_ms: Delay between frames in milliseconds, by default 200.
    :type interval_ms: int
    :param save_vs: Specific Virtual Source IDs to export as static images.
    :type save_vs: Optional[Sequence[int]]
    :param save_dir: Output directory for saved static frames, by default "./saved_figures".
    :type save_dir: PathLike
    :param save_fmt: Image format for static exports, by default "png".
    :type save_fmt: str
    :param save_dpi: Resolution for exported frames, by default 300.
    :type save_dpi: int
    :returns: The Matplotlib animation object.
    :rtype: FuncAnimation
    """
    if not files:
        raise ValueError("Provided file list is empty!")

    view_side_clean = view_side.lower().strip()
    unit_clean = unit.lower().strip()
    dist_scale = 1000.0 if unit_clean == "km" else 1.0
    plot_range = range_m / dist_scale
    plot_offset = pos_offset / dist_scale

    parsed = []
    for p in files:
        date, vs, window, xmode = parse_ncf_stack_filename(str(p))
        parsed.append((p, date, vs, xmode))

    if pclip is not None:
        sub_scan = [np.percentile(np.abs(np.load(p[0])["data"]), pclip) for p in parsed[:50]]
        c0 = float(np.median(sub_scan))
    else:
        c0 = float(clip if clip is not None else 1.0)

    if view_side_clean == "both":
        left_bound, right_bound = -plot_range, plot_range
    elif view_side_clean == "right":
        left_bound, right_bound = plot_offset, plot_range
    else:
        left_bound, right_bound = -plot_range, -plot_offset

    fig, ax = plt.subplots(figsize=(8, 6) if clip_lim else (10, 6), layout="constrained")
    ax.invert_yaxis()
    ax.set_xlabel(f"Offset from Virtual Source ({unit_clean})")
    ax.set_ylabel("Lag time (s)")
    if clip_lim:
        ax.set_xlim(left_bound, right_bound)

    archive0 = np.load(parsed[0][0])
    mesh = ax.pcolormesh(
        archive0["offset"] / dist_scale, archive0["lag"], archive0["data"].T,
        shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0
    )
    vline = ax.axvline(x=0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label("Correlation amplitude")

    title_text = ax.set_title(f"NCF Gather (VS={parsed[0][2]} | {parsed[0][1]} | {parsed[0][3]})")

    processed_frames = set()
    saved_frames = set()
    total_frames = len(parsed)
    pbar = tqdm(total=total_frames, desc="Rendering Video")

    def update(frame_idx: int):
        nonlocal mesh
        path, date, vs, xmode = parsed[frame_idx]
        archive = np.load(path)

        mesh.remove()
        mesh = ax.pcolormesh(
            archive["offset"] / dist_scale, archive["lag"], archive["data"].T,
            shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0
        )
        title_text.set_text(f"NCF Gather (VS={vs} | {date} | {xmode})")

        if save_vs is not None and int(vs) in save_vs and frame_idx not in saved_frames:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            export_path = out_dir / f"NCF_Gather_VS_{vs}.{save_fmt}"
            fig.savefig(export_path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
            saved_frames.add(frame_idx)

        if frame_idx not in processed_frames:
            pbar.update(1)
            processed_frames.add(frame_idx)
            if len(processed_frames) == total_frames:
                pbar.close()

        return mesh, vline, title_text

    return FuncAnimation(fig, update, frames=total_frames, interval=interval_ms, blit=False)

def plot_fk(
    files: List[str],
    VS: Union[str, int],
    *,
    unit: str = "m",
    clip: float | None = None,
    pclip: float | None = 99.0,
    cmap: str = "inferno",
    view_side: Literal["both", "left", "right"] = "right",
    pos_offset: float = 0.0,
    klim: Tuple[float, float] | None = None,
    figsize: Tuple[float, float] = (8, 6),
    vmin: float | None = None,                 
    vmax: float | None = None,
    title: str | None = None,
    show_cbar: bool = True,
    dpi: int = 120,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a static, normalized 2D frequency-wavenumber (f-k) power spectrum 
    for a single pre-processed Virtual Source (VS).

    This function isolates directional wavefields, dynamically calculates the f-k 
    transform to handle missing channels safely, and provides reference phase-velocity 
    overlays for dispersion analysis.

    :param files: List of file paths to the pre-processed NCF numpy archives (.npz).
    :type files: List[str]
    :param VS: Virtual Source number to plot (e.g., 5 or "005").
    :type VS: Union[str, int]
    :param unit: Spatial distance unit used for the x-axis ("m" or "km"). Default is "m".
    :type unit: str
    :param clip: Absolute amplitude limit for power normalization. Used only if `pclip` is None.
    :type clip: Optional[float]
    :param pclip: Percentile for dynamic amplitude clipping (e.g., 99.0).
    :type pclip: Optional[float]
    :param cmap: Matplotlib colormap to use. Default is "inferno".
    :type cmap: str
    :param view_side: Determines which side of the spatial array to process and display ("both", "left", or "right"). 
    :type view_side: Literal["both", "left", "right"]
    :param pos_offset: Spatial exclusion offset to clip out near-source auto-correlation artifacts.
    :type pos_offset: float
    :param klim: Optional tuple (kmin, kmax) specifying the wavenumber (x-axis) limits. 
    :type klim: Optional[Tuple[float, float]]
    :param figsize: Tuple specifying the figure dimensions.
    :type figsize: Tuple[float, float]
    :param vmin: Optional minimum phase velocity (m/s). Plots a cyan dashed reference line.
    :type vmin: Optional[float]
    :param vmax: Optional maximum phase velocity (m/s). Plots a lime dashed reference line.
    :type vmax: Optional[float]
    :param title: Custom title for the plot. If None, auto-generates one.
    :type title: Optional[str]
    :param show_cbar: Toggle visibility of the power amplitude colorbar.
    :type show_cbar: bool
    :param dpi: Resolution of the output plot.
    :type dpi: int
    :returns: A tuple containing the (Figure, Axes) objects.
    :rtype: Tuple[plt.Figure, plt.Axes]
    """
    if not files:
        raise ValueError("Provided file list is empty!")

    view_side, unit = view_side.lower().strip(), unit.lower().strip()
    dist_scale = 1000.0 if unit == "km" else 1.0

    # Locate the target file
    target_path = None
    target_date, target_vs_str, target_xmode = "", "", ""

    for p in files:
        date, vs_str, window, xmode = parse_ncf_stack_filename(p)
        if int(vs_str) == int(VS):
            target_path = p
            target_date = date
            target_vs_str = vs_str
            target_xmode = xmode
            break

    if target_path is None:
        raise FileNotFoundError(f"Could not find a file matching VS={VS} in the provided file list.")

    # Helper to process the specific frame
    def process_fk_frame(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        archive = np.load(path)
        data, lag_axis, offset_axis = archive['data'], archive['lag'], archive['offset']
        
        # Spatial Masking
        mask = np.abs(offset_axis) >= pos_offset
        if view_side == "right": mask &= (offset_axis >= 0)
        elif view_side == "left": mask &= (offset_axis <= 0)
            
        f_axis, k_axis_raw, fk_complex_raw = fk_transform(data[mask, :], lag_axis[1]-lag_axis[0], offset_axis[1]-offset_axis[0])
        sort_idx = np.argsort(-k_axis_raw)
        k_axis, fk_power = -k_axis_raw[sort_idx], np.abs(fk_complex_raw[sort_idx, :])
        
        pos_f_mask = f_axis >= 0
        f_axis, fk_power = f_axis[pos_f_mask], fk_power[:, pos_f_mask]
        
        if view_side == "right": k_mask = k_axis >= 0
        elif view_side == "left": k_mask = k_axis <= 0
        else: k_mask = np.ones_like(k_axis, dtype=bool)
            
        return k_axis[k_mask], f_axis, fk_power[k_mask, :].T

    k_axis, f_axis, fk_power = process_fk_frame(target_path)
    plot_k = k_axis * dist_scale

    # Compute global clip
    if pclip is not None:
        c0 = float(np.percentile(fk_power, pclip))
    else:
        c0 = float(clip if clip is not None else 1.0)
    c0 = c0 if c0 > 0 else 1.0

    # Layout Setup
    fig, ax = plt.subplots(figsize=figsize, layout="constrained", dpi=dpi)
    ax.set_xlabel(f"Wavenumber k (cycles/{unit})")
    ax.set_ylabel("Frequency f (Hz)")

    if klim is not None:
        if view_side == "left" and klim[0] >= 0:
            ax.set_xlim(-klim[1], -klim[0])
        else:
            ax.set_xlim(*klim)

    # Plot the 2D power spectrum
    mesh = ax.pcolormesh(
        plot_k, f_axis, fk_power/c0, 
        shading="gouraud", cmap=cmap, vmin=0, vmax=1.0
    )
    
    ax.set_ylim(0, np.max(f_axis))
    
    # Velocity Overlay Lines (v = f/k => f = v*k)
    if vmin is not None: 
        ax.plot(plot_k, vmin * np.abs(plot_k / dist_scale), color="cyan", linestyle="--", linewidth=1.8, label=f"vmin = {vmin} m/s")
    if vmax is not None: 
        ax.plot(plot_k, vmax * np.abs(plot_k / dist_scale), color="lime", linestyle="--", linewidth=1.8, label=f"vmax = {vmax} m/s")
    
    if vmin is not None or vmax is not None:
        ax.legend(loc="upper left", fontsize=10, framealpha=0.7, facecolor="black", edgecolor="white", labelcolor="white")

    # Center axis guide for double-sided viewing
    if view_side == "both" or (klim and klim[0] <= 0 <= klim[1]):
        ax.axvline(x=0.0, color="white", linestyle=":", linewidth=1.0, alpha=0.4)
        
    if show_cbar:
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label("Normalized Power")
    
    # Title Generation
    if title is None:
        title = f"F-K Spectrum (VS={target_vs_str} | {target_date} | View: {view_side.upper()})"
    
    ax.set_title(title)

    return fig, ax

def animate_fk(
    files: List[str],
    *,
    unit: str = "m",
    clip: float | None = None,
    pclip: float | None = 99.0,
    cmap: str = "inferno",
    view_side: Literal["both", "left", "right"] = "right",
    pos_offset: float = 0.0,
    klim: Tuple[float, float] | None = None,
    figsize: Tuple[float, float] = (8, 6),
    vmin: float | None = None,                 
    vmax: float | None = None,                 
    interval_ms: int = 200,
    save_vs: List[int] | None = None,
    save_dir: str = "./saved_figures",
    save_fmt: str = "png",
    save_dpi: int = 300,
) -> FuncAnimation:
    """
    Animates the normalized 2D frequency-wavenumber (f-k) power spectrum of 
    pre-processed Noise Cross-Correlation Function (NCF) gathers.

    This function dynamically recalculates the spatial grid and f-k transform for 
    every frame, safely handling variable receiver geometries (e.g., dropping nodes). 
    It features dynamic power normalization, directional wavefield isolation, and 
    optional phase-velocity reference overlays. Accepts explicit lists of pre-sorted 
    .npz files to allow for easy slicing and includes hooks for exporting static frames.

    :param files: List of file paths to the pre-processed NCF numpy archives.
    :type files: List[str]
    :param unit: Spatial distance unit used for the x-axis ("m" or "km"). Default is "m".
    :type unit: str
    :param clip: Absolute amplitude limit for power normalization. Used only if `pclip` is None.
    :type clip: Optional[float]
    :param pclip: Percentile for dynamic amplitude clipping (e.g., 99.0). Computes a global median percentile across early frames for stable animation scaling.
    :type pclip: Optional[float]
    :param cmap: Matplotlib colormap to use. Default is "inferno" (standard for power spectra).
    :type cmap: str
    :param view_side: Determines which side of the spatial array to process and which wavenumbers to display ("both", "left", or "right"). 
    :type view_side: Literal["both", "left", "right"]
    :param pos_offset: Spatial exclusion offset from the virtual source. Data within this distance is excluded before the f-k transform to prevent near-source spatial aliasing.
    :type pos_offset: float
    :param klim: Optional tuple (kmin, kmax) specifying the wavenumber (x-axis) limits. Automatically flips sign if `view_side` is "left".
    :type klim: Optional[Tuple[float, float]]
    :param figsize: Tuple specifying the figure dimensions. Default is (8, 6).
    :type figsize: Tuple[float, float]
    :param vmin: Optional minimum phase velocity (m/s). Plots a cyan dashed reference line.
    :type vmin: Optional[float]
    :param vmax: Optional maximum phase velocity (m/s). Plots a lime dashed reference line.
    :type vmax: Optional[float]
    :param interval_ms: Delay between animation frames in milliseconds. Default is 200.
    :type interval_ms: int
    :param save_vs: Optional list of Virtual Source (VS) numbers to save as static high-res images.
    :type save_vs: Optional[List[int]]
    :param save_dir: Directory where the static frames will be saved. Default is "./saved_figures".
    :type save_dir: str
    :param save_fmt: Image format for the saved frames (e.g., "png"). Default is "png".
    :type save_fmt: str
    :param save_dpi: Resolution for the saved frames. Default is 300.
    :type save_dpi: int
    :returns: The constructed Matplotlib FuncAnimation object ready for rendering or display.
    :rtype: FuncAnimation
    """
    if not files:
        raise ValueError("Provided file list is empty!")

    view_side, unit = view_side.lower().strip(), unit.lower().strip()
    dist_scale = 1000.0 if unit == "km" else 1.0

    parsed = []
    for p in files:
        date, vs, window, xmode = parse_ncf_stack_filename(p)
        parsed.append((p, date, vs, xmode))

    def process_fk_frame(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        archive = np.load(path)
        data, lag_axis, offset_axis = archive['data'], archive['lag'], archive['offset']
        
        # Spatial Masking
        mask = np.abs(offset_axis) >= pos_offset
        if view_side == "right": mask &= (offset_axis >= 0)
        elif view_side == "left": mask &= (offset_axis <= 0)
            
        f_axis, k_axis_raw, fk_complex_raw = fk_transform(data[mask, :], lag_axis[1]-lag_axis[0], offset_axis[1]-offset_axis[0])
        sort_idx = np.argsort(-k_axis_raw)
        k_axis, fk_power = -k_axis_raw[sort_idx], np.abs(fk_complex_raw[sort_idx, :])
        
        pos_f_mask = f_axis >= 0
        f_axis, fk_power = f_axis[pos_f_mask], fk_power[:, pos_f_mask]
        
        if view_side == "right": k_mask = k_axis >= 0
        elif view_side == "left": k_mask = k_axis <= 0
        else: k_mask = np.ones_like(k_axis, dtype=bool)
            
        return k_axis[k_mask], f_axis, fk_power[k_mask, :].T

    # Calculate global clip over a subset to save time
    if pclip is not None:
        c0 = float(np.median([np.percentile(process_fk_frame(p[0])[2], pclip) 
                              for p in tqdm(parsed[:50], desc="Scanning global f-k pclip")])) 
    else:
        c0 = float(clip if clip is not None else 1.0)
    c0 = c0 if c0 > 0 else 1.0

    # Layout Setup
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.set_xlabel(f"Wavenumber k (cycles/{unit})")
    ax.set_ylabel("Frequency f (Hz)")

    if klim is not None:
        ax.set_xlim(-klim[1], -klim[0]) if (view_side == "left" and klim[0] >= 0) else ax.set_xlim(*klim)

    # Initial Frame
    k0, f0, data0 = process_fk_frame(parsed[0][0])
    plot_k0 = k0 * dist_scale

    mesh = ax.pcolormesh(plot_k0, f0, data0/c0, shading="gouraud", cmap=cmap, vmin=0, vmax=1.0)
    ax.set_ylim(0, np.max(f0))
    
    # Velocity Overlay Lines
    if vmin is not None: ax.plot(plot_k0, vmin * np.abs(plot_k0 / dist_scale), color="cyan", linestyle="--", linewidth=1.8, label=f"vmin = {vmin} m/s")
    if vmax is not None: ax.plot(plot_k0, vmax * np.abs(plot_k0 / dist_scale), color="lime", linestyle="--", linewidth=1.8, label=f"vmax = {vmax} m/s")
    if vmin is not None or vmax is not None:
        ax.legend(loc="upper left", fontsize=10, framealpha=0.7, facecolor="black", edgecolor="white", labelcolor="white")

    vline = ax.axvline(x=0.0, color="white", linestyle=":", linewidth=1.0, alpha=0.4) if (view_side == "both" or (klim and klim[0] <= 0 <= klim[1])) else None
        
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label("Normalized Power")
    
    date0, vs0, xmode0 = parsed[0][1], parsed[0][2], parsed[0][3]
    title_text = ax.set_title(f"F-K Spectrum (VS={vs0} | {date0} | View: {view_side.upper()})")

    pbar_container = []
    processed_frames = set()
    saved_frames = set()
    total_frames = len(parsed)

    def update(frame_idx):
        nonlocal mesh # Declare nonlocal to overwrite the grid safely
        
        if not pbar_container:
            pbar_container.append(tqdm(total=total_frames, desc="Rendering f-k Video"))

        path, date, vs, xmode = parsed[frame_idx]
        
        # Recalculate axes for dynamic geometry
        k_axis, f_axis, fk_power = process_fk_frame(path)
        plot_k = k_axis * dist_scale
        
        mesh.remove()
        mesh = ax.pcolormesh(plot_k, f_axis, fk_power/c0, shading="gouraud", cmap=cmap, vmin=0, vmax=1.0)
        # -----------------------------------------
        
        title_text.set_text(f"F-K Spectrum (VS={vs} | {date} | View: {view_side.upper()})")
        
        # --- LOGIC: Save Specific Frames ---
        if save_vs is not None and int(vs) in save_vs:
            if frame_idx not in saved_frames:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"FK_Spectrum_VS_{vs}.{save_fmt}")
                fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
                saved_frames.add(frame_idx)
        # -----------------------------------

        if frame_idx not in processed_frames:
            pbar_container[0].update(1)
            processed_frames.add(frame_idx)
        if len(processed_frames) == total_frames:
            pbar_container[0].close()
            
        return (mesh, vline, title_text) if vline is not None else (mesh, title_text)

    ani = FuncAnimation(fig, update, frames=total_frames, interval=interval_ms, blit=False)
    plt.close(fig)
    return ani

# ==============================================================
# 3. Frequency-distance panels (sloth / phase velocity)
# ==============================================================
def _freq_distance_panel(
    positions: np.ndarray,
    freqs: np.ndarray,
    field: np.ndarray,
    *,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    cbar_label: str,
    title: str,
    invert_freq: bool,
    figsize: Tuple[float, float],
    save_path: Optional[PathLike],
    xlabel: str = "Distance along cable (m)",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Shared underlying renderer for 2D (distance x frequency) field maps.

    :param positions: 1D array of along-fiber positions in meters.
    :type positions: np.ndarray
    :param freqs: 1D array of frequencies in Hz.
    :type freqs: np.ndarray
    :param field: 2D array (positions, freqs) of the field to plot.
    :type field: np.ndarray
    :param cmap: Colormap name.
    :type cmap: str
    :param vmin: Minimum amplitude for the colormap.
    :type vmin: Optional[float]
    :param vmax: Maximum amplitude for the colormap.
    :type vmax: Optional[float]
    :param cbar_label: Label for the colorbar.
    :type cbar_label: str
    :param title: Plot title.
    :type title: str
    :param invert_freq: Whether to place low frequencies at the bottom axis.
    :type invert_freq: bool
    :param figsize: Tuple defining the figure dimensions (width, height) in inches.
    :type figsize: Tuple[float, float]
    :param save_path: Destination path to save the figure, or None to skip saving.
    :type save_path: Optional[PathLike]
    :param xlabel: Label for the spatial x-axis, by default "Distance along cable (m)".
    :type xlabel: str
    :returns: Figure and Axes objects.
    :rtype: Tuple[plt.Figure, plt.Axes]
    """
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    masked = np.ma.masked_invalid(field.T)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("0.85")

    mesh = ax.pcolormesh(positions, freqs, masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, shading="gouraud")

    if invert_freq:
        ax.set_ylim(freqs.max(), freqs.min())

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)

    _save(fig, save_path)
    return fig, ax


def plot_phase_velocity_section(
    positions: np.ndarray,
    freqs: np.ndarray,
    vel: np.ndarray,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "turbo",
    invert_freq: bool = True,
    title: str = r"I-FDG phase velocity $\hat{V}(x, f)$",
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[PathLike] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the I-FDG phase-velocity frequency-distance panel (Davis et al. 2026, Fig. 7b).

    :param positions: 1D array of along-fiber positions in meters.
    :type positions: np.ndarray
    :param freqs: 1D array of frequencies in Hz.
    :type freqs: np.ndarray
    :param vel: 2D array (positions, freqs) of phase velocities (m/s).
    :type vel: np.ndarray
    :param vmin: Minimum velocity for color scale. Auto-computes 2nd percentile if None.
    :type vmin: Optional[float]
    :param vmax: Maximum velocity for color scale. Auto-computes 98th percentile if None.
    :type vmax: Optional[float]
    :param cmap: Colormap name, by default "turbo".
    :type cmap: str
    :param invert_freq: Places low frequencies at the bottom axis, by default True.
    :type invert_freq: bool
    :param title: Plot title.
    :type title: str
    :param figsize: Figure size in inches, by default (12, 5).
    :type figsize: Tuple[float, float]
    :param save_path: Destination path to export figure.
    :type save_path: Optional[PathLike]
    :returns: Figure and Axes objects.
    :rtype: Tuple[plt.Figure, plt.Axes]
    """
    if vel.shape != (positions.size, freqs.size):
        raise ValueError(f"vel shape {vel.shape} != (nx={positions.size}, nf={freqs.size})")

    vmin_val = float(np.nanpercentile(vel, 2)) if vmin is None else vmin
    vmax_val = float(np.nanpercentile(vel, 98)) if vmax is None else vmax

    return _freq_distance_panel(
        positions, freqs, vel, cmap=cmap, vmin=vmin_val, vmax=vmax_val,
        cbar_label=r"$V_{\mathrm{phase}}$ (m/s)", title=title,
        invert_freq=invert_freq, figsize=figsize, save_path=save_path,
    )


def _lateral_median(field: np.ndarray, size: int) -> np.ndarray:
    """
    NaN-aware running median across positions (axis 0). DISPLAY smoothing only:
    it imposes lateral smoothness and does not add real spatial resolution
    (which stays capped by source spacing / aperture).

    :param field: 2D array of data to smooth.
    :type field: np.ndarray
    :param size: Window size for the median filter.
    :type size: int
    :returns: The laterally smoothed data.
    :rtype: np.ndarray
    """
    if size <= 1:
        return field
    if size % 2 == 0:
        size += 1
    h = size // 2
    padded = np.pad(field, ((h, h), (0, 0)), mode="edge")
    out = np.empty_like(field)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        for i in range(field.shape[0]):
            out[i] = np.nanmedian(padded[i:i + size], axis=0)
    return out


def plot_sloth_section(
    positions: np.ndarray,
    freqs: np.ndarray,
    s2: np.ndarray,
    *,
    show_velo: bool = False,
    unit: Literal["m", "km"] = "m",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "turbo",
    lateral_median: int = 0,
    invert_freq: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[PathLike] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the stacked I-FDG sloth -- or, with ``show_velo=True``, the phase
    velocity derived from it -- as a (distance x frequency) panel.

    Non-physical samples (Re s^2 <= 0) are masked to NaN and drawn in grey, so
    the same blank pixels appear in both the sloth and velocity views.

    :param positions: 1D along-fiber positions in METERS (converted internally per ``unit``).
    :type positions: np.ndarray
    :param freqs: 1D frequencies in Hz.
    :type freqs: np.ndarray
    :param s2: 2D complex or real sloth (s^2/m^2), shape (positions, freqs).
    :type s2: np.ndarray
    :param show_velo: If True, convert sloth to phase velocity ``V = 1/sqrt(Re s^2)`` and plot that instead of sloth. Default False.
    :type show_velo: bool
    :param unit: Spatial unit for the distance axis AND the plotted quantity. Default "m".
    :type unit: Literal["m", "km"]
    :param vmin: Minimum color limit in the chosen unit.
    :type vmin: Optional[float]
    :param vmax: Maximum color limit in the chosen unit.
    :type vmax: Optional[float]
    :param cmap: Colormap, by default "turbo".
    :type cmap: str
    :param lateral_median: If > 1, apply a NaN-aware running median of this width across channels before plotting. Default 0 (off).
    :type lateral_median: int
    :param invert_freq: Place low frequency at the bottom (pseudo-depth), by default True.
    :type invert_freq: bool
    :param title: Plot title; auto-set for sloth vs velocity if None.
    :type title: Optional[str]
    :param figsize: Figure size.
    :type figsize: Tuple[float, float]
    :param save_path: Export path.
    :type save_path: Optional[PathLike]
    :returns: Figure and Axes objects.
    :rtype: Tuple[plt.Figure, plt.Axes]
    """
    unit = unit.lower().strip()  # type: ignore[assignment]
    if unit not in ("m", "km"):
        raise ValueError("unit must be 'm' or 'km'")
    dist_scale = 1000.0 if unit == "km" else 1.0

    # Mask non-physical sloth (Re s^2 <= 0) before any conversion.
    s2r = np.asarray(np.real(s2), dtype=float)
    with np.errstate(invalid="ignore"):
        s2r = np.where(s2r > 0.0, s2r, np.nan)

    pos_plot = np.asarray(positions, dtype=float) / dist_scale
    xlabel = f"Distance along cable ({unit})"

    if show_velo:
        # phase velocity = 1/sqrt(sloth); m/s -> divide by dist_scale for km/s
        with np.errstate(invalid="ignore", divide="ignore"):
            field = (1.0 / np.sqrt(s2r)) / dist_scale
        cbar_label = rf"$V_{{\mathrm{{phase}}}}$ ({unit}/s)"
        default_title = r"I-FDG phase velocity $\hat{V}(x, f)$"
        lo_pct = 2.0
    else:
        # sloth s^2/m^2 -> s^2/km^2 multiplies by dist_scale^2 (= 1/V^2 units)
        field = s2r * (dist_scale ** 2)
        cbar_label = rf"Re $\hat{{s}}^2$ (s$^2$/{unit}$^2$)"
        default_title = r"I-FDG sloth $\hat{s}^2(x, f)$"
        lo_pct = 0.0

    if lateral_median and lateral_median > 1:
        field = _lateral_median(field, int(lateral_median))

    vmin_val = (0.0 if lo_pct == 0.0 else float(np.nanpercentile(field, lo_pct))) if vmin is None else vmin
    vmax_val = float(np.nanpercentile(field, 98)) if vmax is None else vmax

    return _freq_distance_panel(
        pos_plot, freqs, field, cmap=cmap, vmin=vmin_val, vmax=vmax_val,
        cbar_label=cbar_label, title=title or default_title,
        invert_freq=invert_freq, figsize=figsize, save_path=save_path,
        xlabel=xlabel,
    )


# ==============================================================
# 3. Dispersion cross-check
# ==============================================================
def plot_dispersion_compare(
    freqs: np.ndarray,
    c_grad: np.ndarray,
    *,
    position_m: Optional[float] = None,
    ref_freq: Optional[np.ndarray] = None,
    ref_vel: Optional[np.ndarray] = None,
    c_band: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 5),
    save_path: Optional[PathLike] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Overlay a local I-FDG dispersion curve against an independent phase-shift pick.

    :param freqs: 1D frequency axis (Hz) of the gradiometry estimate.
    :type freqs: np.ndarray
    :param c_grad: 1D local I-FDG phase velocity (m/s).
    :type c_grad: np.ndarray
    :param position_m: Position label for plot header.
    :type position_m: Optional[float]
    :param ref_freq: Reference frequency axis from independent picks.
    :type ref_freq: Optional[np.ndarray]
    :param ref_vel: Reference picked phase velocities (m/s).
    :type ref_vel: Optional[np.ndarray]
    :param c_band: Lower and upper velocity bound arrays for lateral spread shading.
    :type c_band: Optional[Tuple[np.ndarray, np.ndarray]]
    :param title: Custom plot title.
    :type title: Optional[str]
    :param figsize: Figure dimensions, by default (6, 5).
    :type figsize: Tuple[float, float]
    :param save_path: Destination path to save image file.
    :type save_path: Optional[PathLike]
    :returns: Figure and Axes objects.
    :rtype: Tuple[plt.Figure, plt.Axes]
    """
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    if c_band is not None:
        ax.fill_between(freqs, c_band[0], c_band[1], color="C0", alpha=0.2, label="I-FDG lateral spread")

    ax.plot(freqs, c_grad, "o-", color="C0", lw=2, ms=4, label="I-FDG (gradiometry)")

    if ref_freq is not None and ref_vel is not None:
        ax.plot(ref_freq, ref_vel, "s--", color="C3", lw=1.8, ms=4, label="das_ani dispersion pick")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")

    loc_str = f" at x = {position_m:.0f} m" if position_m is not None else ""
    ax.set_title(title or f"Dispersion cross-check{loc_str}")

    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="best")

    _save(fig, save_path)
    return fig, ax