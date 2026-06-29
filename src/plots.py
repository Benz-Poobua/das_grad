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

Style mirrors das_ani/src/plots.py and src/inv.py (numpydoc docstrings,
pcolormesh with 'gouraud' shading, turbo/seismic colormaps, 300 dpi saves).
The module is import-light: numpy + matplotlib only.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Sequence, Tuple, Union, List, Literal 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams

from src.utils import parse_ncf_stack_filename, fk_transform


logger = logging.getLogger(__name__)
PathLike = Union[str, Path]

_RC = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "font.size": 11,
}

# ===========================================================================
# Plotting Configuration
# ===========================================================================
params = {
    'savefig.dpi': 300,
    'axes.labelsize': 14,
    'axes.titlesize': 18,
    'font.size': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'figure.figsize': [12, 6],
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
}
rcParams.update(params)


def _save(fig: plt.Figure, save_path: Optional[PathLike]) -> None:
    if save_path:
        p = Path(save_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        logger.info("Saved figure -> %s", p)
        print(f"Figure saved to: {p}")


# ==============================================================
# 1. Virtual source gather (QC)
# ==============================================================
def plot_vsg(
    files: List[str],
    VS: Union[str, int],
    *,
    unit: str = "m",
    clip: float | None = 0.05,
    pclip: float | None = None,
    cmap: str = "seismic",
    range_m: float = 4000.0,
    clip_lim: bool = True,
    view_side: Literal["both", "left", "right"] = "both",
    pos_offset: float = 0.0,
    figsize: Tuple[float, float] | None = None,
    title: str | None = None,
    show_cbar: bool = True,
    dpi: int = 120,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a static, spatially and temporally pre-processed Noise Cross-Correlation 
    Function (NCF) gather for a single Virtual Source (VS).

    This function automatically aligns with the zero-offset trace, handling cases where 
    node geometries vary across files.

    :param files: List of file paths to the pre-processed NCF numpy archives (.npz).
    :param VS: Virtual Source number to plot (e.g., 5 or "005"). The function scans the list for it.
    :param unit: Spatial distance unit for the x-axis ("m" or "km"). Default is "m".
    :param clip: Absolute amplitude limit for colorbar scaling. Used only if `pclip` is None.
    :param pclip: Percentile for dynamic amplitude clipping (e.g., 99.0).
    :param cmap: Matplotlib colormap to use. Default is "seismic".
    :param range_m: Maximum spatial distance (in meters or km based on `unit`) to display.
    :param clip_lim: If True, strictly limits the x-axis bounds based on `range_m`, `view_side`, 
                     and `pos_offset`.
    :param view_side: Determines which side of the virtual source gather to display 
                      ("both", "left", or "right"). Default is "both".
    :param pos_offset: Spatial exclusion offset from the virtual source to clip out near-source noise.
    :param figsize: Optional tuple defining figure dimensions (width, height) in inches.
    :param title: Custom title for the plot. If None, auto-generates one.
    :param show_cbar: Toggle visibility of the correlation amplitude colorbar.
    :param dpi: Resolution of the output plot.
    :returns: A tuple containing the (Figure, Axes) objects.
    """
    if not files:
        raise ValueError("Provided file list is empty!")

    view_side, unit = view_side.lower().strip(), unit.lower().strip()
    dist_scale = 1000.0 if unit == "km" else 1.0
    plot_range, plot_offset = range_m / dist_scale, pos_offset / dist_scale

    # Find the specific VS file from the list
    target_path = None
    target_date, target_window, target_vs_str, target_xmode = "", "", "", ""

    for p in files:
        date, vs_str, window, xmode = parse_ncf_stack_filename(p)
        if int(vs_str) == int(VS):
            target_path = p
            target_date = date
            target_window = window
            target_vs_str = vs_str
            target_xmode = xmode
            break

    if target_path is None:
        raise FileNotFoundError(f"Could not find a file matching VS={VS} in the provided file list.")

    # Load Data from .npz
    archive = np.load(target_path)
    current_offset = archive['offset'] / dist_scale
    lag_axis = archive['lag']
    data = archive['data'].T

    # Compute clipping limits
    if pclip is not None:
        c0 = float(np.percentile(np.abs(data), pclip))
    else:
        c0 = float(clip if clip is not None else 1.0)

    # Establish plotting limits based on view_side and pos_offset
    if view_side == "both": 
        left_bound, right_bound = -plot_range, plot_range
    elif view_side == "right": 
        left_bound, right_bound = plot_offset, plot_range
    else: 
        left_bound, right_bound = -plot_range, -plot_offset

    # Setup Plot
    if figsize is None:
        figsize = (8, 6) if clip_lim else (10, 6)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained", dpi=dpi)
    ax.invert_yaxis()
    
    ax.set_xlabel(f"Offset from Virtual Source ({unit})")
    ax.set_ylabel("Lag time (s)")

    if clip_lim: 
        ax.set_xlim(left_bound, right_bound)

    mesh = ax.pcolormesh(
        current_offset, lag_axis, data, 
        shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0
    )
    
    ax.axvline(x=0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)

    if show_cbar:
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label("Correlation amplitude")

    # Title generation
    if title is None:
        title = f"NCF Gather (VS={target_vs_str} | {target_date} | {target_xmode})"
    
    ax.set_title(title)
    
    return fig, ax

def animate_vsg(
    files: List[str],
    *,
    unit: str = "m",
    clip: float | None = 0.05,
    pclip: float | None = None,
    cmap: str = "seismic",
    range_m: float = 4000.0,
    clip_lim: bool = True,
    view_side: Literal["both", "left", "right"] = "both",
    pos_offset: float = 0.0,
    interval_ms: int = 200,
    save_vs: List[int] | None = None,
    save_dir: str = "./saved_figures",
    save_fmt: str = "png",
    save_dpi: int = 300,
) -> FuncAnimation:
    """
    Animates spatially and temporally pre-processed Noise Cross-Correlation Function (NCF) 
    gathers using Matplotlib's pcolormesh.

    This function seamlessly handles variable receiver geometries (e.g., urban datasets where 
    nodes drop offline) by dynamically redrawing the spatial grid for each frame. It features 
    dynamic percentile clipping, directional viewing, near-field offset masking, and selective 
    frame saving. Accepts an explicit list of pre-sorted .npz files for easy slicing.

    :param files: List of file paths to the pre-processed NCF numpy archives.
    :param unit: Spatial distance unit for the x-axis ("m" or "km"). Default is "m".
    :param clip: Absolute amplitude limit for colorbar scaling. Used only if `pclip` is None.
    :param pclip: Percentile for dynamic amplitude clipping (e.g., 99.0). Computes a global 
                  median percentile across early frames for stable animation scaling.
    :param cmap: Matplotlib colormap to use. Default is "seismic".
    :param range_m: Maximum spatial distance (in meters or km based on `unit`) to display.
    :param clip_lim: If True, strictly limits the x-axis bounds based on `range_m`, `view_side`, 
                     and `pos_offset`.
    :param view_side: Determines which side of the virtual source gather to display 
                      ("both", "left", or "right"). Default is "both".
    :param pos_offset: Spatial exclusion offset from the virtual source to clip out near-source noise.
    :param interval_ms: Delay between animation frames in milliseconds. Default is 200.
    :param save_vs: Optional list of Virtual Source (VS) numbers to save as static high-res images.
    :param save_dir: Directory where the static frames will be saved. Default is "./saved_figures".
    :param save_fmt: Image format for the saved frames (e.g., "png"). Default is "png".
    :param save_dpi: Resolution for the saved frames. Default is 300.
    :returns: The constructed Matplotlib FuncAnimation object ready for rendering or display.
    """
    if not files:
        raise ValueError("Provided file list is empty!")

    view_side, unit = view_side.lower().strip(), unit.lower().strip()
    dist_scale = 1000.0 if unit == "km" else 1.0
    plot_range, plot_offset = range_m / dist_scale, pos_offset / dist_scale

    parsed = []
    for p in files:
        date, vs, window, xmode = parse_ncf_stack_filename(p)
        parsed.append((p, date, vs, xmode))

    if pclip is not None:
        c0 = float(np.median([np.percentile(np.abs(np.load(p[0])['data']), pclip) 
                              for p in tqdm(parsed[:50], desc="Scanning global pclip")])) 
    else:
        c0 = float(clip if clip is not None else 1.0)

    if view_side == "both": left_bound, right_bound = -plot_range, plot_range
    elif view_side == "right": left_bound, right_bound = plot_offset, plot_range
    else: left_bound, right_bound = -plot_range, -plot_offset

    # Figure Layout using Constrained Layout
    fig, ax = plt.subplots(figsize=(8, 6) if clip_lim else (10, 6), layout="constrained")
    ax.invert_yaxis()
    ax.set_xlabel(f"Offset from Virtual Source ({unit})")
    ax.set_ylabel("Lag time (s)")
    if clip_lim: 
        ax.set_xlim(left_bound, right_bound)

    # Load Initial Frame
    archive0 = np.load(parsed[0][0])
    current_offset0 = archive0['offset'] / dist_scale
    lag_axis0 = archive0['lag']
    data0 = archive0['data'].T
    
    mesh = ax.pcolormesh(current_offset0, lag_axis0, data0, shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0)
    vline = ax.axvline(x=0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label("Correlation amplitude")
    
    date0, vs0, xmode0 = parsed[0][1], parsed[0][2], parsed[0][3]
    title_text = ax.set_title(f"NCF Gather (VS={vs0} | {date0} | {xmode0})")

    # Animation tracking variables
    pbar_container = []
    processed_frames = set()
    saved_frames = set()
    total_frames = len(parsed)

    def update(frame_idx):
        nonlocal mesh  # Declare nonlocal so we can overwrite the mesh
        
        if not pbar_container:
            pbar_container.append(tqdm(total=total_frames, desc="Rendering Video"))

        path, date, vs, xmode = parsed[frame_idx]
        archive = np.load(path)
        dA = archive['data'].T
        
        # Load the dynamic offset axis for this specific frame
        current_offset = archive['offset'] / dist_scale
        lag_axis = archive['lag']
        

        mesh.remove()
        mesh = ax.pcolormesh(current_offset, lag_axis, dA, shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0)
        # ------------------------------------------
        
        title_text.set_text(f"NCF Gather (VS={vs} | {date} | {xmode})")
        
        # --- LOGIC: Save Specific Frames ---
        if save_vs is not None and int(vs) in save_vs:
            if frame_idx not in saved_frames:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"NCF_Gather_VS_{vs}.{save_fmt}")
                fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
                saved_frames.add(frame_idx)
        # -----------------------------------
        
        if frame_idx not in processed_frames:
            pbar_container[0].update(1)
            processed_frames.add(frame_idx)
        if len(processed_frames) == total_frames:
            pbar_container[0].close()

        return mesh, vline, title_text

    ani = FuncAnimation(fig, update, frames=total_frames, interval=interval_ms, blit=False)
    plt.close(fig)
    return ani

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
    :param VS: Virtual Source number to plot (e.g., 5 or "005").
    :param unit: Spatial distance unit used for the x-axis ("m" or "km"). Default is "m".
    :param clip: Absolute amplitude limit for power normalization. Used only if `pclip` is None.
    :param pclip: Percentile for dynamic amplitude clipping (e.g., 99.0).
    :param cmap: Matplotlib colormap to use. Default is "inferno".
    :param view_side: Determines which side of the spatial array to process and display ("both", "left", or "right"). 
    :param pos_offset: Spatial exclusion offset to clip out near-source auto-correlation artifacts.
    :param klim: Optional tuple (kmin, kmax) specifying the wavenumber (x-axis) limits. 
    :param figsize: Tuple specifying the figure dimensions.
    :param vmin: Optional minimum phase velocity (m/s). Plots a cyan dashed reference line.
    :param vmax: Optional maximum phase velocity (m/s). Plots a lime dashed reference line.
    :param title: Custom title for the plot. If None, auto-generates one.
    :param show_cbar: Toggle visibility of the power amplitude colorbar.
    :param dpi: Resolution of the output plot.
    :returns: A tuple containing the (Figure, Axes) objects.
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
    :param unit: Spatial distance unit used for the x-axis ("m" or "km"). Default is "m".
    :param clip: Absolute amplitude limit for power normalization. Used only if `pclip` is None.
    :param pclip: Percentile for dynamic amplitude clipping (e.g., 99.0). Computes a global 
                  median percentile across early frames for stable animation scaling.
    :param cmap: Matplotlib colormap to use. Default is "inferno" (standard for power spectra).
    :param view_side: Determines which side of the spatial array to process and which wavenumbers 
                      to display ("both", "left", or "right"). 
    :param pos_offset: Spatial exclusion offset from the virtual source. Data within this distance 
                       is excluded before the f-k transform to prevent near-source spatial aliasing.
    :param klim: Optional tuple (kmin, kmax) specifying the wavenumber (x-axis) limits. 
                 Automatically flips sign if `view_side` is "left".
    :param figsize: Tuple specifying the figure dimensions. Default is (8, 6).
    :param vmin: Optional minimum phase velocity (m/s). Plots a cyan dashed reference line.
    :param vmax: Optional maximum phase velocity (m/s). Plots a lime dashed reference line.
    :param interval_ms: Delay between animation frames in milliseconds. Default is 200.
    :param save_vs: Optional list of Virtual Source (VS) numbers to save as static high-res images.
    :param save_dir: Directory where the static frames will be saved. Default is "./saved_figures".
    :param save_fmt: Image format for the saved frames (e.g., "png"). Default is "png".
    :param save_dpi: Resolution for the saved frames. Default is 300.
    :returns: The constructed Matplotlib FuncAnimation object ready for rendering or display.
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
# 2. Frequency-distance panels (sloth / phase velocity)
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
) -> None:
    """Shared renderer for (distance x frequency) maps. ``field`` is (nx, nf)."""
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=figsize)
        masked = np.ma.masked_invalid(field.T)            # (nf, nx)
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad("0.85")                           # grey = no estimate
        mesh = ax.pcolormesh(positions, freqs, masked, cmap=cmap_obj,
                             vmin=vmin, vmax=vmax, shading="gouraud")
        # low frequency = deeper: put it at the bottom of the axis
        if invert_freq:
            ax.set_ylim(freqs.max(), freqs.min())
        ax.set_xlabel("Distance along cable (m)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(cbar_label)
        _save(fig, save_path)
        plt.show()


def plot_phase_velocity_section(
    positions: np.ndarray,
    freqs: np.ndarray,
    vel: np.ndarray,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "turbo",
    invert_freq: bool = True,
    title: str = "I-FDG phase velocity  $\\hat{V}(x, f)$",
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[PathLike] = None,
) -> None:
    """
    Plot the I-FDG phase-velocity frequency-distance panel (Davis et al. 2026,
    Fig. 7b): horizontal axis = position along the fiber, vertical axis =
    frequency, colour = local surface-wave phase velocity. Lower frequencies
    sense deeper structure, so the axis is inverted by default (low f at the
    bottom) to read like a pseudo-depth section.

    :param positions: (nx,) along-fiber positions (m).
    :param freqs: (nf,) frequency axis (Hz).
    :param vel: (nx, nf) phase velocity (m/s); NaN where unconstrained.
    :param vmin/vmax: Colour limits (m/s). Default: 2nd/98th percentiles.
    :param invert_freq: Put low frequency at the bottom. Default True.
    """
    if vel.shape != (positions.size, freqs.size):
        raise ValueError(f"vel shape {vel.shape} != (nx={positions.size}, nf={freqs.size})")
    if vmin is None:
        vmin = float(np.nanpercentile(vel, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(vel, 98))
    _freq_distance_panel(
        positions, freqs, vel, cmap=cmap, vmin=vmin, vmax=vmax,
        cbar_label="$V_{\\mathrm{phase}}$ (m/s)", title=title,
        invert_freq=invert_freq, figsize=figsize, save_path=save_path,
    )


def plot_sloth_section(
    positions: np.ndarray,
    freqs: np.ndarray,
    s2: np.ndarray,
    *,
    cmap: str = "turbo",
    invert_freq: bool = True,
    title: str = "I-FDG sloth  $\\hat{s}^2(x, f)$",
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[PathLike] = None,
) -> None:
    """
    Plot the (real part of the) stacked sloth estimate as a frequency-distance
    panel. ``s2`` may be complex (the imaginary part is a transport-residual
    quality diagnostic); only ``Re s2`` is mapped.

    :param s2: (nx, nf) complex or real sloth (s^2/m^2).
    """
    s2r = np.real(np.asarray(s2))
    with np.errstate(invalid="ignore"):
        s2r = np.where(s2r > 0, s2r, np.nan)
    vmax = float(np.nanpercentile(s2r, 98))
    _freq_distance_panel(
        positions, freqs, s2r, cmap=cmap, vmin=0.0, vmax=vmax,
        cbar_label="Re $\\hat{s}^2$ (s$^2$/m$^2$)", title=title,
        invert_freq=invert_freq, figsize=figsize, save_path=save_path,
    )


# ==============================================================
# 3. Dispersion cross-check against das_ani picks
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
) -> None:
    """
    Overlay a local I-FDG dispersion curve against an independent das_ani
    phase-shift (MASW) pick at the same position -- the key validation that
    gradiometry recovers the same surface-wave phase velocity as the
    aperture-averaged dispersion panel (Davis et al. 2026, Fig. 7c).

    :param freqs: (nf,) frequency axis (Hz) of the gradiometry estimate.
    :param c_grad: (nf,) local I-FDG phase velocity at the chosen position (m/s).
    :param position_m: Position label for the title (m).
    :param ref_freq: Optional (m,) frequency axis of the das_ani pick (Hz).
    :param ref_vel: Optional (m,) das_ani picked phase velocity (m/s).
    :param c_band: Optional (lo, hi) arrays for a shaded lateral spread of the
                   gradiometry estimate around ``c_grad``.
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=figsize)
        if c_band is not None:
            ax.fill_between(freqs, c_band[0], c_band[1], color="C0", alpha=0.2,
                            label="I-FDG lateral spread")
        ax.plot(freqs, c_grad, "o-", color="C0", lw=2, ms=4, label="I-FDG (gradiometry)")
        if ref_freq is not None and ref_vel is not None:
            ax.plot(ref_freq, ref_vel, "s--", color="C3", lw=1.8, ms=4,
                    label="das_ani dispersion pick")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Phase velocity (m/s)")
        loc = f" at x = {position_m:.0f} m" if position_m is not None else ""
        ax.set_title(title or f"Dispersion cross-check{loc}")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=10)
        _save(fig, save_path)
        plt.show()
