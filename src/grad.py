"""
:module: src/grad.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Config-driven I-FDG workflow driver -- das_grad's analogue of
          das_ani's src/cc.py.

Workflow (Davis et al. 2026, Fig. 6; Gradiometry_Theory Sec. 10):

    for every VSG (from das_ani):
        1. torus masking            (src.mask.torus_mask)
        2. channel taper            (edge-artifact suppression)
        3. temporal FFT, band-limit (src.gradiometry.temporal_fft)
        4. pseudospectral Laplacian (src.laplacian.{fiber_1d|grid_2d})
        5. per-VSG sloth, eq. 9     (src.gradiometry.fdg_sloth)
    6. stack over virtual sources, eq. 10   (src.gradiometry.ifdg_stack)
    7. phase velocity V = Re(s2)^(-1/2)     (src.gradiometry.sloth_to_velocity)
    8. quality mask + median filter         (src.post)
    9. save products .npz                   (src.post.save_products)

Run:
    python -m src.grad --config configs/urban_grad.yaml --verbose
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from src.gradiometry import fdg_sloth, ifdg_stack, sloth_to_velocity, temporal_fft
from src.laplacian import laplacian_fiber, laplacian_grid
from src.mask import apply_mask, channel_taper_weights, torus_mask
from src.post import median_filter_1d, quality_mask, save_products
from src.utils import get_cfg, get_tqdm, load_config, timeit
from src.vsg import VSG, common_positions, discover_vsgs, load_vsg

logger = logging.getLogger(__name__)

LAPLACIAN_MODES = ("fiber_1d", "grid_2d")


def process_vsg(vsg: VSG, cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Steps 1-5 for one VSG: mask -> taper -> FFT -> Laplacian -> sloth.

    :param vsg: Loaded :class:`src.vsg.VSG`.
    :param cfg: Full das_grad config mapping.
    :return: (s2_i, freqs): complex (nch, nfreq) per-VSG sloth and the
             frequency axis (Hz).
    """
    # ---- 1. Torus masking ----
    w = torus_mask(
        vsg.lag, vsg.offset,
        v_inner=float(get_cfg(cfg, ["mask", "v_inner"], required=True)),
        v_outer=float(get_cfg(cfg, ["mask", "v_outer"], required=True)),
        t_inner=float(get_cfg(cfg, ["mask", "t_inner"], 5.0)),
        t_outer=float(get_cfg(cfg, ["mask", "t_outer"], -5.0)),
        taper_sec=float(get_cfg(cfg, ["mask", "taper_sec"], 1.0)),
        causal_only=bool(get_cfg(cfg, ["mask", "causal_only"], True)),
    )
    d = apply_mask(vsg.data, w)

    # ---- 2. Channel taper (limit spatial-FFT wraparound) ----
    alpha = float(get_cfg(cfg, ["laplacian", "channel_taper_alpha"], 0.2))
    d *= channel_taper_weights(vsg.nch, alpha)[:, None]

    # ---- 3. Temporal FFT, restricted to the usable band ----
    V, freqs = temporal_fft(
        d, vsg.fs,
        f_min=float(get_cfg(cfg, ["band", "f_min"], required=True)),
        f_max=float(get_cfg(cfg, ["band", "f_max"], required=True)),
        axis=-1,
    )

    # ---- 4. Pseudospectral Laplacian ----
    mode = str(get_cfg(cfg, ["laplacian", "mode"], "fiber_1d")).lower()
    if mode == "fiber_1d":
        lap_V = laplacian_fiber(
            V, vsg.dx,
            offset=vsg.offset,
            include_curvature=bool(get_cfg(cfg, ["laplacian", "include_curvature"], True)),
        )
    elif mode == "grid_2d":
        # 2-D geometry: data must be reshaped (ny, nx, nfreq) by the caller's
        # geometry block. v0 supports square channel layouts declared via
        # laplacian.grid_ny; rows are assumed row-major along the cable.
        ny = int(get_cfg(cfg, ["laplacian", "grid_ny"], required=True))
        nx = vsg.nch // ny
        if ny * nx != vsg.nch:
            raise ValueError(f"grid_2d: grid_ny={ny} does not tile nch={vsg.nch}.")
        dy = float(get_cfg(cfg, ["laplacian", "grid_dy"], vsg.dx))
        lap_V = laplacian_grid(
            V.reshape(ny, nx, -1), vsg.dx, dy, axes=(0, 1)
        ).reshape(vsg.nch, -1)
    else:
        raise ValueError(f"laplacian.mode must be one of {LAPLACIAN_MODES}; got {mode!r}")

    # ---- 5. Per-VSG sloth (Davis et al. 2026, eq. 9) ----
    return fdg_sloth(V, lap_V, freqs, freq_axis=-1), freqs


@timeit
def run_pipeline(cfg: Mapping[str, Any]) -> Path:
    """
    Full I-FDG run over all discovered VSGs; returns the product .npz path.
    """
    fs = float(get_cfg(cfg, ["data", "fs_proc"], required=True))
    dx = float(get_cfg(cfg, ["data", "dx"], required=True))
    first_chan = int(get_cfg(cfg, ["data", "first_chan"], 0))

    vsg_root = get_cfg(cfg, ["paths", "vsg_root"], required=True)
    pattern = str(get_cfg(cfg, ["paths", "vsg_pattern"], "*_cc_*.npy"))
    out_root = Path(str(get_cfg(cfg, ["paths", "output_root"], "./data/grad"))).expanduser()
    tag = str(get_cfg(cfg, ["output", "tag"], "grad"))

    files = discover_vsgs(vsg_root, pattern)
    tqdm = get_tqdm()

    per_vsg: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    vsgs_meta: list[str] = []
    freqs: Optional[np.ndarray] = None
    first: Optional[VSG] = None

    for f in tqdm(files, desc="I-FDG over VSGs"):
        vsg = load_vsg(f, fs=fs, dx=dx)
        if first is None:
            first = vsg
        elif vsg.nch != first.nch or vsg.nlag != first.nlag:
            logger.warning("Skipping %s: shape %s differs from %s.",
                           vsg.name, vsg.data.shape, first.data.shape)
            continue
        s2_i, fr = process_vsg(vsg, cfg)
        if freqs is None:
            freqs = fr
        per_vsg.append(s2_i)
        offsets.append(vsg.offset)
        vsgs_meta.append(vsg.name)

    if not per_vsg or first is None or freqs is None:
        raise RuntimeError("No usable VSGs were processed.")
    logger.info("Stacking %d per-VSG sloth estimates (I-FDG eq. 10).", len(per_vsg))

    # ---- 6. Stack over virtual sources ----
    # Near-source exclusion: within ~a wavelength of its own VS every gather
    # is near-field and biased; excluding those channels per VSG (they stay
    # covered by the other sources) improves the stacked sloth by an order
    # of magnitude on synthetics. Recommended r_exclude_m ~ one wavelength
    # at the center of the band.
    r_excl = float(get_cfg(cfg, ["stack", "r_exclude_m"], 0.0))
    weights = None
    if r_excl > 0.0:
        weights = [(np.abs(off) >= r_excl).astype(np.float64) for off in offsets]
        covered = np.sum(weights, axis=0)
        if np.any(covered == 0):
            logger.warning(
                "stack.r_exclude_m=%.0f m leaves %d channels with no "
                "contributing VSG (they will be NaN).",
                r_excl, int(np.sum(covered == 0)),
            )
    s2 = ifdg_stack(per_vsg, weights=weights)

    # ---- 7-8. Velocity + quality + median filter ----
    vel = sloth_to_velocity(
        s2,
        v_min=get_cfg(cfg, ["post", "v_min"], None),
        v_max=get_cfg(cfg, ["post", "v_max"], None),
    )
    valid = quality_mask(
        first.nch, offsets,
        edge_frac=float(get_cfg(cfg, ["post", "edge_frac"], 0.1)),
        r_min_m=float(get_cfg(cfg, ["post", "r_min_m"], 0.0)),
    )
    vel[~valid, :] = np.nan
    med = int(get_cfg(cfg, ["post", "median_size"], 5))
    if med > 1:
        vel = median_filter_1d(vel, med, axis=0)

    # ---- 9. Save ----
    positions = common_positions([first], dx, first_chan=first_chan)
    out_path = out_root / f"grad_vsch_{tag}.npz"
    return save_products(
        out_path,
        s2=s2, vel=vel, freqs=freqs, positions=positions, valid=valid,
        meta={
            "n_vsg": len(per_vsg),
            "vsg_files": vsgs_meta,
            "fs_proc": fs, "dx": dx, "first_chan": first_chan,
            "config": {k: v for k, v in cfg.items()},
        },
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interferometric frequency-domain gradiometry (I-FDG) on das_ani VSGs"
    )
    p.add_argument("--config", type=str, required=True,
                   help="Path to config file (.yaml/.yml/.json)")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args(args=argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    run_pipeline(load_config(args.config))

# Example:
# python -m src.grad --config configs/urban_grad.yaml --verbose
