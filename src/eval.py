"""
:module: src/eval.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Quantitative validation of the I-FDG pipeline on analytic
          synthetic VSGs with a KNOWN dispersion curve c(f).

Experiments (mirrors the role of das_ani's src/eval.py):

- recovery   : write synthetic single-mode VSGs to disk in the das_ani file
               convention, run the full production pipeline
               (src.grad.run_pipeline), and measure the relative error of
               the recovered phase velocity against the true c(f) --
               with and without the fiber curvature term.
- timing     : wall time per VSG of the core pipeline.

Outputs:
- benchmark_results.csv  (one row per experiment configuration)
- run_manifest.json

Run:
    python -m src.eval --outdir data/benchmarks/synth --n_vsg 9
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.grad import run_pipeline
from src.synth import dispersion_exponential, make_synthetic_vsg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    experiment: str
    laplacian_mode: str
    include_curvature: bool
    n_vsg: int
    nch: int
    nlag: int
    noise_rel: float

    wall_sec: float
    sec_per_vsg: float

    # Recovery fidelity over the valid (interior) region of the product
    rel_err_median: Optional[float] = None
    rel_err_p95: Optional[float] = None
    rel_err_max: Optional[float] = None
    frac_valid: Optional[float] = None
    note: Optional[str] = None


def _write_synthetic_deployment(
    out_dir: Path,
    *,
    n_vsg: int,
    nch: int,
    dx: float,
    fs: float,
    nlag: int,
    f0: float,
    noise_rel: float,
    c_of_f,
    rng: np.random.Generator,
) -> None:
    """Save n_vsg synthetic VSGs as das_ani-convention .npy files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Spread virtual sources across the interior of the fiber.
    vs_rows = np.linspace(0.15 * nch, 0.85 * nch, n_vsg).astype(int)
    for vs in vs_rows:
        v = make_synthetic_vsg(
            nch=nch, dx=dx, vs_row=int(vs), fs=fs, nlag=nlag,
            c_of_f=c_of_f, f0=f0, noise_rel=noise_rel, rng=rng,
        )
        np.save(out_dir / f"20250101_000000_synth_cc_{int(vs):03d}_v1.npy",
                v.data)


def _grad_cfg(vsg_root: Path, out_root: Path, *, fs: float, dx: float,
              f_min: float, f_max: float, c_ref: float,
              include_curvature: bool) -> dict:
    """Config dict for run_pipeline matching the synthetic deployment."""
    return {
        "paths": {"vsg_root": str(vsg_root), "vsg_pattern": "*_cc_*.npy",
                  "output_root": str(out_root)},
        "data": {"fs_proc": fs, "dx": dx, "first_chan": 0},
        "mask": {
            # Constant-velocity corridor wide enough to bracket the full
            # dispersion range of the synthetic c(f).
            "v_inner": 0.6 * c_ref, "v_outer": 1.7 * c_ref,
            "t_inner": 2.0, "t_outer": -2.0,
            "taper_sec": 0.5, "causal_only": True,
        },
        "band": {"f_min": f_min, "f_max": f_max},
        "laplacian": {"mode": "fiber_1d",
                      "include_curvature": include_curvature,
                      "channel_taper_alpha": 0.2},
        # Near-source exclusion radius ~ 2 wavelengths at band center.
        "stack": {"r_exclude_m": 2.0 * c_ref / (0.5 * (f_min + f_max))},
        "post": {"median_size": 5, "edge_frac": 0.15,
                 "r_min_m": 0.0,
                 "v_min": 50.0, "v_max": 10000.0},
        "output": {"tag": "synth_curv" if include_curvature else "synth_nocurv"},
    }


def recovery_metrics(product_npz: Path, c_of_f) -> dict:
    """
    Compare the recovered velocity cube against the true dispersion curve.

    :return: dict with median/p95/max relative error over valid samples and
             the fraction of finite samples.
    """
    z = np.load(product_npz, allow_pickle=False)
    vel = z["vel"]                 # (nch, nfreq)
    freqs = z["freqs"]
    c_true = np.asarray(c_of_f(freqs), dtype=np.float64)[None, :]

    finite = np.isfinite(vel)
    if not np.any(finite):
        return {"rel_err_median": np.nan, "rel_err_p95": np.nan,
                "rel_err_max": np.nan, "frac_valid": 0.0}
    rel = np.abs(vel - c_true) / c_true
    rel = rel[finite]
    return {
        "rel_err_median": float(np.median(rel)),
        "rel_err_p95": float(np.percentile(rel, 95)),
        "rel_err_max": float(np.max(rel)),
        "frac_valid": float(np.mean(finite)),
    }


def run_recovery_experiment(
    outdir: Path,
    *,
    n_vsg: int,
    nch: int,
    dx: float,
    fs: float,
    nlag: int,
    f_min: float,
    f_max: float,
    f0: float,
    noise_rel: float,
    seed: int,
) -> list[RunResult]:
    """Synthetic end-to-end recovery, with and without the curvature term."""
    rng = np.random.default_rng(seed)
    c_of_f = dispersion_exponential(c_low=1200.0, c_high=800.0, f_corner=2.0)
    c_ref = float(c_of_f(np.array([0.5 * (f_min + f_max)]))[0])

    vsg_dir = outdir / "synthetic_vsgs"
    _write_synthetic_deployment(
        vsg_dir, n_vsg=n_vsg, nch=nch, dx=dx, fs=fs, nlag=nlag,
        f0=f0, noise_rel=noise_rel, c_of_f=c_of_f, rng=rng,
    )

    results: list[RunResult] = []
    for include_curvature in (True, False):
        cfg = _grad_cfg(vsg_dir, outdir / "products", fs=fs, dx=dx,
                        f_min=f_min, f_max=f_max, c_ref=c_ref,
                        include_curvature=include_curvature)
        t0 = time.perf_counter()
        product = run_pipeline(cfg)
        wall = time.perf_counter() - t0

        fid = recovery_metrics(product, c_of_f)
        results.append(RunResult(
            experiment="recovery",
            laplacian_mode="fiber_1d",
            include_curvature=include_curvature,
            n_vsg=n_vsg, nch=nch, nlag=nlag, noise_rel=noise_rel,
            wall_sec=wall, sec_per_vsg=wall / max(1, n_vsg),
            note=product.name,
            **fid,
        ))
        logger.info(
            "[recovery] curvature=%s | rel_err median=%.4f p95=%.4f | "
            "valid=%.2f | %.2fs (%.3fs/VSG)",
            include_curvature, fid["rel_err_median"], fid["rel_err_p95"],
            fid["frac_valid"], wall, wall / max(1, n_vsg),
        )
    return results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic-recovery benchmark for the das_grad I-FDG pipeline"
    )
    p.add_argument("--outdir", type=str, default="./data/benchmarks/synth")
    p.add_argument("--n_vsg", type=int, default=9)
    p.add_argument("--nch", type=int, default=400)
    p.add_argument("--dx", type=float, default=8.0)
    p.add_argument("--fs", type=float, default=50.0)
    p.add_argument("--nlag", type=int, default=2001)
    p.add_argument("--f_min", type=float, default=1.0)
    p.add_argument("--f_max", type=float, default=8.0)
    p.add_argument("--f0", type=float, default=4.0, help="Ricker central frequency")
    p.add_argument("--noise_rel", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20260611)
    return p.parse_args(args=argv)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = vars(args).copy()
    with open(outdir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    results = run_recovery_experiment(
        outdir,
        n_vsg=int(args.n_vsg), nch=int(args.nch), dx=float(args.dx),
        fs=float(args.fs), nlag=int(args.nlag),
        f_min=float(args.f_min), f_max=float(args.f_max),
        f0=float(args.f0), noise_rel=float(args.noise_rel),
        seed=int(args.seed),
    )

    # CSV via pandas when available, else a minimal writer.
    csv_path = outdir / "benchmark_results.csv"
    rows = [asdict(r) for r in results]
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    except ImportError:  # pragma: no cover
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    logger.info("Saved results: %s", csv_path)


if __name__ == "__main__":
    main()

# Example:
# python -m src.eval --outdir data/benchmarks/synth --n_vsg 9 --noise_rel 0.01
