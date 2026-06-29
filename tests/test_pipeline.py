"""
End-to-end I-FDG pipeline on a synthetic deployment with known dispersion:
write das_ani-convention VSG files -> run_pipeline -> compare the recovered
phase-velocity cube against the true c(f).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.eval import recovery_metrics, run_recovery_experiment
from src.grad import run_pipeline
from src.synth import dispersion_exponential, make_synthetic_vsg
from tests.conftest import DX, FS, NLAG

NCH = 256
N_VSG = 5
F_MIN, F_MAX = 1.5, 8.0


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    """Synthetic VSG files on disk + matching pipeline config."""
    tmp = tmp_path_factory.mktemp("grad_e2e")
    vsg_dir = tmp / "vsgs"
    vsg_dir.mkdir()
    c_of_f = dispersion_exponential(c_low=1200.0, c_high=800.0, f_corner=2.0)

    vs_rows = np.linspace(0.2 * NCH, 0.8 * NCH, N_VSG).astype(int)
    for vs in vs_rows:
        v = make_synthetic_vsg(nch=NCH, dx=DX, vs_row=int(vs), fs=FS,
                               nlag=NLAG, c_of_f=c_of_f, f0=4.0)
        np.save(vsg_dir / f"20250101_000000_synth_cc_{int(vs):03d}_v1.npy", v.data)

    cfg = {
        "paths": {"vsg_root": str(vsg_dir), "vsg_pattern": "*_cc_*.npy",
                  "output_root": str(tmp / "products")},
        "data": {"fs_proc": FS, "dx": DX, "first_chan": 0},
        "mask": {"v_inner": 550.0, "v_outer": 1700.0,
                 "t_inner": 2.0, "t_outer": -2.0,
                 "taper_sec": 0.5, "causal_only": True},
        "band": {"f_min": F_MIN, "f_max": F_MAX},
        "laplacian": {"mode": "fiber_1d", "include_curvature": True,
                      "channel_taper_alpha": 0.2},
        "stack": {"r_exclude_m": 400.0},
        "post": {"median_size": 5, "edge_frac": 0.15, "r_min_m": 0.0,
                 "v_min": 50.0, "v_max": 10000.0},
        "output": {"tag": "e2e"},
    }
    return cfg, c_of_f


def test_pipeline_product_structure(deployment):
    cfg, _ = deployment
    out = run_pipeline(cfg)
    assert out.exists() and out.name == "grad_vsch_e2e.npz"
    z = np.load(out, allow_pickle=False)
    for key in ("s2", "vel", "freqs", "positions", "valid", "meta_json"):
        assert key in z, f"missing product key {key}"
    nch, nfreq = z["vel"].shape
    assert nch == NCH and nfreq == z["freqs"].size
    assert z["positions"].shape == (NCH,)
    assert z["s2"].dtype == np.complex64


def test_pipeline_recovers_dispersion_curve(deployment):
    cfg, c_of_f = deployment
    out = run_pipeline(cfg)
    z = np.load(out)
    vel, freqs, valid = z["vel"], z["freqs"], z["valid"]
    c_true = c_of_f(freqs)

    # Frequency-wise array-median profile vs truth, away from the aperture
    # limit (lambda < aperture/4 -> f >= ~2.3 Hz here).
    sel = freqs >= 2.5
    prof = np.nanmedian(vel[valid, :], axis=0)
    rel = np.abs(prof[sel] - c_true[sel]) / c_true[sel]
    assert np.nanmax(rel) < 0.03, (
        f"dispersion profile error up to {np.nanmax(rel):.3f} "
        f"(freqs {freqs[sel][np.nanargmax(rel)]:.2f} Hz)"
    )
    # Pixel-level: median over the whole valid cube.
    rel_px = np.abs(vel[valid][:, sel] - c_true[sel][None, :]) / c_true[sel][None, :]
    assert np.nanmedian(rel_px) < 0.02


def test_recovery_experiment_runner(tmp_path):
    """src.eval orchestration: rows for curvature on/off, sane metrics."""
    results = run_recovery_experiment(
        tmp_path, n_vsg=3, nch=200, dx=DX, fs=FS, nlag=1001,
        f_min=2.0, f_max=8.0, f0=4.0, noise_rel=0.0, seed=1,
    )
    assert {r.include_curvature for r in results} == {True, False}
    for r in results:
        assert r.experiment == "recovery"
        assert r.rel_err_median is not None and r.rel_err_median < 0.05
        assert 0.0 < r.frac_valid <= 1.0
        assert r.wall_sec > 0


def test_recovery_metrics_empty_product(tmp_path):
    p = tmp_path / "empty.npz"
    np.savez(p, vel=np.full((4, 3), np.nan), freqs=np.array([1.0, 2.0, 3.0]))
    out = recovery_metrics(p, dispersion_exponential())
    assert out["frac_valid"] == 0.0
