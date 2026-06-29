"""Torus masking, VSG I/O, synthetic generator, and post-processing."""
from __future__ import annotations

import numpy as np
import pytest

from src.mask import apply_mask, channel_taper_weights, torus_mask
from src.post import median_filter_1d, quality_mask
from src.synth import dispersion_exponential, make_synthetic_vsg
from src.vsg import VSG, lag_axis, load_vsg, parse_vs_index, discover_vsgs
from tests.conftest import C0, DX, FS, NCH, NLAG, homogeneous_vsg


# ---------------------------------------------------------------------------
# Torus mask
# ---------------------------------------------------------------------------
def test_torus_mask_keeps_direct_arrival_kills_late_noise():
    vsg = homogeneous_vsg()
    w = torus_mask(vsg.lag, vsg.offset, v_inner=600.0, v_outer=1700.0,
                   t_inner=2.0, t_outer=-2.0, taper_sec=0.5)
    assert w.shape == vsg.data.shape
    assert np.all((w >= 0.0) & (w <= 1.0))

    ch = vsg.vs_row + 80          # r = 640 m -> direct arrival at 0.64 s
    r = vsg.r[ch]
    i_arr = np.argmin(np.abs(vsg.lag - r / C0))
    assert w[ch, i_arr] == pytest.approx(1.0)
    # Far outside the corridor (late coda) and the acausal side: muted.
    i_late = np.argmin(np.abs(vsg.lag - (r / 600.0 + 4.0)))
    assert w[ch, i_late] == 0.0
    assert np.all(w[:, vsg.lag < 0] == 0.0)   # causal_only


def test_torus_mask_acausal_mirror():
    vsg = homogeneous_vsg()
    w = torus_mask(vsg.lag, vsg.offset, v_inner=600.0, v_outer=1700.0,
                   t_inner=2.0, t_outer=-2.0, taper_sec=0.5, causal_only=False)
    ch = vsg.vs_row + 80
    r = vsg.r[ch]
    i_acausal = np.argmin(np.abs(vsg.lag + r / C0))
    assert w[ch, i_acausal] == pytest.approx(1.0)


def test_torus_mask_rejects_bad_velocity():
    vsg = homogeneous_vsg()
    with pytest.raises(ValueError):
        torus_mask(vsg.lag, vsg.offset, v_inner=-1.0, v_outer=800.0,
                   t_inner=1.0, t_outer=-1.0)


def test_apply_mask_shape_check(rng):
    with pytest.raises(ValueError):
        apply_mask(rng.standard_normal((3, 4)), np.ones((4, 3)))


def test_channel_taper_endpoints():
    w = channel_taper_weights(101, 0.2)
    assert w[0] == pytest.approx(0.0, abs=1e-12)
    assert w[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.all(w[40:60] == 1.0)


# ---------------------------------------------------------------------------
# VSG I/O (das_ani file contract)
# ---------------------------------------------------------------------------
def test_parse_vs_index():
    assert parse_vs_index("20250722_025000_urban_cc_080_v1.npy") == 80
    assert parse_vs_index("20250722_cc_080_7d_v1.npy") == 80
    with pytest.raises(ValueError):
        parse_vs_index("nothing_here.npy")


def test_lag_axis_requires_odd():
    lag = lag_axis(2001, 50.0)
    assert lag[1000] == 0.0 and lag[0] == pytest.approx(-20.0)
    with pytest.raises(ValueError):
        lag_axis(2000, 50.0)


def test_load_vsg_roundtrip(tmp_path, rng):
    data = rng.standard_normal((NCH, NLAG)).astype(np.float32)
    p = tmp_path / "20250722_025000_test_cc_040_v1.npy"
    np.save(p, data)
    v = load_vsg(p, fs=FS, dx=DX)
    assert v.vs_row == 40
    assert v.offset[40] == 0.0
    assert v.offset[41] == pytest.approx(DX)
    assert v.fs == pytest.approx(FS)
    assert v.dx == pytest.approx(DX)
    assert np.allclose(v.data, data)


def test_discover_vsgs_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_vsgs(tmp_path)


def test_vsg_validates_geometry(rng):
    with pytest.raises(ValueError):
        VSG(data=rng.standard_normal((4, 11)), lag=np.zeros(10),
            offset=np.zeros(4), vs_row=0)
    with pytest.raises(ValueError):
        VSG(data=rng.standard_normal((4, 11)), lag=np.zeros(11),
            offset=np.zeros(4), vs_row=7)


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_arrival_at_r_over_c():
    vsg = homogeneous_vsg()
    for ch in (vsg.vs_row + 60, vsg.vs_row - 90):
        r = vsg.r[ch]
        t_peak = vsg.lag[np.argmax(np.abs(vsg.data[ch]))]
        assert t_peak == pytest.approx(r / C0, abs=0.15), (
            f"channel at r={r:.0f} m peaks at {t_peak:.2f}s, expected ~{r / C0:.2f}s"
        )


def test_synthetic_causal_only_acausal_is_zero():
    vsg = homogeneous_vsg()
    assert np.all(vsg.data[:, vsg.lag < 0] == 0.0)


def test_synthetic_dispersion_changes_moveout():
    c_disp = dispersion_exponential(c_low=1200.0, c_high=800.0, f_corner=2.0)
    v = make_synthetic_vsg(nch=NCH, dx=DX, vs_row=NCH // 2, fs=FS, nlag=NLAG,
                           c_of_f=c_disp, f0=4.0)
    assert v.data.shape == (NCH, NLAG)
    assert np.isfinite(v.data).all()


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def test_median_filter_removes_outlier_keeps_nan():
    a = np.ones((21, 2))
    a[10, 0] = 100.0       # outlier
    a[5, 1] = np.nan
    out = median_filter_1d(a, 5, axis=0)
    assert out[10, 0] == pytest.approx(1.0)
    assert out[5, 1] == pytest.approx(1.0)  # NaN bridged by neighbours


def test_quality_mask_edges_and_near_source():
    offs = [np.arange(100, dtype=float) * 8.0 - 200.0]   # VS at row 25
    valid = quality_mask(100, offs, edge_frac=0.1, r_min_m=50.0)
    assert not valid[:10].any() and not valid[-10:].any()
    # rows 20..30 are within 50 m of the (single) source -> masked
    assert not valid[25]
    assert valid[50]
