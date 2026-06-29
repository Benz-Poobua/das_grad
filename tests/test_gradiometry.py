"""Sloth solvers: TDG / FDG / I-FDG correctness on analytic fields."""
from __future__ import annotations

import numpy as np
import pytest

from src.gradiometry import (
    fdg_sloth,
    fdg_sloth_stack,
    ifdg_stack,
    sloth_to_velocity,
    tdg_sloth,
    temporal_fft,
)
from src.laplacian import laplacian_fiber
from src.mask import apply_mask, channel_taper_weights, torus_mask
from tests.conftest import C0, DX, F_MAX, F_MIN, FS, NCH, base_cfg, homogeneous_vsg, interior_mask


# ---------------------------------------------------------------------------
# temporal_fft
# ---------------------------------------------------------------------------
def test_temporal_fft_band_selection(rng):
    x = rng.standard_normal((4, 500))
    V, freqs = temporal_fft(x, fs=50.0, f_min=2.0, f_max=10.0)
    assert freqs.min() > 2.0 - 0.11 and freqs.max() <= 10.0
    assert 0.0 not in freqs
    assert V.shape == (4, freqs.size)


def test_temporal_fft_rejects_empty_band(rng):
    with pytest.raises(ValueError):
        temporal_fft(rng.standard_normal((2, 100)), fs=50.0, f_min=24.9, f_max=24.95)


# ---------------------------------------------------------------------------
# FDG on the analytic homogeneous VSG
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def homo_chain():
    """Masked+tapered homogeneous VSG, its spectrum and fiber Laplacian."""
    vsg = homogeneous_vsg()
    cfg = base_cfg()
    w = torus_mask(vsg.lag, vsg.offset, v_inner=600.0, v_outer=1700.0,
                   t_inner=2.0, t_outer=-2.0, taper_sec=0.5)
    d = apply_mask(vsg.data, w) * channel_taper_weights(vsg.nch, 0.2)[:, None]
    V, freqs = temporal_fft(d, vsg.fs, f_min=F_MIN, f_max=F_MAX)
    lap = laplacian_fiber(V, vsg.dx, offset=vsg.offset, include_curvature=True)
    return vsg, V, lap, freqs


def test_fdg_recovers_homogeneous_velocity(homo_chain):
    vsg, V, lap, freqs = homo_chain
    s2 = fdg_sloth(V, lap, freqs)
    vel = sloth_to_velocity(s2, v_min=50.0, v_max=10000.0)
    rel = np.abs(vel[interior_mask(vsg), :] - C0) / C0
    assert np.nanmedian(rel) < 0.03, (
        f"FDG failed to recover c={C0} m/s (median rel err "
        f"{np.nanmedian(rel):.4f})"
    )


def test_fdg_imaginary_part_is_small(homo_chain):
    """For a clean Helmholtz field, Im(s2) << Re(s2) (transport residual)."""
    vsg, V, lap, freqs = homo_chain
    s2 = fdg_sloth(V, lap, freqs)
    s2_i = s2[interior_mask(vsg), :]
    ratio = np.abs(np.imag(s2_i)) / np.abs(np.real(s2_i))
    assert np.nanmedian(ratio) < 0.2


def test_fdg_sloth_stack_collapses_frequency(homo_chain):
    vsg, V, lap, freqs = homo_chain
    s2_stack = fdg_sloth_stack(V, lap, freqs)
    assert s2_stack.shape == (vsg.nch,)
    vel = sloth_to_velocity(s2_stack, v_min=50.0, v_max=10000.0)
    rel = np.abs(vel[interior_mask(vsg)] - C0) / C0
    assert np.nanmedian(rel) < 0.03


def test_fdg_rejects_zero_frequency(homo_chain):
    _, V, lap, freqs = homo_chain
    bad = freqs.copy()
    bad[0] = 0.0
    with pytest.raises(ValueError):
        fdg_sloth(V, lap, bad)


# ---------------------------------------------------------------------------
# TDG (least squares over lag)
# ---------------------------------------------------------------------------
def test_tdg_recovers_homogeneous_sloth():
    vsg = homogeneous_vsg()
    w = torus_mask(vsg.lag, vsg.offset, v_inner=600.0, v_outer=1700.0,
                   t_inner=2.0, t_outer=-2.0, taper_sec=0.5)
    d = apply_mask(vsg.data, w) * channel_taper_weights(vsg.nch, 0.2)[:, None]
    lap = laplacian_fiber(d, vsg.dx, offset=vsg.offset, include_curvature=True)
    s2 = tdg_sloth(d, lap, vsg.fs, lag_axis=-1)
    vel = sloth_to_velocity(s2, v_min=50.0, v_max=10000.0)
    rel = np.abs(vel[interior_mask(vsg)] - C0) / C0
    # np.gradient acceleration is 2nd-order FD -> looser tolerance than FDG.
    assert np.nanmedian(rel) < 0.06


# ---------------------------------------------------------------------------
# I-FDG stacking
# ---------------------------------------------------------------------------
def test_ifdg_stack_plain_is_mean(rng):
    a = rng.standard_normal((6, 4)) + 1j * rng.standard_normal((6, 4))
    b = rng.standard_normal((6, 4)) + 1j * rng.standard_normal((6, 4))
    out = ifdg_stack([a, b])
    assert np.allclose(out, (a + b) / 2.0)


def test_ifdg_stack_unit_weights_match_plain(rng):
    a = rng.standard_normal((6, 4)) + 0j
    b = rng.standard_normal((6, 4)) + 0j
    w = np.ones(6)
    assert np.allclose(ifdg_stack([a, b]), ifdg_stack([a, b], weights=[w, w]))


def test_ifdg_stack_weighted_excludes_and_nans(rng):
    a = np.full((4, 3), 1.0 + 0j)
    b = np.full((4, 3), 3.0 + 0j)
    wa = np.array([1.0, 1.0, 0.0, 0.0])
    wb = np.array([1.0, 0.0, 1.0, 0.0])
    out = ifdg_stack([a, b], weights=[wa, wb])
    assert np.allclose(out[0, :], 2.0)   # both contribute
    assert np.allclose(out[1, :], 1.0)   # only a
    assert np.allclose(out[2, :], 3.0)   # only b
    assert np.all(np.isnan(out[3, :].real))  # no coverage -> NaN


def test_ifdg_stack_validates_inputs(rng):
    with pytest.raises(ValueError):
        ifdg_stack([])
    with pytest.raises(ValueError):
        ifdg_stack([np.zeros((2, 2)), np.zeros((3, 2))])
    with pytest.raises(ValueError):
        ifdg_stack([np.zeros((2, 2))], weights=[])


# ---------------------------------------------------------------------------
# Sloth -> velocity
# ---------------------------------------------------------------------------
def test_sloth_to_velocity_masks_nonphysical():
    s2 = np.array([1.0 / 1000.0**2, -1e-6, 0.0, 1.0 / 100.0**2]) + 0j
    v = sloth_to_velocity(s2, v_min=200.0, v_max=5000.0)
    assert v[0] == pytest.approx(1000.0)
    assert np.isnan(v[1]) and np.isnan(v[2])
    assert np.isnan(v[3])  # 100 m/s < v_min
