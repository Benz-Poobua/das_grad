"""Pseudospectral derivative operators: exactness and geometry."""
from __future__ import annotations

import numpy as np
import pytest

from src.laplacian import laplacian_fiber, laplacian_grid, spectral_derivative
from src.synth import outgoing_kernel
from src.utils import cosine_taper

N, DX = 256, 8.0


def _periodic_mode(m: int) -> tuple[np.ndarray, float]:
    """Complex exponential with exactly m cycles over the window."""
    x = np.arange(N) * DX
    k = 2.0 * np.pi * m / (N * DX)
    return np.exp(1j * k * x), k


@pytest.mark.parametrize("m", [3, 13, 40])
@pytest.mark.parametrize("order", [1, 2])
def test_spectral_derivative_exact_on_periodic_modes(m, order):
    f, k = _periodic_mode(m)
    d = spectral_derivative(f, DX, order=order, axis=0)
    expected = (1j * k) ** order * f
    assert np.allclose(d, expected, rtol=1e-9, atol=1e-12)


def test_spectral_derivative_axis_handling(rng):
    a = rng.standard_normal((N, 7))
    d_cols = spectral_derivative(a, DX, order=2, axis=0)
    for j in range(7):
        d_1d = spectral_derivative(a[:, j], DX, order=2, axis=0)
        assert np.allclose(d_cols[:, j], d_1d)


def test_fiber_laplacian_cylindrical_wave_interior():
    """
    The cylindrical kernel satisfies (d2/dr2 + (1/r) d/dr) U = -k^2 U
    exactly; the pseudospectral operator must reproduce -k^2 in the interior
    (edges contaminated by windowing leakage are excluded).
    """
    c0, freq = 1000.0, 4.0
    k = 2.0 * np.pi * freq / c0
    vs = N // 2
    offset = (np.arange(N) - vs) * DX
    r = np.abs(offset)
    U = outgoing_kernel(k * np.maximum(r, 1e-3))
    U[vs] = 0.0
    Uw = U * cosine_taper(N, 0.2)

    lap = laplacian_fiber(Uw[:, None], DX, offset=offset, include_curvature=True)[:, 0]
    idx = np.arange(N)
    interior = (idx > 40) & (idx < N - 40) & (r > 400.0)
    ratio = np.real(-lap[interior] / Uw[interior]) / k**2
    assert np.median(np.abs(ratio - 1.0)) < 0.08
    # Dropping the curvature term must (slightly) degrade the estimate.
    lap_nc = laplacian_fiber(Uw[:, None], DX, offset=offset,
                             include_curvature=False)[:, 0]
    ratio_nc = np.real(-lap_nc[interior] / Uw[interior]) / k**2
    assert np.median(np.abs(ratio_nc - 1.0)) >= np.median(np.abs(ratio - 1.0)) - 1e-6


def test_fiber_laplacian_requires_offset_for_curvature():
    with pytest.raises(ValueError, match="offset"):
        laplacian_fiber(np.zeros((8, 2)), DX, include_curvature=True)


def test_grid_laplacian_exact_on_plane_waves():
    ny, nx, dy, dx = 32, 48, 10.0, 5.0
    y = np.arange(ny) * dy
    x = np.arange(nx) * dx
    ky = 2 * np.pi * 3 / (ny * dy)
    kx = 2 * np.pi * 5 / (nx * dx)
    U = np.exp(1j * (ky * y[:, None] + kx * x[None, :]))
    lap = laplacian_grid(U, dx, dy, axes=(0, 1))
    assert np.allclose(lap, -(kx**2 + ky**2) * U, rtol=1e-9, atol=1e-12)


def test_grid_laplacian_broadcasts_over_frequency():
    ny, nx = 16, 16
    U = np.random.default_rng(0).standard_normal((ny, nx, 5)) * (1 + 0j)
    lap = laplacian_grid(U, 5.0, 5.0, axes=(0, 1))
    for j in range(5):
        lap_j = laplacian_grid(U[:, :, j], 5.0, 5.0, axes=(0, 1))
        assert np.allclose(lap[:, :, j], lap_j)
