"""
:module: src/laplacian.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Pseudospectral spatial-derivative operators for gradiometry.

Two geometries are provided:

- ``grid_2d``  : the Davis et al. (2026) operator. 2-D spatial FFT over the
  map coordinates, multiplication by -(kx^2 + ky^2), inverse FFT. Exact for
  band-limited, regularly sampled fields; carries none of the
  finite-difference stencil bias of de Ridder & Curtis (2017, Fig. 3).

- ``fiber_1d`` : the DAS case. A single straight fiber samples only the
  along-fiber second derivative d^2/dl^2. For a VSG whose virtual source
  lies ON the same fiber, the far-field fundamental mode is axisymmetric
  about the source -- U = U(r), r = |l - l_s| -- so the full horizontal
  Laplacian along the fiber is EXACTLY the cylindrical form

      lap U = d^2 U / dr^2 + (1/r) dU/dr,

  with d/dr = sign(l - l_s) d/dl. The optional curvature term (1/r) dU/dr
  accounts for geometric spreading; dropping it biases the sloth near the
  source (relative error ~ 1/(2 (kr)^2) for a cylindrical wave) and is
  controlled by ``include_curvature``.

All operators act on the CHANNEL axis (axis 0) of complex frequency-domain
VSG slabs of shape (nch, nfreq) -- or (ny, nx, nfreq) for the grid -- and
assume the field has been channel-tapered (see mask.channel_taper_weights)
so the periodic FFT does not see array-edge discontinuities.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _wavenumber(n: int, d: float) -> np.ndarray:
    """Angular wavenumber axis (rad/m) for an n-point FFT with spacing d."""
    return 2.0 * np.pi * np.fft.fftfreq(n, d=float(d))


def spectral_derivative(arr: np.ndarray, d: float, *, order: int = 1,
                        axis: int = 0) -> np.ndarray:
    """
    Pseudospectral derivative of arbitrary order along ``axis``:
    FFT -> multiply by (i k)^order -> inverse FFT.

    Works for real or complex input; always returns complex.

    :param arr: Input array.
    :param d: Sample spacing along ``axis`` (m).
    :param order: Derivative order (1 = gradient, 2 = second derivative).
    :param axis: Axis along which to differentiate.
    """
    arr = np.asarray(arr)
    n = arr.shape[axis]
    k = _wavenumber(n, d)
    shape = [1] * arr.ndim
    shape[axis] = n
    mult = (1j * k.reshape(shape)) ** int(order)
    return np.fft.ifft(np.fft.fft(arr, axis=axis) * mult, axis=axis)


def laplacian_fiber(
    arr: np.ndarray,
    dx: float,
    *,
    offset: np.ndarray | None = None,
    include_curvature: bool = True,
    r_min: float | None = None,
) -> np.ndarray:
    """
    Along-fiber Laplacian of a (nch, ...) frequency-domain VSG slab.

    lap = d^2/dl^2                                  (include_curvature=False)
    lap = d^2/dr^2 + (1/r) d/dr                     (include_curvature=True)

    where r = |offset| and d/dr = sign(offset) * d/dl. The curvature term is
    exact for an axisymmetric (in-line virtual source) single-mode field.

    :param arr: (nch, nfreq) complex (or real) field; channel axis first.
    :param dx: Channel spacing (m).
    :param offset: (nch,) signed offsets from the VS (m). Required when
                   ``include_curvature`` is True.
    :param include_curvature: Add the (1/r) d/dr geometric-spreading term.
    :param r_min: Clip radius (m) below which the curvature term is held at
                  its r_min value to avoid the 1/r singularity at the VS.
                  Default: one channel spacing. The near-source region is
                  unreliable regardless (near-field, |H0| singularity) and
                  should be excluded via post.quality_mask.
    :return: complex array, same shape as ``arr``.
    """
    arr = np.asarray(arr)
    d2 = spectral_derivative(arr, dx, order=2, axis=0)
    if not include_curvature:
        return d2

    if offset is None:
        raise ValueError("laplacian_fiber: offset is required when include_curvature=True.")
    offset = np.asarray(offset, dtype=np.float64)
    if offset.shape[0] != arr.shape[0]:
        raise ValueError(
            f"laplacian_fiber: offset length {offset.shape[0]} != nch {arr.shape[0]}"
        )
    if r_min is None:
        r_min = float(dx)

    d1 = spectral_derivative(arr, dx, order=1, axis=0)
    r = np.maximum(np.abs(offset), float(r_min))
    sgn = np.sign(offset)
    sgn[sgn == 0] = 1.0  # the VS row itself; excluded downstream anyway
    shape = [1] * arr.ndim
    shape[0] = arr.shape[0]
    curv = (sgn / r).reshape(shape) * d1
    return d2 + curv


def laplacian_grid(
    arr: np.ndarray,
    dx: float,
    dy: float,
    *,
    axes: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """
    Pseudospectral 2-D horizontal Laplacian (Davis et al. 2026 workflow,
    steps 4-6): FFT2 -> multiply by -(kx^2 + ky^2) -> IFFT2.

    :param arr: (ny, nx, ...) field with the two map axes given by ``axes``.
    :param dx: Spacing along ``axes[1]`` (m).
    :param dy: Spacing along ``axes[0]`` (m).
    :return: complex array, same shape.
    """
    arr = np.asarray(arr)
    ay, ax = axes
    ky = _wavenumber(arr.shape[ay], dy)
    kx = _wavenumber(arr.shape[ax], dx)

    shape_y = [1] * arr.ndim
    shape_y[ay] = arr.shape[ay]
    shape_x = [1] * arr.ndim
    shape_x[ax] = arr.shape[ax]
    k2 = ky.reshape(shape_y) ** 2 + kx.reshape(shape_x) ** 2

    spec = np.fft.fft2(arr, axes=axes)
    return np.fft.ifft2(spec * (-k2), axes=axes)
