"""
:module: src/synth.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Analytic synthetic VSGs for validating the gradiometry pipeline.

A single-mode VSG in the far field is, frequency by frequency, the 2-D
Helmholtz Green's function: an outgoing cylindrical wave

    V(r, w) = A(w) * H0^(1)( k(w) r ),     k(w) = w / c(w),

(Gradiometry_Theory eqs. 66-67, 86; Davis et al. 2026). This module builds
exactly that field for a USER-CHOSEN dispersion curve c(w), inverse-FFTs it
to the lag domain, and wraps it as a :class:`src.vsg.VSG`. Because the true
c(w) is known, the recovery error of TDG/FDG/I-FDG can be measured
quantitatively -- this drives src/eval.py and tests/.

When scipy is installed the exact Hankel function is used; otherwise the
far-field asymptotic sqrt(2/(pi k r)) exp(i(kr - pi/4)) is substituted
(relative Helmholtz residual ~ 1/(8 (kr)^2), negligible for kr >~ 8).
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from src.vsg import VSG

logger = logging.getLogger(__name__)

try:
    from scipy.special import hankel2 as _hankel2
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - depends on environment
    _hankel2 = None
    HAVE_SCIPY = False


def outgoing_kernel(x: np.ndarray) -> np.ndarray:
    """
    Outgoing 2-D Helmholtz point-source kernel UNDER THE NUMPY FFT SIGN
    CONVENTION.

    Physics texts (Gradiometry_Theory eq. 32, e^{+i omega t} forward
    transform) write the outgoing cylindrical wave as H0^(1)(kr). numpy's
    ``rfft``/``irfft`` use the OPPOSITE sign (forward kernel e^{-i 2 pi f t}),
    under which a spectrum factor e^{-i k r} -- i.e. the CONJUGATE kernel
    H0^(2)(kr) = conj(H0^(1)(kr)) for real kr -- produces an arrival DELAYED
    by r/c, as required for a causal expanding wavetrain. (Getting this
    wrong time-reverses the gather and wraps the arrivals to the end of the
    lag window.) Both kernels satisfy the same Helmholtz equation, so the
    sloth estimate is unaffected -- but the torus mask, which selects the
    causal moveout, very much is.

    Exact via scipy when available, otherwise the large-argument asymptotic
    sqrt(2/(pi x)) exp(-i (x - pi/4)) (relative Helmholtz residual
    ~ 1/(8 x^2), negligible for x >~ 8).
    """
    x = np.asarray(x, dtype=np.float64)
    if HAVE_SCIPY:
        return _hankel2(0, x)
    safe = np.maximum(x, 1e-6)
    return np.sqrt(2.0 / (np.pi * safe)) * np.exp(-1j * (safe - np.pi / 4.0))


def ricker_spectrum(freqs: np.ndarray, f0: float) -> np.ndarray:
    """
    Amplitude spectrum of a zero-phase Ricker wavelet with central frequency
    f0 (the ambient source-time autocorrelation used by Davis et al. 2026).
    """
    f = np.asarray(freqs, dtype=np.float64)
    a = (f / float(f0)) ** 2
    return a * np.exp(1.0 - a)


def dispersion_exponential(
    c_low: float = 1200.0,
    c_high: float = 800.0,
    f_corner: float = 2.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Smooth, normally dispersive phase-velocity curve

        c(f) = c_high + (c_low - c_high) * exp(-f / f_corner),

    decreasing from ~c_low at f -> 0 (deep-sensing, fast) toward c_high at
    high f (shallow, slow) -- the typical fundamental-mode trend.

    :return: Callable c(f) accepting an array of frequencies in Hz.
    """
    def c_of_f(f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=np.float64)
        return float(c_high) + (float(c_low) - float(c_high)) * np.exp(-f / float(f_corner))
    return c_of_f


def make_synthetic_vsg(
    *,
    nch: int,
    dx: float,
    vs_row: int,
    fs: float,
    nlag: int,
    c_of_f: Callable[[np.ndarray], np.ndarray],
    f0: float = 4.0,
    causal_only: bool = True,
    noise_rel: float = 0.0,
    rng: np.random.Generator | None = None,
) -> VSG:
    """
    Build one analytic single-mode VSG on a straight fiber.

    Construction: for every positive frequency f of the lag axis,
        V(r, f) = A(f) H0^(1)(2 pi f r / c(f)),  A = Ricker(f; f0),
    then irfft over frequency to the (nch, nlag) lag domain. The result is
    a causal expanding wavetrain identical in character to Davis et al.
    (2026, Fig. 5b) after torus masking. The VS row itself (r = 0, where H0
    is singular) is set to zero.

    :param nch: Channels along the fiber.
    :param dx: Channel spacing (m).
    :param vs_row: Virtual-source row index.
    :param fs: Sampling rate (Hz).
    :param nlag: Length of the lag axis; forced odd (das_ani convention 2M+1).
    :param c_of_f: Dispersion curve c(f) in m/s (see dispersion_exponential).
    :param f0: Ricker central frequency (Hz).
    :param causal_only: If True the acausal half stays zero; otherwise the
                        causal half is mirrored (symmetric NCF).
    :param noise_rel: Optional white-noise level relative to the rms signal.
    :param rng: numpy Generator for the noise.
    :return: :class:`VSG` with exact geometry attached.
    """
    if nlag % 2 == 0:
        nlag += 1
    M = (nlag - 1) // 2
    if not (0 <= vs_row < nch):
        raise ValueError(f"vs_row {vs_row} outside [0, {nch})")

    offset = (np.arange(nch, dtype=np.float64) - float(vs_row)) * float(dx)
    r = np.abs(offset)

    # Build the causal half on the (M+1)-point grid tau = 0..M/fs.
    n_c = M + 1
    freqs = np.fft.rfftfreq(n_c, d=1.0 / float(fs))
    spec = np.zeros((nch, freqs.size), dtype=np.complex128)

    pos = freqs > 0.0
    f_pos = freqs[pos]
    c = np.asarray(c_of_f(f_pos), dtype=np.float64)
    if np.any(c <= 0):
        raise ValueError("c_of_f returned non-positive velocities.")
    k = 2.0 * np.pi * f_pos / c                        # (nf,)
    A = ricker_spectrum(f_pos, f0)

    live = r > 0
    kr = np.outer(r[live], k)                          # (nch_live, nf_pos)
    spec_live = A[None, :] * outgoing_kernel(kr)
    spec[np.ix_(np.where(live)[0], np.where(pos)[0])] = spec_live

    causal = np.fft.irfft(spec, n=n_c, axis=1)

    data = np.zeros((nch, nlag), dtype=np.float64)
    data[:, M:] = causal
    if not causal_only:
        data[:, :M + 1] += causal[:, ::-1]
        data[:, M] /= 2.0   # zero lag was added twice

    if noise_rel > 0.0:
        rng = rng or np.random.default_rng(0)
        sigma = float(noise_rel) * float(np.std(data[live]))
        data += sigma * rng.standard_normal(data.shape)

    peak = float(np.max(np.abs(data)))
    if peak > 0:
        data /= peak

    lag = np.arange(-M, M + 1, dtype=np.float64) / float(fs)
    return VSG(
        data=data.astype(np.float32),
        lag=lag,
        offset=offset,
        vs_row=int(vs_row),
        name=f"synthetic_cc_{vs_row:03d}",
        meta={"f0": f0, "causal_only": causal_only,
              "hankel": "exact" if HAVE_SCIPY else "asymptotic"},
    )
