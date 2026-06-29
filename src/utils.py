"""
:module: src/utils.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Shared utilities for the DAS gradiometry (das_grad) pipeline.

Provenance: load_config / get_cfg / nextpow2 / timeit are copied (lightly
trimmed -- no torch/CUDA hooks) from the companion das_ani repository
(https://github.com/Benz-Poobua/das_ani, src/utils.py) so that das_grad has
no import-time dependency on das_ani. The VSG *file format* is the only
coupling between the two projects; see src/vsg.py.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar, Union, Tuple, Optional, Literal, List

from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.fft import fft2, fftshift, ifft2, ifftshift

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike, Path]
F = TypeVar("F", bound=Callable[..., Any])


# ==============================================================
# 1. Config helpers (copied from das_ani)
# ==============================================================
def load_config(path: PathLike) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    suf = p.suffix.lower()
    if suf in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "YAML config requested but PyYAML is not installed. "
                "Run: pip install pyyaml"
            ) from e
        with p.open("r") as f:
            cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                raise ValueError("Config root must be a mapping/dict.")
            return cfg

    if suf == ".json":
        with p.open("r") as f:
            cfg = json.load(f)
            if not isinstance(cfg, dict):
                raise ValueError("Config root must be a mapping/dict.")
            return cfg

    raise ValueError(f"Unsupported config extension: {suf} (use .yaml/.yml/.json)")


def get_cfg(cfg: Mapping[str, Any], keys: Sequence[str], default: Any = None,
            *, required: bool = False) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            if required:
                raise KeyError(f"Missing config key: {'.'.join(keys)}")
            return default
        cur = cur[k]
    return cur

def parse_ncf_stack_filename(fname: str) -> Tuple[str, str, str, str]:
    base = os.path.basename(fname)

    m = re.match(r"(.+?)_cc_(\d+)_([^_]+)_(.+)\.np[yz]$", base)

    if m is None:
        raise ValueError(f"Stack filename not recognized: {fname}")

    date, vs, window, mode = m.groups()
    return date, vs, window, mode

def get_vs_number(filepath: str) -> int:
    """
    Extracts the Virtual Source integer from a standard NCF filename for proper sorting.
    Handles variable prefix lengths by looking for the '_cc_[number]' pattern.
    """
    filename = os.path.basename(filepath)
    match = re.search(r'_cc_(\d+)', filename)
    
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not parse VS number from filename: {filename}")

# ==============================================================
# 2. Math helpers
# ==============================================================
def nextpow2(x: Union[int, float]) -> int:
    xf = float(x)
    if xf <= 1.0:
        return 1
    return int(2 ** int(np.ceil(np.log2(xf))))


def cosine_taper(n: int, alpha: float) -> np.ndarray:
    """
    Tukey (tapered-cosine) window of length ``n``, numpy-only.

    Equivalent to ``scipy.signal.windows.tukey(n, alpha)``: flat top with a
    raised-cosine ramp over a fraction ``alpha/2`` of the window at each end.
    ``alpha=0`` is boxcar, ``alpha=1`` is a Hann window.

    :param n: Window length (samples).
    :param alpha: Fraction of the window inside the tapered regions.
    :return: 1-D float64 window of length n.
    """
    if n <= 0:
        return np.zeros(0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    w = np.ones(n, dtype=np.float64)
    if alpha == 0.0 or n == 1:
        return w
    edge = int(np.floor(alpha * (n - 1) / 2.0))
    if edge < 1:
        return w
    ramp = 0.5 * (1.0 + np.cos(np.pi * (np.arange(edge + 1) / edge - 1.0)))
    w[: edge + 1] = ramp
    w[n - edge - 1:] = ramp[::-1]
    return w


# ==============================================================
# 3. Timing decorator (trimmed das_ani version, no CUDA hooks)
# ==============================================================
def timeit(func: F) -> F:
    """Log the wall time of a function call at INFO level."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        log = logging.getLogger(func.__module__)
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        log.info("[%s] elapsed = %.3f s", func.__name__, time.perf_counter() - t0)
        return result
    return wrapper


# ==============================================================
# 4. Optional-dependency shims
# ==============================================================
def get_tqdm():
    """Return tqdm.tqdm if installed, else a transparent fallback."""
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:  # pragma: no cover - depends on environment
        def _fallback(x=None, *a, **k):
            return x
        return _fallback
    

# ==============================================================
# 5. FK filtering (from Haipeng Li)
# ==============================================================
def fk_transform(
    data: np.ndarray,
    dt: float,
    dx: float,
    pad_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the 2D Forward Fourier Transform from the (x, t) domain to the (k, f) domain.

    **DSP Theory:**
    The FK transform (Frequency-Wavenumber) decomposes a wavefield into its constituent 
    plane waves. In DAS data, the horizontal axis (x) represents the fiber distance and 
    the vertical axis (t) represents time. The 2D FFT maps these to spatial frequency 
    (wavenumber, k) and temporal frequency (f).

    :param data: 2D array of DAS data, shape (n_channels, n_samples).
    :param dt: Temporal sampling interval in seconds.
    :param dx: Spatial sampling interval (channel spacing) in meters.
    :param pad_shape: Optional tuple (nx_pad, nt_pad) for zero-padding. Padding improves 
                      spectral resolution and prevents circular convolution artifacts.
    :return: (f_axis, k_axis, fk_spectrum) 
             f_axis: temporal frequency vector (Hz)
             k_axis: spatial wavenumber vector (1/m)
             fk_spectrum: 2D complex-valued Fourier spectrum, centered (shifted).
    """
    if data.ndim != 2:
        raise ValueError(f"'data' must be 2D (nx, nt); got {data.ndim}D")

    shape = pad_shape if pad_shape is not None else data.shape
    nx_out, nt_out = shape

    fk_spectrum = fftshift(fft2(data, s=shape))

    k_axis = fftshift(np.fft.fftfreq(nx_out, dx))
    f_axis = fftshift(np.fft.fftfreq(nt_out, dt))

    return f_axis, k_axis, fk_spectrum


def fk_inverse(
    fk_spectrum: np.ndarray,
    orig_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Transforms a complex FK spectrum back to the space-time (x, t) domain.

    :param fk_spectrum: 2D complex spectrum in (k, f) domain, assumed to be shifted.
    :param orig_shape: Optional tuple (nx, nt) to crop the result back to original size, 
                       undoing any padding applied during the forward transform.
    :return: Real-valued space-time wavefield.
    """
    if fk_spectrum.ndim != 2:
        raise ValueError(f"'fk_spectrum' must be 2D; got {fk_spectrum.ndim}D")

    data = ifft2(ifftshift(fk_spectrum))

    if orig_shape is not None:
        nx, nt = orig_shape
        data = data[:nx, :nt]

    return data.real

def fk_filter(
    data: np.ndarray,
    dt: float,
    dx: float,
    vmin: float,
    vmax: float,
    mode: Literal["eliminate", "extract"] = "eliminate",
    direction: Literal["both", "right", "left"] = "both",
    pad_factor: Tuple[int, int] = (1, 1),
    smooth: Literal["no", "gaussian", "uniform"] = "no",
    sigma: float = 1.0,
    uniform_size: int = 1,
) -> np.ndarray:
    """
    Applies a velocity-based fan filter in the Frequency-Wavenumber (FK) domain.

    **DSP Theory:**
    In the FK domain, coherent seismic waves appear as energy organized along linear 
    trajectories. The slope of these lines represents the apparent phase velocity (v = f/k). 
    FK filtering allows for the separation of signal from noise based on velocity 
    and propagation direction.

    **Applications in DAS:**
    - **Surface Wave Removal:** Eliminating slow-moving Scholte or Rayleigh waves 
      (low v) to reveal faster body waves.
    - **Directional Steering:** Extracting only waves traveling in one direction along 
      the fiber (e.g., separating upgoing vs. downgoing waves in a borehole).
    - **Noise Suppression:** Removing "ringing" interrogator noise which often maps 
      to specific k=0 or f=0 regions.

    :param data: Space-time array (nch, nt).
    :param dt: Temporal sampling (s).
    :param dx: Spatial sampling (m).
    :param vmin: Minimum velocity threshold (m/s).
    :param vmax: Maximum velocity threshold (m/s).
    :param mode: "eliminate" to mute energy within [vmin, vmax]; 
                 "extract" to keep only energy within [vmin, vmax].
    :param direction: "right" (k > 0), "left" (k < 0), or "both".
    :param pad_factor: Multiplier for padding (nx_in*pad_factor[0], nt_in*pad_factor[1]).
    :param smooth: Type of taper to apply to the mask edges to prevent Gibbs ringing.
    :param sigma: Standard deviation for Gaussian smoothing.
    :param uniform_size: Window size for uniform smoothing.
    :return: Filtered space-time wavefield.
    """
    nx_in, nt_in = data.shape
    nx_pad = int(nx_in * pad_factor[0])
    nt_pad = int(nt_in * pad_factor[1])

    freqs, ks, fk_data = fk_transform(data, dt, dx, pad_shape=(nx_pad, nt_pad))

    fk_data = np.flip(fk_data, axis=0)

    f_grid, k_grid = np.meshgrid(freqs, ks, indexing="xy")

    with np.errstate(divide="ignore", invalid="ignore"):
        v_grid = f_grid / k_grid
        v_grid[k_grid == 0] = np.inf

    mask = np.ones_like(fk_data.real)

    mask[(np.abs(v_grid) >= vmin) & (np.abs(v_grid) <= vmax)] = 0.0

    if direction == "right":
        mask[k_grid < 0] = 1.0
    elif direction == "left":
        mask[k_grid > 0] = 1.0

    if smooth == "gaussian":
        mask = gaussian_filter(mask, sigma=sigma)
    elif smooth == "uniform":
        mask = uniform_filter(mask, size=uniform_size)
    elif smooth != "no":
        raise ValueError(f"Invalid smooth mode: {smooth}")

    if mode == "eliminate":
        fk_data *= mask
    elif mode == "extract":
        fk_data *= (1.0 - mask)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    fk_data = np.flip(fk_data, axis=0)

    return fk_inverse(fk_data, orig_shape=(nx_in, nt_in))