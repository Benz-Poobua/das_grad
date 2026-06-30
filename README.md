# DAS-GRAD: Interferometric Frequency-Domain Gradiometry for Distributed Acoustic Sensing

## Overview

`das_grad` estimates **local, frequency-dependent surface-wave phase velocities** from DAS virtual source gathers (VSGs) by **interferometric frequency-domain gradiometry (I-FDG)** — the pointwise inversion of the 2-D Helmholtz equation introduced by Davis et al. (2026) for OBN data, here adapted to straight-fiber DAS geometry.

It is the downstream companion of [`das_ani`](https://github.com/Benz-Poobua/das_ani):

```text
das_ani  : continuous DAS -> preprocess -> short-lag CC (Zhang 2026) -> stack -> VSGs
das_grad : VSGs -> torus mask -> I-FDG -> local sloth s²(x, f) -> phase velocity V(x, f)
                                                       |
                       (future) 1-D inversion per x -> Vs(x, z) -> E-FWI starting model
```

Whereas the das_ani dispersion route (f–v panels → picks → inversion) yields **array-averaged** dispersion, gradiometry solves the Helmholtz relation **pointwise** — no dispersion picking, no two-station far-field criterion, a velocity at every channel — and, because I-FDG stacks over **virtual sources** rather than frequencies, the result keeps its frequency axis and with it depth sensitivity.

---

## Method in five equations

A single surface-wave mode obeys the 2-D scalar wave equation with sloth (squared slowness) $s^2 = 1/c^2$ (de Ridder & Curtis 2017):

$$\big[\nabla^2 - s^2(\mathbf{x})\,\partial_t^2\big]\,\tilde u(\mathbf{x},t) = 0.$$

**TDG** (time domain, least squares over lag; Davis et al. eq. 3):

$$\widehat{s^2}(\mathbf{x}) = \frac{\sum_t \partial_t^2\tilde u \, \nabla^2 \tilde u}{\sum_t \big|\partial_t^2\tilde u\big|^2}.$$

**FDG** (per frequency; Davis et al. eq. 5):

$$\widehat{s^2}(\mathbf{x},\omega) = \frac{\widetilde U^*\,\nabla^2 \widetilde U}{(i\omega)^2\,\big|\widetilde U\big|^2}.$$

**I-FDG** applies (3) to a VSG $\widetilde V(\mathbf{x},\mathbf{x}_s,\omega_\tau)$ — legitimate because a VSG is, mode by mode, a 2-D Helmholtz field in the receiver coordinate (Green's-function retrieval) — and stacks over virtual sources (Davis et al. eqs. 9–10):

$$\widehat{s^2}(\mathbf{x},\omega_\tau) = \frac{1}{N_s}\sum_{\mathbf{x}_s} \frac{\widetilde V^*\,\nabla^2 \widetilde V}{(i\omega_\tau)^2\big|\widetilde V\big|^2}, \qquad \widehat{V}(\mathbf{x},\omega_\tau) = \big[\widehat{s^2}\big]^{-1/2}.$$

### The DAS twist: the fiber Laplacian

Davis et al. evaluate $\nabla^2$ with a 2-D pseudospectral operator on a regular OBN grid. A single straight fiber only samples $\partial^2/\partial \ell^2$ — but for **in-line virtual sources** (the das_ani geometry: source and receivers on the same cable) the single-mode field is axisymmetric about the source, $U=U(r)$, $r=|\ell-\ell_s|$, so the full horizontal Laplacian along the fiber is exactly the cylindrical form

$$\nabla^2 U = \frac{\partial^2 U}{\partial r^2} + \frac{1}{r}\frac{\partial U}{\partial r}.$$

`das_grad` implements both: `laplacian.mode: fiber_1d` (with the $(1/r)\,\partial_r$ curvature term, recommended ON) and the Davis-faithful `grid_2d` for gridded layouts. Both are pseudospectral — no finite-difference stencil bias.

---

## Installation

```bash
python -m venv das_grad
source das_grad/bin/activate
pip install --upgrade pip
pip install -e .            # numpy, scipy, pyyaml, tqdm, pandas -- no torch
pip install -e ".[dev]"     # + pytest, ruff, black, mypy
pytest                      # run the test suite (~30 s)
```

The package is deliberately light: the entire pipeline is FFTs and elementwise math (Davis et al. processed 49 OBN VSGs in < 8 min on one CPU node; the fiber case is lighter). The heavy compute — VSG construction — lives in `das_ani`.

---

## Repository Structure
```text
.
├── README.md
├── pyproject.toml
├── sherlock_setup.sh            # HPC module environment (CPU)
│
├── data/
│   ├── ncf_pre/                 # preprocessed stacked VSGs (folded)
│   └── ncf_torus/               # torus-maked stacked VSGs
│
├── src/
│   ├── utils.py                 # config/timing + math helpers (copied from das_ani)
│   ├── ncf.py                   # das_ani VSG script legacy
│   ├── mask.py                  # torus (moveout-annulus) masking
│   ├── laplacian.py             # pseudospectral fiber_1d / grid_2d operators
│   ├── gradiometry.py           # TDG / FDG / I-FDG solvers + VSG stacking
│   └── post.py                  # quality mask, median filter, product export
│
└── tests/                       # pytest suite
```

---

## Input: das_ani VSGs

`das_grad` reads das_ani outputs

```text
<basename>_cc_<vs:03d>_<mode>.npy            raw VSGs (per file, per source)
YYYYMMDD[_HHMMSS]_cc_<vs:03d>_<window>_<mode>.npy   stacks (recommended input)
```

float32, shape `(nch, 2M+1)`; row *i* ↔ channel `first_chan + i`; lag axis `arange(-M, M+1)/fs_proc`; `<vs>` is the virtual-source row. The geometry not stored in the `.npy` (`fs_proc`, `dx`, `first_chan`) is supplied by the `data:` block of the das_grad config and **must match the das_ani run** that produced the files.

---

## Assumptions & validity (read before interpreting)

- **Single mode**: enforce with the band limits + torus mask; check a das_ani dispersion panel for guided-P / higher-mode contamination.
- **Smooth medium**: the sloth field interpretation is adiabatic/JWKB; strong scatterers (sharp lateral contrasts ≲ λ) enter the Mie regime (cf. the salt-body discussion in Davis et al.).
- **Regular sampling**: the pseudospectral Laplacian requires uniform dx and no spatial aliasing in the band.
- **In-line sources** (fiber_1d): the cylindrical-Laplacian identity assumes the VS lies on the fiber — exactly what das_ani produces. Strongly curved fiber sections violate it; use `grid_2d` on gridded layouts instead.
- **Aperture limit**: $f_{\min} \gtrsim 3c/L$; below it the spatial curvature is not resolvable.
- **DAS measures strain(-rate), not displacement**: a uniform axial response rescales $\widetilde V$ by an $\mathbf{x}$-independent factor that cancels in the sloth ratio; laterally varying coupling/site response does not — treat strong coupling variations as a quality issue (cf. Lin et al. 2012).

---

## Citation

If you use this codebase, please cite:

> Davis, D., Shragge, J., de Ridder, S., Girard, A. J. & Pandey, A. (2026).
> *Interferometric frequency-domain gradiometry.* **Geophys. J. Int.**, 245(3), 1–16.
> <https://doi.org/10.1093/gji/ggag146>

> de Ridder, S. A. L. & Curtis, A. (2017). *Seismic gradiometry using
> ambient seismic noise in an anisotropic Earth.* **Geophys. J. Int.**, 209(2), 1168–1179.

and for the VSG construction engine:

> Zhang, W.-Q. (2026). *Accelerating cross-correlation for long sequences
> with short lag constraints.* **Digital Signal Processing**, 168, 105509.

> Poobua, S., Li, H., & Biondi, B. L. *Minimum-Effort DAS
> Cross-Correlation:* SEP Report 199, Stanford University.

---
## License

This project is licensed under the MIT License. See the `LICENSE` file for full text.