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

$$\big[\nabla^2 - s^2(\mathbf{x})\,\partial_t^2\big]\,\tilde u(\mathbf{x},t) = 0. \tag{1}$$

**TDG** (time domain, least squares over lag; Davis et al. eq. 3):

$$\widehat{s^2}(\mathbf{x}) = \frac{\sum_t \partial_t^2\tilde u \, \nabla^2 \tilde u}{\sum_t \big|\partial_t^2\tilde u\big|^2}. \tag{2}$$

**FDG** (per frequency; Davis et al. eq. 5):

$$\widehat{s^2}(\mathbf{x},\omega) = \frac{\widetilde U^*\,\nabla^2 \widetilde U}{(i\omega)^2\,\big|\widetilde U\big|^2}. \tag{3}$$

**I-FDG** applies (3) to a VSG $\widetilde V(\mathbf{x},\mathbf{x}_s,\omega_\tau)$ — legitimate because a VSG is, mode by mode, a 2-D Helmholtz field in the receiver coordinate (Green's-function retrieval) — and stacks over virtual sources (Davis et al. eqs. 9–10):

$$\widehat{s^2}(\mathbf{x},\omega_\tau) = \frac{1}{N_s}\sum_{\mathbf{x}_s} \frac{\widetilde V^*\,\nabla^2 \widetilde V}{(i\omega_\tau)^2\big|\widetilde V\big|^2}, \qquad \widehat{V}(\mathbf{x},\omega_\tau) = \big[\widehat{s^2}\big]^{-1/2}. \tag{4,5}$$

### The DAS twist: the fiber Laplacian

Davis et al. evaluate $\nabla^2$ with a 2-D pseudospectral operator on a regular OBN grid. A single straight fiber only samples $\partial^2/\partial \ell^2$ — but for **in-line virtual sources** (the das_ani geometry: source and receivers on the same cable) the single-mode field is axisymmetric about the source, $U=U(r)$, $r=|\ell-\ell_s|$, so the full horizontal Laplacian along the fiber is exactly the cylindrical form

$$\nabla^2 U = \frac{\partial^2 U}{\partial r^2} + \frac{1}{r}\frac{\partial U}{\partial r}.$$

`das_grad` implements both: `laplacian.mode: fiber_1d` (with the $(1/r)\,\partial_r$ curvature term, recommended ON) and the Davis-faithful `grid_2d` for gridded layouts. Both are pseudospectral — no finite-difference stencil bias.

---

## Verified accuracy (synthetic benchmark)

`src/eval.py` builds analytic single-mode VSGs ($\widetilde V = A(\omega)\,H_0(k(\omega) r)$, numpy-FFT sign convention) with a **known** dispersion curve $c(f)$, writes them in the das_ani file convention, runs the full production pipeline, and measures recovery. Representative result (9 VSGs, 400 ch × 8 m, 1–8 Hz, no noise):

| f (Hz) | 1.0 | 1.9 | 2.8 | 3.7 | 4.5 | 5.4 | 6.3 | 7.2 |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|
| c true (m/s) | 1040 | 955 | 900 | 864 | 842 | 827 | 817 | 811 |
| c recovered  | 1046 | 956 | 900 | 865 | 842 | 827 | 817 | 811 |
| rel. err (%) | 0.59 | 0.18 | 0.03 | 0.02 | 0.02 | 0.01 | 0.01 | 0.02 |

Pixel-level: median 0.10 %, p95 4 % (curvature ON). Two implementation details matter — both are asserted by the test suite:

1. **Near-source exclusion** (`stack.r_exclude_m`): each VSG is near-field within ~a wavelength of its own source; excluding those channels per VSG in the eq.-10 stack (they stay covered by the other sources) improves the stack by an **order of magnitude**.
2. **Spectral time derivative** in TDG: the textbook 2nd-order finite difference biases the sloth by $(\omega\,\Delta t)^2/12$ (≈ 8 % at 8 Hz @ 50 Hz); the FFT derivative removes it.

The lowest usable frequency is **aperture-limited** ($\lambda \lesssim$ aperture/3, i.e. $f_{\min}\gtrsim 3c/L$), the highest by single-mode separation — read both off a das_ani dispersion panel before running.

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
├── Makefile
├── sherlock_setup.sh            # HPC module environment (CPU)
│
├── configs/
│   └── urban_grad.yaml          # I-FDG parameters for the urban deployment
│
├── slurm/
│   ├── run_grad_urban.slurm     # production I-FDG on das_ani VSGs
│   └── run_eval_synth.slurm     # synthetic-recovery benchmark
│
├── data/
│   ├── grad/                    # products: grad_vsch_<tag>.npz
│   └── benchmarks/              # synthetic benchmark outputs
│
├── src/
│   ├── utils.py                 # config/timing helpers (copied from das_ani)
│   ├── vsg.py                   # das_ani VSG file contract + geometry
│   ├── mask.py                  # torus (moveout-annulus) masking
│   ├── laplacian.py             # pseudospectral fiber_1d / grid_2d operators
│   ├── gradiometry.py           # TDG / FDG / I-FDG solvers + VSG stacking
│   ├── post.py                  # quality mask, median filter, product export
│   ├── synth.py                 # analytic Helmholtz VSGs (known c(f))
│   ├── grad.py                  # config-driven workflow driver (CLI)
│   └── eval.py                  # synthetic-recovery benchmark (CLI)
│
└── tests/                       # pytest suite
```

---

## Input: das_ani VSGs

`das_grad` reads das_ani outputs directly — no converter step:

```text
<basename>_cc_<vs:03d>_<mode>.npy            raw VSGs (per file, per source)
YYYYMMDD[_HHMMSS]_cc_<vs:03d>_<window>_<mode>.npy   stacks (recommended input)
```

float32, shape `(nch, 2M+1)`; row *i* ↔ channel `first_chan + i`; lag axis `arange(-M, M+1)/fs_proc`; `<vs>` is the virtual-source row. The geometry not stored in the `.npy` (`fs_proc`, `dx`, `first_chan`) is supplied by the `data:` block of the das_grad config and **must match the das_ani run** that produced the files.

---

## Workflow

```bash
make grad      # python -m src.grad --config configs/urban_grad.yaml --verbose
make eval      # python -m src.eval --outdir data/benchmarks/synth
make test      # pytest
```

### Config reference (`configs/urban_grad.yaml`)

| Block | Keys | Purpose |
|-------|------|---------|
| `paths` | `vsg_root`, `vsg_pattern`, `output_root` | where VSGs live / products go |
| `data` | `fs_proc`, `dx`, `first_chan` | geometry of the das_ani run |
| `mask` | `v_inner`, `v_outer`, `t_inner`, `t_outer`, `taper_sec`, `causal_only` | torus filter: keep `tau ∈ [r/v_outer + t_outer, r/v_inner + t_inner]` per channel (Davis et al. Sec. 3.2) |
| `band` | `f_min`, `f_max` | usable single-mode band (from a dispersion panel) |
| `laplacian` | `mode`, `include_curvature`, `channel_taper_alpha` | `fiber_1d` (+ cylindrical curvature, recommended) or `grid_2d` |
| `stack` | `r_exclude_m` | per-VSG near-source exclusion radius (~1–2 λ at band center) |
| `post` | `median_size`, `edge_frac`, `r_min_m`, `v_min`, `v_max` | quality masking + NaN-aware median filter |
| `output` | `tag` | product name `grad_vsch_<tag>.npz` |

### Output product

`grad_vsch_<tag>.npz` with keys: `s2` (complex64 `(nch, nfreq)` stacked sloth — imaginary part ≈ transport residual, a quality diagnostic), `vel` (float32 `(nch, nfreq)` phase velocity, NaN where invalid), `freqs` (Hz), `positions` (absolute along-fiber meters), `valid` (channel mask), `meta_json` (config echo + contributing VSG list).

The `(x, f)` velocity cube is the direct input for per-channel 1-D surface-wave inversion (e.g. via disba/evodcinv, already dependencies of das_ani) toward a pseudo-3-D $V_S(x, z)$ — the long-wavelength E-FWI starting model that motivates the whole chain (Davis et al. 2026, Sec. 6.4).

---

## Running on HPC (SLURM)

```bash
sbatch slurm/run_grad_urban.slurm     # production I-FDG (CPU)
sbatch slurm/run_eval_synth.slurm     # synthetic benchmark
```

Both scripts `cd` to the repo root and `source sherlock_setup.sh` (python + scipy/pandas modules; no GPU required). Adjust the `cd` path and partition to your cluster.

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

> Poobua, S., Li, H., & Biondi, B. L. *Minimum-Effort DAS Cross-Correlation.*
> Stanford Exploration Project report **SEP-199**.

---
## License

This project is licensed under the MIT License. See the `LICENSE` file for full text.
