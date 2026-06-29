"""
Shared fixtures for the das_grad test suite.

Run from the repo root with:  pytest  (or: make test)

The suite is numpy-only at its core; scipy (when present) upgrades the
synthetic Hankel kernel from asymptotic to exact. Tolerances are chosen to
pass in both situations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Compact synthetic geometry used across the suite (fast: < 1 s per VSG).
NCH = 256
DX = 8.0
FS = 50.0
NLAG = 1201          # 2M+1, M=600 -> 12 s of causal lag
C0 = 1000.0          # homogeneous reference velocity (m/s)
F_MIN = 1.5
F_MAX = 8.0


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260611)


def base_cfg(include_curvature: bool = True) -> dict:
    """Pipeline config matching the compact synthetic geometry."""
    return {
        "mask": {"v_inner": 600.0, "v_outer": 1700.0,
                 "t_inner": 2.0, "t_outer": -2.0,
                 "taper_sec": 0.5, "causal_only": True},
        "band": {"f_min": F_MIN, "f_max": F_MAX},
        "laplacian": {"mode": "fiber_1d",
                      "include_curvature": include_curvature,
                      "channel_taper_alpha": 0.2},
        "stack": {"r_exclude_m": 400.0},
        "post": {"median_size": 5, "edge_frac": 0.15, "r_min_m": 0.0,
                 "v_min": 50.0, "v_max": 10000.0},
    }


def homogeneous_vsg(vs_row: int = NCH // 2):
    """Single-mode VSG in a homogeneous c = C0 medium."""
    from src.synth import make_synthetic_vsg
    return make_synthetic_vsg(
        nch=NCH, dx=DX, vs_row=vs_row, fs=FS, nlag=NLAG,
        c_of_f=lambda f: np.full_like(np.asarray(f, dtype=float), C0),
        f0=4.0,
    )


def interior_mask(vsg, *, edge=0.15, r_min=400.0) -> np.ndarray:
    """Boolean channel mask: away from array edges and from the source."""
    idx = np.arange(vsg.nch)
    n_edge = int(edge * vsg.nch)
    return (idx >= n_edge) & (idx < vsg.nch - n_edge) & (vsg.r > r_min)
