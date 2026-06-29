"""
:module: src/vsg.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Virtual Source Gather (VSG) I/O for the das_grad pipeline.

das_grad consumes the VSGs produced by the companion das_ani repository
(https://github.com/Benz-Poobua/das_ani). The file contract is:

    <basename>_cc_<vs:03d>_<mode>.npy            (raw, per file)
    YYYYMMDD[_HHMMSS]_cc_<vs:03d>_<window>_<mode>.npy   (stacked)

    - float32 array of shape (nch, 2*M + 1)
    - row i  <-> receiver channel  first_chan + i  (absolute cable index)
    - column axis = correlation lag, np.arange(-M, M+1) / fs_proc seconds
    - <vs> = virtual-source row index (relative to first_chan)

The geometry that is NOT stored in the .npy (channel spacing dx, processing
sampling rate fs) comes from the das_grad YAML sidecar config -- mirroring
how das_ani itself carries dx/fs in its config rather than in the output.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from src.utils import PathLike

logger = logging.getLogger(__name__)

#: Matches "..._cc_080_v1.npy" and "..._cc_080_7d_v1.npy"; group(1) = VS index.
_VS_RE = re.compile(r"_cc_(\d+)")


@dataclass
class VSG:
    """
    One virtual source gather with its geometry.

    :param data: (nch, nlag) float array; rows are channels along the fiber.
    :param lag: (nlag,) lag axis in seconds, symmetric about zero.
    :param offset: (nch,) signed along-fiber offset of every channel from the
                   virtual source, in meters (negative = before the VS).
    :param vs_row: row index of the virtual source within ``data``.
    :param name: provenance label (usually the source filename).
    """
    data: np.ndarray
    lag: np.ndarray
    offset: np.ndarray
    vs_row: int
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        self.lag = np.asarray(self.lag, dtype=np.float64)
        self.offset = np.asarray(self.offset, dtype=np.float64)
        if self.data.ndim != 2:
            raise ValueError(f"VSG data must be 2D (nch, nlag); got {self.data.shape}")
        nch, nlag = self.data.shape
        if self.lag.shape != (nlag,):
            raise ValueError(f"lag axis length {self.lag.shape} != nlag {nlag}")
        if self.offset.shape != (nch,):
            raise ValueError(f"offset length {self.offset.shape} != nch {nch}")
        if not (0 <= self.vs_row < nch):
            raise ValueError(f"vs_row {self.vs_row} outside [0, {nch})")

    # ----- Convenience properties -----
    @property
    def nch(self) -> int:
        return int(self.data.shape[0])

    @property
    def nlag(self) -> int:
        return int(self.data.shape[1])

    @property
    def dt(self) -> float:
        return float(self.lag[1] - self.lag[0])

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    @property
    def dx(self) -> float:
        return float(abs(self.offset[1] - self.offset[0]))

    @property
    def r(self) -> np.ndarray:
        """Absolute source-receiver distance |offset| (meters)."""
        return np.abs(self.offset)


def parse_vs_index(filename: str) -> int:
    """Extract the virtual-source index from a das_ani VSG filename."""
    m = _VS_RE.search(Path(filename).name)
    if m is None:
        raise ValueError(f"Cannot parse VS index ('_cc_<n>') from: {filename}")
    return int(m.group(1))


def lag_axis(nlag: int, fs: float) -> np.ndarray:
    """
    Reconstruct the das_ani lag axis for a (nch, nlag) VSG: nlag = 2M+1,
    lag = np.arange(-M, M+1) / fs.
    """
    if nlag % 2 != 1:
        raise ValueError(f"VSG lag axis must have odd length (2M+1); got {nlag}")
    M = (nlag - 1) // 2
    return np.arange(-M, M + 1, dtype=np.float64) / float(fs)


def load_vsg(
    path: PathLike,
    *,
    fs: float,
    dx: float,
    vs_row: Optional[int] = None,
) -> VSG:
    """
    Load one das_ani VSG (.npy) and attach its geometry.

    :param path: Path to the ``*_cc_<vs>_*.npy`` file.
    :param fs: Processing sampling rate of the NCFs (Hz) -- das_ani's
               ``fs_proc = data.fs_raw / ingest.decimation``.
    :param dx: Channel spacing along the fiber (m) -- das_ani's ``data.dx``.
    :param vs_row: Virtual-source row index. Default: parsed from the
                   filename (the das_ani convention stores it there).
    :return: Populated :class:`VSG`.
    """
    p = Path(path).expanduser()
    obj = np.load(p, allow_pickle=False)

    if isinstance(obj, np.lib.npyio.NpzFile):
        # das_ani ncf_pre: a folded directional (s1/s2) gather saved as .npz
        # that carries its own geometry. It is a window CENTRED on the virtual
        # source, so we trust the stored `offset` (zero at the VS) and `lag`
        # (which may be one-sided/causal) rather than recomputing them.
        if "data" not in obj:
            raise ValueError(f"{p.name}: .npz missing 'data' (keys={list(obj.keys())}).")
        data = np.asarray(obj["data"], dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"{p.name}: expected 2D VSG; got shape {data.shape}")
        lag = (np.asarray(obj["lag"], dtype=np.float64)
               if "lag" in obj else lag_axis(data.shape[1], fs))
        if "offset" in obj:
            offset = np.asarray(obj["offset"], dtype=np.float64)
        else:
            r = parse_vs_index(p.name) if vs_row is None else vs_row
            offset = (np.arange(data.shape[0], dtype=np.float64) - float(r)) * float(dx)
        if vs_row is None:
            vs_row = int(np.argmin(np.abs(offset)))   # VS = the zero-offset channel
    else:
        # das_ani raw stack: a 2-D .npy over the full array; geometry is
        # recomputed from the virtual-source channel index in the filename.
        data = np.asarray(obj)
        if data.ndim != 2:
            raise ValueError(f"{p.name}: expected 2D VSG; got shape {data.shape}")
        if vs_row is None:
            vs_row = parse_vs_index(p.name)
        lag = lag_axis(data.shape[1], fs)
        offset = (np.arange(data.shape[0], dtype=np.float64) - float(vs_row)) * float(dx)

    nch = data.shape[0]
    if not (0 <= vs_row < nch):
        raise ValueError(f"{p.name}: vs_row={vs_row} outside [0, {nch})")
    return VSG(data=data, lag=lag, offset=offset, vs_row=int(vs_row), name=p.name)


def discover_vsgs(root: PathLike, pattern: str = "*_cc_*.npy") -> List[Path]:
    """
    Recursively collect VSG files under ``root`` matching ``pattern``,
    sorted by name. Raises if none are found (fail fast on a wrong path).
    """
    root = Path(root).expanduser()
    files = sorted(p for p in root.rglob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No VSG files matching {pattern!r} under {root}")
    logger.info("Discovered %d VSGs under %s", len(files), root)
    return files


def common_positions(vsgs: Sequence[VSG], dx: float, first_chan: int = 0) -> np.ndarray:
    """
    Absolute along-fiber channel positions (m) shared by a set of VSGs.
    All VSGs from one das_ani run share the same channel rows, so the map
    coordinate of the stacked sloth is simply row * dx (offset by
    ``first_chan`` for absolute cable positions).
    """
    nch = {v.nch for v in vsgs}
    if len(nch) != 1:
        raise ValueError(f"VSGs disagree on channel count: {sorted(nch)}")
    n = nch.pop()
    return (np.arange(n, dtype=np.float64) + float(first_chan)) * float(dx)
