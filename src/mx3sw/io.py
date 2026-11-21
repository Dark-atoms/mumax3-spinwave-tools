"""Readers for MuMax3 output: table.txt and OVF (v1/v2, text/binary) files."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["read_table", "read_ovf", "read_ovf_series"]

# Check values defined by the OVF 2.0 specification
_CHECK4 = 1234567.0
_CHECK8 = 123456789012345.0


def read_table(path) -> pd.DataFrame:
    """Read a MuMax3 ``table.txt`` into a DataFrame.

    Column names keep the quantity and drop the unit, e.g.
    ``"# t (s)"`` -> ``"t"``, ``"m.region4x ()"`` -> ``"m.region4x"``.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().lstrip("#").strip()
    names = [n for n in re.split(r"\s*\([^)]*\)\s*", header) if n.strip()]
    names = [n.strip() for n in names]
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if len(names) == df.shape[1]:
        df.columns = names
    return df


def _parse_header(f):
    meta = {}
    while True:
        raw = f.readline()
        if not raw:
            raise ValueError("Reached end of file before the data block")
        line = raw.decode("latin-1").strip()
        if not line.startswith("#"):
            continue
        body = line.lstrip("#").strip()
        if ":" not in body:
            continue
        key, value = (s.strip() for s in body.split(":", 1))
        key = key.lower()
        if key == "begin" and value.lower().startswith("data"):
            meta["data"] = value.split()[1:]  # e.g. ["Binary", "4"] or ["Text"]
            return meta
        if key == "desc":
            label, _, time_text = value.partition(":")
            if label.strip().lower() in ("total simulation time", "time (s)"):
                try:
                    seconds = float(time_text.split()[0])
                except (ValueError, IndexError):
                    seconds = float("nan")
                meta["simulation_time"] = seconds if np.isfinite(seconds) else float("nan")
        meta[key] = value


def read_ovf(path, *, return_times: bool = False) -> np.ndarray | tuple[np.ndarray, float]:
    """Read one OVF file written by MuMax3.

    Supports OVF 2.0 (text, binary 4, binary 8) and OVF 1.0 text/binary.

    With ``return_times=True``, return ``(array, time_in_seconds)``. The time
    comes from MuMax3's ``Desc: Total simulation time: ... s`` header. Missing,
    malformed or non-finite times are represented by ``numpy.nan``.

    Returns
    -------
    numpy.ndarray
        Shape ``(valuedim, znodes, ynodes, xnodes)``, the same layout as
        ``mumax3-convert -numpy``. For magnetisation, index 0/1/2 = mx/my/mz.
    """
    path = Path(path)
    with path.open("rb") as f:
        first = f.readline().decode("latin-1")
        version = 2 if "2.0" in first else 1
        meta = _parse_header(f)

        nx, ny, nz = (int(meta[k]) for k in ("xnodes", "ynodes", "znodes"))
        dim = int(meta.get("valuedim", 3))
        n = nx * ny * nz * dim
        kind = meta["data"][0].lower()

        if kind == "text":
            # The data block ends at the "# End: Data Text" line; parse it in one go
            # with NumPy's C parser instead of splitting line by line in Python.
            block = f.read()
            end = block.find(b"#")
            if end != -1:
                block = block[:end]
            data = np.fromstring(block.decode("latin-1"), dtype=np.float64, sep=" ")[:n]
        elif kind == "binary":
            size = int(meta["data"][1])
            if size not in (4, 8):
                raise ValueError(f"Unsupported binary size: {size}")
            # OVF 2.0 is little-endian, OVF 1.0 is big-endian
            order = "<" if version == 2 else ">"
            dtype = np.dtype(f"{order}f{size}")
            check = np.frombuffer(f.read(size), dtype=dtype)[0]
            expected = _CHECK4 if size == 4 else _CHECK8
            if not np.isclose(check, expected):
                raise ValueError(f"Bad OVF check value {check!r} in {path}")
            data = np.frombuffer(f.read(n * size), dtype=dtype).astype(np.float64)
        else:
            raise ValueError(f"Unknown OVF data block: {meta['data']}")

    if data.size != n:
        raise ValueError(f"Expected {n} values, found {data.size} in {path}")
    # File order: component fastest, then x, y, z
    array = data.reshape(nz, ny, nx, dim).transpose(3, 0, 1, 2).copy()
    if return_times:
        return array, meta.get("simulation_time", float("nan"))
    return array


def read_ovf_series(
    folder, pattern: str = "m*.ovf", *, return_times: bool = False
) -> tuple[list[Path], np.ndarray] | tuple[list[Path], np.ndarray, np.ndarray]:
    """Read every OVF file in ``folder`` matching ``pattern`` (sorted by name).

    Returns the file list and an array of shape ``(time, valuedim, z, y, x)``.
    Files whose shape differs from the first one are skipped with a warning.
    With ``return_times=True``, also return a one-dimensional array of times in
    seconds, aligned with the retained files and frames (``numpy.nan`` when
    unavailable). Times are read from each file, not inferred from its name.
    """
    files = sorted(Path(folder).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {folder}")
    frames, used, times = [], [], []
    for p in files:
        arr, seconds = read_ovf(p, return_times=True)
        if frames and arr.shape != frames[0].shape:
            print(f"[mx3sw] skipping {p.name}: shape {arr.shape} != {frames[0].shape}")
            continue
        frames.append(arr)
        used.append(p)
        times.append(seconds)
    stack = np.stack(frames)
    if return_times:
        return used, stack, np.asarray(times)
    return used, stack
