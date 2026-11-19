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
        meta[key] = value


def read_ovf(path) -> np.ndarray:
    """Read one OVF file written by MuMax3.

    Supports OVF 2.0 (text, binary 4, binary 8) and OVF 1.0 text/binary.

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
            values = []
            for raw in f:
                line = raw.decode("latin-1").strip()
                if not line or line.startswith("#"):
                    continue
                values.extend(float(v) for v in line.split())
                if len(values) >= n:
                    break
            data = np.asarray(values[:n], dtype=np.float64)
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
    return data.reshape(nz, ny, nx, dim).transpose(3, 0, 1, 2).copy()


def read_ovf_series(folder, pattern: str = "m*.ovf") -> tuple[list[Path], np.ndarray]:
    """Read every OVF file in ``folder`` matching ``pattern`` (sorted by name).

    Returns the file list and an array of shape ``(time, valuedim, z, y, x)``.
    Files whose shape differs from the first one are skipped with a warning.
    """
    files = sorted(Path(folder).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {folder}")
    frames, used = [], []
    for p in files:
        arr = read_ovf(p)
        if frames and arr.shape != frames[0].shape:
            print(f"[mx3sw] skipping {p.name}: shape {arr.shape} != {frames[0].shape}")
            continue
        frames.append(arr)
        used.append(p)
    return used, np.stack(frames)
