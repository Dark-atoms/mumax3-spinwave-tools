"""Run MuMax3 scripts from Python."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .io import read_table

__all__ = ["run_mumax3"]


def run_mumax3(script: str, name: str, workdir=".", mumax3: str = "mumax3", verbose: bool = False):
    """Write ``script`` to ``<name>.mx3``, run MuMax3 on it, and return the table.

    Parameters
    ----------
    script : str
        MuMax3 input script.
    name : str
        Base name; output goes to ``<workdir>/<name>.out``.
    mumax3 : str
        Path to the MuMax3 executable (must be on PATH by default).

    Returns
    -------
    table : pandas.DataFrame or None
        Contents of ``table.txt`` if the script wrote one.
    outdir : pathlib.Path
        Output folder, to read OVF files with :func:`mx3sw.read_ovf_series`.
    """
    if shutil.which(mumax3) is None:
        raise FileNotFoundError(f"'{mumax3}' not found. Install MuMax3 and add it to PATH.")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    scriptfile = workdir / f"{name}.mx3"
    scriptfile.write_text(script, encoding="utf-8")

    p = subprocess.run([mumax3, str(scriptfile)], cwd=workdir, capture_output=True, text=True)
    if verbose or p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"MuMax3 exited with code {p.returncode}")

    outdir = workdir / f"{name}.out"
    table_path = outdir / "table.txt"
    table = read_table(table_path) if table_path.exists() else None
    return table, outdir
