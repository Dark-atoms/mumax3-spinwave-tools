import numpy as np
import pytest

import mx3sw


def _write_ovf2(path, arr, mode="text"):
    """Write a (3, nz, ny, nx) array as a minimal OVF 2.0 file."""
    dim, nz, ny, nx = arr.shape
    header = (
        "# OOMMF OVF 2.0\n# Segment count: 1\n# Begin: Segment\n# Begin: Header\n"
        f"# valuedim: {dim}\n# xnodes: {nx}\n# ynodes: {ny}\n# znodes: {nz}\n"
        "# End: Header\n"
    )
    flat = arr.transpose(1, 2, 3, 0).reshape(-1, dim)
    with open(path, "wb") as f:
        f.write(header.encode())
        if mode == "text":
            f.write(b"# Begin: Data Text\n")
            for row in flat:
                f.write((" ".join(f"{v:.9g}" for v in row) + "\n").encode())
            f.write(b"# End: Data Text\n")
        else:
            size = 4 if mode == "b4" else 8
            check = 1234567.0 if size == 4 else 123456789012345.0
            dt = np.dtype(f"<f{size}")
            f.write(f"# Begin: Data Binary {size}\n".encode())
            f.write(np.array([check], dtype=dt).tobytes())
            f.write(flat.astype(dt).tobytes())
            f.write(f"\n# End: Data Binary {size}\n".encode())
        f.write(b"# End: Segment\n")


@pytest.mark.parametrize("mode", ["text", "b4", "b8"])
def test_read_ovf_roundtrip(tmp_path, mode):
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(3, 2, 4, 5)).astype(np.float32).astype(float)
    p = tmp_path / f"m_{mode}.ovf"
    _write_ovf2(p, arr, mode)
    out = mx3sw.read_ovf(p)
    assert out.shape == arr.shape
    assert np.allclose(out, arr, atol=1e-6)


def test_read_ovf_series(tmp_path):
    for i in range(3):
        _write_ovf2(tmp_path / f"m{i:06d}.ovf", np.full((3, 1, 2, 2), float(i)))
    files, stack = mx3sw.read_ovf_series(tmp_path)
    assert len(files) == 3 and stack.shape == (3, 3, 1, 2, 2)
    assert stack[2].max() == 2.0


def test_read_table(tmp_path):
    p = tmp_path / "table.txt"
    p.write_text("# t (s)\tmx ()\tm.region4x ()\tB_exty (T)\n0\t0.1\t0.2\t0.1\n1e-11\t0.3\t0.4\t0.1\n")
    t = mx3sw.read_table(p)
    assert list(t.columns) == ["t", "mx", "m.region4x", "B_exty"]
    assert t["mx"].iloc[1] == pytest.approx(0.3)


def test_peak_frequency():
    dt = 25e-12
    t = np.arange(4000) * dt
    sig = np.sin(2 * np.pi * 7.5e9 * t) + 0.2 * np.sin(2 * np.pi * 3e9 * t)
    assert mx3sw.peak_frequency(sig, dt) == pytest.approx(7.5e9, rel=0.01)


def test_peak_frequency_uneven_times():
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.uniform(20e-12, 30e-12, 800))
    sig = np.sin(2 * np.pi * 9.25e9 * t)
    assert mx3sw.peak_frequency(sig, times=t) == pytest.approx(9.25e9, rel=0.01)


def test_dispersion_map_finds_wave():
    dt, dx = 25e-12, 10e-9
    t = np.arange(512)[:, None] * dt
    x = np.arange(256)[None, :] * dx
    k0, f0 = 2 * np.pi / (16 * dx), 5e9
    m = np.cos(k0 * x - 2 * np.pi * f0 * t)
    k, f, P = mx3sw.dispersion_map(m, dt, dx)
    i, j = np.unravel_index(np.argmax(P), P.shape)
    assert abs(abs(f[i]) - f0) / f0 < 0.02
    assert abs(abs(k[j]) - k0) / k0 < 0.02


def test_dispersion_map_sign_convention():
    # a wave travelling in +x must appear at k > 0 and f > 0
    dt, dx = 25e-12, 10e-9
    t = np.arange(256)[:, None] * dt
    x = np.arange(128)[None, :] * dx
    m = np.cos(2 * np.pi / (16 * dx) * x - 2 * np.pi * 5e9 * t)
    k, f, P = mx3sw.dispersion_map(m, dt, dx)
    assert np.all(np.diff(f) > 0)
    # real signals also have the mirror peak at (-k, -f); look at the k > 0 half
    Pk = P[:, k > 0]
    i, j = np.unravel_index(np.argmax(Pk), Pk.shape)
    assert f[i] > 0
    assert Pk[f < 0].max() < 1e-6 * Pk.max()


def test_wavelength_from_profile():
    x = np.linspace(0, 10e-6, 2001)
    assert mx3sw.wavelength_from_profile(x, np.cos(2 * np.pi * x / 1.5e-6)) == pytest.approx(1.5e-6, rel=0.01)


def test_kittel_fit_recovers_ms():
    B = np.linspace(0.01, 0.15, 15)
    f = mx3sw.kittel_inplane(B, 8e5)
    ms, _ = mx3sw.fit_kittel_ms(B, f, Ms_guess=5e5)
    assert ms == pytest.approx(8e5, rel=1e-4)


def test_exchange_dispersion_k0_is_larmor():
    assert mx3sw.exchange_dispersion(0.0, 0.1, 13e-12, 8e5) == pytest.approx(mx3sw.GAMMA * 0.1 / (2 * np.pi))
