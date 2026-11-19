"""Spin-wave dispersion of a Permalloy waveguide (needs MuMax3 + a CUDA GPU).

A sinc field pulse in a narrow antenna region excites a broad band of spin
waves. The 2D FFT of m_x(t, x) gives f(k), compared with the exchange
dispersion. Absorbing (high-damping) regions at both ends stop reflections.
"""

import matplotlib.pyplot as plt
import numpy as np

import mx3sw

# --- parameters -----------------------------------------------------------
fmax = 20e9                 # highest excited frequency (Hz)
T = 10e-9                   # simulated time (s)
dt = 1 / (2 * fmax)         # sampling interval (Nyquist for fmax)
Nx, Ny = 1024, 32
sizeX, sizeY, sizeZ = 10e-6, 1e-6, 8e-9
dx = sizeX / Nx
Ms, Aex, alpha, By = 800e3, 13e-12, 0.01, 0.1

script = f"""
SetGridSize({Nx}, {Ny}, 1)
SetCellSize({dx}, {sizeY / Ny}, {sizeZ})
Msat = {Ms}
Aex = {Aex}
alpha = {alpha}

// absorbing ends and antenna
DefRegion(1, XRange(-Inf, -4.5e-6)); alpha.SetRegion(1, 1)
DefRegion(5, XRange(4.5e-6, Inf));   alpha.SetRegion(5, 1)
DefRegion(2, XRange(-4.0e-6, -3.9e-6))

m = Uniform(0, 1, 0)
B_ext = vector(0, {By}, 0)
Relax()

B_ext.SetRegion(2, vector(1e-3 * sinc(2*pi*{fmax}*(t - {T}/2)), {By}, 0))
AutoSave(CropY(m, {Ny // 2}, {Ny // 2 + 1}), {dt})
Run({T})
"""

_, outdir = mx3sw.run_mumax3(script, "py_dispersion")
_, m = mx3sw.read_ovf_series(outdir, "m_*.ovf")      # (time, 3, 1, 1, Nx)
mx = m[:, 0, 0, 0, Nx // 2:]                        # downstream half of the guide

k, f, P = mx3sw.dispersion_map(mx, dt, dx)
plt.pcolormesh(k / 1e6, f / 1e9, np.log10(P + 1e-12), shading="auto", cmap="inferno")
kk = np.linspace(k.min(), k.max(), 400)
plt.plot(kk / 1e6, mx3sw.exchange_dispersion(kk, By, Aex, Ms) / 1e9, "c--", lw=1, label="exchange theory")
plt.ylim(0, fmax / 1e9)
plt.xlabel("k (rad/µm)")
plt.ylabel("f (GHz)")
plt.legend()
plt.savefig("dispersion.png", dpi=200)
