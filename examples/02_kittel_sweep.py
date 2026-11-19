"""FMR frequency vs. in-plane bias field, fitted with the Kittel formula.

For each field value a small field pulse kicks a thin film, the average
m_x(t) is recorded, and its spectral peak is the FMR frequency.
Fitting the peaks with the Kittel formula should give back Msat.
"""

import matplotlib.pyplot as plt
import numpy as np

import mx3sw

Ms, Aex, alpha = 800e3, 13e-12, 0.01
dt, T = 10e-12, 5e-9
fields = np.linspace(0.02, 0.15, 8)
peaks = []

for B in fields:
    script = f"""
    SetGridSize(64, 64, 1)
    SetCellSize(5e-9, 5e-9, 5e-9)
    SetPBC(8, 8, 0)
    Msat = {Ms}
    Aex = {Aex}
    alpha = {alpha}
    m = Uniform(0, 1, 0)
    B_ext = vector(0, {B}, 0)
    Relax()
    B_ext = vector(1e-3 * sinc(2*pi*40e9*(t - 50e-12)), {B}, 0)
    TableAutoSave({dt})
    Run({T})
    """
    table, _ = mx3sw.run_mumax3(script, f"kittel_{B * 1e3:.0f}mT")
    peaks.append(mx3sw.peak_frequency(table["mx"], times=table["t"]))

peaks = np.array(peaks)
ms_fit, ms_err = mx3sw.fit_kittel_ms(fields, peaks)
print(f"Fitted Ms = {ms_fit / 1e3:.1f} ± {ms_err / 1e3:.1f} kA/m (input {Ms / 1e3:.0f})")

plt.plot(fields * 1e3, peaks / 1e9, "o", label="MuMax3")
plt.plot(fields * 1e3, mx3sw.kittel_inplane(fields, ms_fit) / 1e9, "-", label="Kittel fit")
plt.xlabel("B (mT)")
plt.ylabel("f (GHz)")
plt.legend()
plt.savefig("kittel.png", dpi=200)
