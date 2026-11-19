"""Analytical formulas for checking simulations against theory."""

from __future__ import annotations

import numpy as np

__all__ = ["MU0", "GAMMA", "kittel_inplane", "exchange_dispersion", "fit_kittel_ms"]

MU0 = 4e-7 * np.pi  # vacuum permeability (T m / A)
GAMMA = 1.7595e11   # electron gyromagnetic ratio (rad / s / T), MuMax3 default


def kittel_inplane(B, Ms: float, gamma: float = GAMMA):
    """Kittel FMR frequency (Hz) of a thin film magnetised in-plane.

    f = gamma / (2 pi) * sqrt(B (B + mu0 Ms)),  B in tesla, Ms in A/m.
    """
    B = np.asarray(B, dtype=float)
    return gamma / (2 * np.pi) * np.sqrt(B * (B + MU0 * Ms))


def exchange_dispersion(k, B: float, Aex: float, Ms: float, gamma: float = GAMMA):
    """Exchange spin-wave dispersion (Hz), dipolar terms neglected.

    f(k) = gamma / (2 pi) * (B + 2 Aex k^2 / Ms),  k in rad/m.
    """
    k = np.asarray(k, dtype=float)
    return gamma / (2 * np.pi) * (B + 2 * Aex * k**2 / Ms)


def fit_kittel_ms(B, f, gamma: float = GAMMA, Ms_guess: float = 8e5):
    """Fit Ms (A/m) to simulated FMR frequencies with the in-plane Kittel formula.

    Returns ``(Ms, Ms_std)``.
    """
    from scipy.optimize import curve_fit

    B = np.asarray(B, dtype=float)
    f = np.asarray(f, dtype=float)
    popt, pcov = curve_fit(lambda b, ms: kittel_inplane(b, ms, gamma), B, f, p0=[Ms_guess])
    return float(popt[0]), float(np.sqrt(pcov[0, 0]))
