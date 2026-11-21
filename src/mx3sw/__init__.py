"""mx3sw: small helpers for spin-wave analysis of MuMax3 simulations."""

from .io import read_ovf, read_ovf_series, read_table
from .run import run_mumax3
from .spectrum import (dispersion_map, fft_spectrum, peak_frequency, resample_uniform,
                       wavelength_from_profile)
from .theory import GAMMA, MU0, exchange_dispersion, fit_kittel_ms, kittel_inplane

__version__ = "0.3.0"

__all__ = [
    "read_ovf", "read_ovf_series", "read_table", "run_mumax3",
    "resample_uniform", "fft_spectrum", "peak_frequency", "dispersion_map", "wavelength_from_profile",
    "kittel_inplane", "exchange_dispersion", "fit_kittel_ms", "GAMMA", "MU0",
]
