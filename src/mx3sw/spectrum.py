"""Frequency spectra and dispersion maps of spin-wave signals."""

from __future__ import annotations

import numpy as np

__all__ = [
    "resample_uniform", "fft_spectrum", "peak_frequency", "dispersion_map", "wavelength_from_profile",
]


def resample_uniform(times, signal):
    """Linearly resample a signal onto an evenly spaced time grid.

    MuMax3's adaptive solver writes ``table.txt`` rows at slightly uneven
    times, which biases an FFT that assumes a fixed ``dt``.

    Returns ``(dt, resampled_signal)``.
    """
    times = np.asarray(times, dtype=float)
    grid = np.linspace(times[0], times[-1], times.size)
    return float(grid[1] - grid[0]), np.interp(grid, times, np.asarray(signal, dtype=float))


def fft_spectrum(signal, dt: float | None = None, window: str | None = "hann",
                 remove_mean: bool = True, times=None):
    """One-sided amplitude spectrum of a time signal (e.g. ``table["mx"]``).

    Parameters
    ----------
    signal : array-like
        Samples taken every ``dt`` seconds.
    dt : float, optional
        Sampling interval in seconds (the ``tableautosave``/``autosave`` period).
    times : array-like, optional
        Sample times (e.g. ``table["t"]``). Use this instead of ``dt`` for
        MuMax3 tables: the signal is resampled to an even grid first.
    window : {"hann", None}
        Window applied before the FFT to reduce spectral leakage.
    remove_mean : bool
        Subtract the mean first so the DC bin does not dominate.

    Returns
    -------
    freq : ndarray (Hz), amplitude : ndarray
    """
    if times is not None:
        dt, x = resample_uniform(times, signal)
    elif dt is None:
        raise ValueError("Give either dt or times")
    else:
        x = np.asarray(signal, dtype=float)
    if remove_mean:
        x = x - x.mean()
    if window == "hann":
        x = x * np.hanning(x.size)
    elif window is not None:
        raise ValueError("window must be 'hann' or None")
    amp = np.abs(np.fft.rfft(x)) / x.size
    freq = np.fft.rfftfreq(x.size, d=dt)
    return freq, amp


def peak_frequency(signal, dt: float | None = None, fmin: float = 0.0,
                   fmax: float | None = None, **kw) -> float:
    """Frequency (Hz) of the strongest spectral peak between ``fmin`` and ``fmax``.

    Accepts the same ``times=`` keyword as :func:`fft_spectrum`.
    """
    freq, amp = fft_spectrum(signal, dt, **kw)
    mask = freq > fmin
    if fmax is not None:
        mask &= freq <= fmax
    if not mask.any():
        raise ValueError("No frequency bins in the requested range")
    return float(freq[mask][np.argmax(amp[mask])])


def dispersion_map(m_tx, dt: float, dx: float, window: bool = True):
    """2D FFT of a space-time map m(t, x) to get the dispersion relation f(k).

    Parameters
    ----------
    m_tx : array (n_time, n_x)
        One magnetisation component sampled along a line of the waveguide.
    dt, dx : float
        Time step (s) and cell size along x (m).
    window : bool
        Apply a Hann window in both time and space.

    Returns
    -------
    k : ndarray (rad/m, sorted), f : ndarray (Hz, sorted), power : ndarray (n_time, n_x)
        A wave travelling in +x, m ~ cos(k x - 2 pi f t), appears at k > 0 and f > 0.
    """
    m = np.asarray(m_tx, dtype=float)
    m = m - m.mean(axis=0, keepdims=True)
    if window:
        m = m * np.outer(np.hanning(m.shape[0]), np.hanning(m.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(m))) ** 2
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(m.shape[1], d=dx))
    # numpy's fft2 puts exp(i(kx - wt)) at (+k, -f); flip the time axis so that
    # waves travelling in +x have positive k and positive f
    f = -np.fft.fftshift(np.fft.fftfreq(m.shape[0], d=dt))
    order = np.argsort(f)
    return k, f[order], power[order]


def wavelength_from_profile(x, profile, min_height: float = 0.0) -> float:
    """Average wavelength from the spacing of maxima in a 1D spin-wave profile.

    Returns ``nan`` if fewer than two peaks are found.
    """
    from scipy.signal import find_peaks

    x = np.asarray(x, dtype=float)
    peaks, _ = find_peaks(np.asarray(profile, dtype=float), height=min_height)
    if len(peaks) < 2:
        return float("nan")
    return float(np.mean(np.diff(x[peaks])))
