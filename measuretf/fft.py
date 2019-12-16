"""FFT tools."""

import numpy as np


def frequency_vector(n, fs, sided="single"):
    """Frequency values of filter with n taps sampled at fs up to Nyquist.

    Parameters
    ----------
    n : int
        Number of taps in FIR filter
    fs : int
        Sampling frequency in Hertz

    Returns
    -------
    (n // 2 + 1) ndarray
        Frequencies in Hz

    ALTERNATIVELY:
        np.abs(np.fft.fftfreq(n, 1)[:n // 2 + 1])

    """
    if sided == "single":
        f = np.arange(n // 2 + 1, dtype=float) * fs / n
    elif sided == "double":
        f = np.arange(n, dtype=float) * fs / n
    else:
        raise ValueError("Invalid value for sided.")
    return f


def time_vector(n, fs):
    """Time values of filter with n taps sampled at fs.

    Parameters
    ----------
    n : int
        number of taps in FIR filter
    fs : int
        sampling frequency in Hertz

    Returns
    -------
    (n) ndarray
        times in seconds

    """
    T = 1 / fs
    return np.arange(n, dtype=float) * T


def amplitude_spectrum(x, axis=-1, norm=True):
    """Convert time domain signal to single sided amplitude spectrum.

    Parameters
    ----------
    x : ndarray
        Real signal, which can be multidimensional (see axis).
    axis : int, optional
        Transformation is done along this axis. Default is -1 (last axis).
    norm: bool, optinal
        If True, normalize the response in frequency domain such that the
        amplitude of sinusoids is conserved.

    Returns
    -------
    ndarray
        Frequency response X with `X.shape[axis] == x.shape[axis] // 2 + 1`.
        The single sided spectrum.


    Notes
    -----
    Frequency spectrum is normalized for conservation of ampltiude.

    If len(x[axis]) is even, x[-1] contains the term representing both positive
    and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely
    real. If len(x[axis]) is odd, there is no term at fs/2; x[-1] contains the
    largest positive frequency (fs/2*(n-1)/n), and is complex in the general
    case.

    """
    # move time axis to front
    x = np.moveaxis(x, axis, 0)

    n = x.shape[0]

    X = np.fft.rfft(x, axis=0)

    if norm:
        X /= n

    # sum complex and real part
    if n % 2 == 0:
        # zero and nyquist element only appear once in complex spectrum
        X[1:-1] *= 2
    else:
        # there is no nyquist element
        X[1:] *= 2

    # and back again
    X = np.moveaxis(X, 0, axis)

    return X


def inverse_amplitude_spectrum(X, isEvenSampled=True, axis=-1, norm=True):
    """Convert single sided spectrum to time domain signal.

    Parameters
    ----------
    X : ndarray
        Complex frequency response, which can be mutlidimensional (see axis).
    n : int or None, optional
        Number of taps of resulting signal. If `None`, n is taken as
        len(X[axis]).
    axis : TYPE, optional
        Axis of X along which transformation is performed. Default is -1 (last
        axis).
    norm: bool, optinal
        Set to true if frequency response was normalized such that the
        amplitude of sinusoids is conserved.

    Returns
    -------
    ndarray
        Real signal.

    Notes
    -----
    Frequency spectrum is normalized for conservation of ampltiude.

    If n is even, X[-1] should contain the term representing both
    positive and negative Nyquist frequency (+fs/2 and -fs/2), and must
    also be purely real. If n is odd, there is no term at fs/2; X[-1]
    contains the largest positive frequency (fs/2*(n-1)/n), and is
    complex in the general case.

    """
    Xhalved = X.copy()

    # move frequency axis to first position
    Xhalved = np.moveaxis(Xhalved, axis, 0)

    if isEvenSampled:
        n = (Xhalved.shape[0] - 1) * 2
        Xhalved[1:-1] /= 2
    else:
        n = Xhalved.shape[0] * 2 - 1
        Xhalved[1:] /= 2

    # move frequency axis back
    Xhalved = np.moveaxis(Xhalved, 0, axis)

    x = np.fft.irfft(Xhalved, n=n, axis=axis)

    if norm:
        x *= n

    return x
