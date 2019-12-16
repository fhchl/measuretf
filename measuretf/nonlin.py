"""Non-linear tools."""

import numpy as np


def exponential_sweep_harmonic_delay(T, fs, N, f_start=None, f_end=None):
    """Delay of harmonic impulse response.

    From `Farina`_.

    Parameters
    ----------
    T : float
        Iength of impulse response.
    N : int
        Order of harmonic. First harmonic is fundamental.
    f_start, f_end : float or None, optional
        Start and stop frequencies of exponential sweep.

    Returns
    -------
    TYPE
        Description

    .. _Farina:
       A. Farina, “Simultaneous
       of impulse response and distortion
       with a swept-sine techniqueMinnaar, Pauli,” in Proc. AES 108th conv,
       Paris, France, 2000, pp. 1–15.

    """
    n_tap = int(np.round(T * fs))

    if f_start is None:
        f_start = fs / n_tap
    if f_end is None:
        f_end = fs / 2

    w_start = 2 * np.pi * f_start
    w_end = 2 * np.pi * f_end

    return T * np.log(N) / np.log(w_end / w_start)


def harmonic_spectrum(r, fs, order=10):
    """Energy of non-linear components of sweept impulse response.

    Parameters
    ----------
    r : ndarray
        Impulse response from sine sweept measurement.
    fs : int
        sample rate
    order : int, optional
        Number of harmonics.

    Returns
    -------
    ndarray, length order
        The energy of the nth order harmonics in the impules response.

    """
    n = r.size
    orders = np.arange(1, order + 2)
    T = n / fs

    # find max and circshift it to tap 0
    r = np.roll(r, -np.argmax(np.abs(r) ** 2))

    # delays of non-linear components
    dts = exponential_sweep_harmonic_delay(T, fs, orders)
    dns = np.round((T - dts) * fs).astype(int)
    dns[0] = n

    e = np.zeros(order)

    # fundamental
    n_start = int(round((dns[1] + dns[0]) / 2))
    n_end = int(round((dns[-1] / 2)))
    e[0] = np.sum(np.abs(r[n_start:]) ** 2)
    e[0] += np.sum(np.abs(r[:n_end]) ** 2)

    # higher order
    for i in orders[:-2]:
        n_start = int(round((dns[i + 1] + dns[i]) / 2))
        n_end = int(round((dns[i] + dns[i - 1]) / 2))
        e[i] = np.sum(np.abs(r[n_start:n_end]) ** 2)

    return e


def thd(r, fs, order=10):
    """Total Harmonic Distortion from swept sine measurement.

    Parameters
    ----------
    r : ndarray
        Impulse response from sine sweept measurement.
    fs : int
        sample rate
    order : int, optional
        Number of harmonics.

    Returns
    -------
    float
        THD.

    """
    e = harmonic_spectrum(r, fs, order=order)
    return np.sqrt(np.sum(e[1:]) / e[0])
