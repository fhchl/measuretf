"""Filtering and windowing."""

import matplotlib.pyplot as plt
import numpy as np
from measuretf.fft import frequency_vector, time_vector
from measuretf.utils import find_nearest
from response import Response
from scipy.signal import convolve, get_window, hann, resample_poly


def lowpass_by_frequency_domain_window(fs, x, fstart, fstop, axis=-1):
    """Lowpass time signal with half hann between fstart and fstop.

    Parameters
    ----------
    fs : int
        Sampling frequency
    x : array like
        Real time domain signal
    fstart : float
        Starting frequency of window
    fstop : TYPE
        Ending frequency of window
    axis : TYPE, optional
        signal is assumed to be along x[axis]

    Returns
    -------
    TYPE
        Description

    Raises
    ------
    ValueError
        If fstart or fstop don't fit in the frequency range.

    """
    n = x.shape[axis]
    f = frequency_vector(n, fs)

    # corresponding indices
    _, start = find_nearest(f, fstart)
    _, stop = find_nearest(f, fstop)

    if not (start and stop):
        raise ValueError("Frequencies are to large.")

    # the window
    window_width = stop - start
    windowed_samples = np.arange(start, stop)

    symmetric_window = hann(2 * window_width)
    half_window = symmetric_window[window_width:]

    # frequency domain
    X_windowed = np.fft.rfft(x, axis=axis)
    X_windowed = np.moveaxis(X_windowed, axis, 0)
    X_windowed[windowed_samples] = (
        X_windowed[windowed_samples].T * half_window.T
    ).T  # broadcasting
    X_windowed[stop:] = 0
    X_windowed = np.moveaxis(X_windowed, 0, axis)

    return np.fft.irfft(X_windowed, n=n)


def sample_window(n, startwindow, stopwindow, window="hann"):
    """Create a sample domain window."""
    swindow = np.ones(n)

    if startwindow is not None:
        length = startwindow[1] - startwindow[0]
        w = get_window(window, 2 * length, fftbins=False)[:length]
        swindow[: startwindow[0]] = 0
        swindow[startwindow[0] : startwindow[1]] = w

    if stopwindow is not None:
        # stop window
        length = stopwindow[1] - stopwindow[0]
        w = get_window(window, 2 * length, fftbins=False)[length:]
        swindow[stopwindow[0] + 1 : stopwindow[1] + 1] = w
        swindow[stopwindow[1] + 1 :] = 0

    return swindow


def time_window(fs, n, startwindow, stopwindow, window="hann"):
    """Create a time domain window."""
    times = time_vector(n, fs)

    if startwindow is not None:
        startwindow_n = [find_nearest(times, t)[1] for t in startwindow]
    else:
        startwindow_n = None
    if stopwindow:
        stopwindow_n = [find_nearest(times, t)[1] for t in stopwindow]
    else:
        stopwindow_n = None

    twindow = sample_window(n, startwindow_n, stopwindow_n, window=window)

    return twindow


def freq_window(fs, n, startwindow, stopwindow, window="hann"):
    """Create a frequency domain window."""
    freqs = frequency_vector(n, fs)
    nf = len(freqs)

    if startwindow is not None:
        startwindow_n = [find_nearest(freqs, f)[1] for f in startwindow]
    else:
        startwindow_n = None

    if stopwindow is not None:
        stopwindow_n = [find_nearest(freqs, f)[1] for f in stopwindow]
    else:
        stopwindow_n = None

    fwindow = sample_window(nf, startwindow_n, stopwindow_n, window=window)

    return fwindow


def mutliconvolve(sound, h, plot=False):
    """Convolve signel channel sound with multichannel impulse response."""
    nch = h.shape[-1]
    nfilt = h.shape[0]
    nsound = sound.shape[0]

    # convolve sounds
    convsound = np.zeros((nfilt + nsound - 1, nch + 1))
    for i in range(nch):
        convsound[:, i] = convolve(sound, h[:, i])

    # convolve reference with unit filter
    h_unit = np.zeros(h.shape[0])
    h_unit[0] = 1
    convsound[:, -1] = convolve(sound, h_unit)

    # normalize
    convsound *= 1 / np.max(np.abs(convsound)) * 0.95

    if plot:
        fig, axes = plt.subplots(nrows=nch + 1, figsize=(20, 30))
        for i, ax in enumerate(axes):
            ax.plot(convsound[:, i])
            ax.set_ylim((-1, 1))

    return convsound


def filter_sound(
    fs_soundcard,
    h,
    fs_filter,
    sound,
    fs_sound,
    npri,
    plot=False,
    normalize=True,
    include_reference=False,
):
    """Convolve filter and sound with approriate resampling.

    Parameters
    ----------
    fs_soundcard : TYPE
        Description
    h : ndarray
        Control filters (no primary filters!)
    fs : int
        samplerate
    sound : ndarray
        sound file
    fs_sound : TYPE
        sample rate sound file
    npri : TYPE
        Add this many allpassed sounds to the top of the convsound

    """
    from adafilt import olafilt

    if fs_sound != fs_soundcard:
        up, down = (fs_soundcard / fs_sound).as_integer_ratio()
        print("Resampling sound with {}/{}".format(up, down))
        sound = resample_poly(sound, up, down)
        print("Done.")

    if fs_filter != fs_soundcard:
        up, down = (fs_soundcard / fs_filter).as_integer_ratio()
        print("Resampling filter with {}/{}".format(up, down))

        if plot:
            Response.from_time(fs_filter, h.T).plot(
                figsize=(10, 8), dblim=(-35, 2), flim=(10, fs_soundcard / 2)
            )

        h = resample_poly(h, up, down) * down / up

        if plot:
            Response.from_time(fs_soundcard, h.T).plot(
                figsize=(10, 8), dblim=(-35, 2), flim=(10, fs_soundcard / 2)
            )

        print("Done.")

    # add allpass diracs for primary sources
    h_pri = np.zeros((h.shape[0], npri))
    h_pri[0, :] = 1
    h = np.concatenate((h_pri, h), axis=1)

    # convolve sounds
    convsound = np.zeros((sound.shape[0], h.shape[1]))
    for i in range(h.shape[1]):
        convsound[:, i] = olafilt(h[:, i], sound)

    if include_reference == True:
        convsound = np.concatenate((convsound, sound[:, None]), axis=1)
    elif isinstance(include_reference, int):
        for i in range(include_reference):
            convsound = np.concatenate((convsound, sound[:, None]), axis=1)


    if normalize:
        maxamp = np.max(np.abs(convsound))
        convsound /= (maxamp * 1.05)

    if plot:
        fig, axes = plt.subplots(nrows=convsound.shape[1], figsize=(20, 30))
        t = np.arange(convsound.shape[0])[::10]/fs_soundcard
        for i, ax in enumerate(axes):
            ax.plot(t, convsound[::10, i])
            ax.set_ylim(-1, 1)

    return convsound
