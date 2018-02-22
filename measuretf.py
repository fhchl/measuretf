"""Measurement signals."""

import numpy as np

from scipy.signal import hann, butter, lfilter
from scipy.io import loadmat


"""EXITATION SIGNALS"""


def exponential_sweep(T, fs, tfade=0.05, f_start=None, f_end=None):

    """Generate exponential sweep.

    Sweep constructed in time domain as described by `Farina`_ plus windowing.

    Parameters
    ----------
    T : float
        length of sweep
    fs : int
        sampling frequency
    tfade : float
        Fade in and out time with Hann window.

    Returns
    -------
    ndarray, floats, shape (round(T*fs),)
        An exponential sweep

    .. _Farina:
       A. Farina, “Simultaneous measurement of impulse response and distortion
       with a swept-sine techniqueMinnaar, Pauli,” in Proc. AES 108th conv,
       Paris, France, 2000, pp. 1–15.

    """
    ntaps = int(np.round(T * fs))

    # start and stop frequencies
    if f_start is None:
        f_start = fs / ntaps
    if f_end is None:
        f_end = fs / 2

    w_start = 2 * np.pi * f_start
    w_end = 2 * np.pi * f_end

    # constuct sweep
    t = np.linspace(0, T, ntaps)
    sweep = np.sin(w_start * T / np.log(w_end / w_start) *
                   (np.exp(t / T * np.log(w_end / w_start)) - 1))
    sweep = sweep * 0.95  # some buffer in amplitude

    # fade beginning and end
    n_fade = round(tfade * fs)
    fading_window = hann(2 * n_fade)
    sweep[:n_fade] = sweep[:n_fade] * fading_window[:n_fade]
    sweep[-n_fade:] = sweep[-n_fade:] * fading_window[-n_fade:]

    return sweep


def exponential_sweep_harmonic_delay(T, fs, N, f_start=None, f_end=None):
    """Delay of harmonic impulse response.

    From `Farina`_.

    Parameters
    ----------
    T : float
        Iength of impulse response.
    N : int
        Order of harmonic.
    f_start, f_end : float or None, optional
        Start and stop frequencies of exponential sweep.

    Returns
    -------
    TYPE
        Description

    .. _Farina:
       A. Farina, “Simultaneous measurement of impulse response and distortion
       with a swept-sine techniqueMinnaar, Pauli,” in Proc. AES 108th conv,
       Paris, France, 2000, pp. 1–15.
    """
    ntaps = int(np.round(T * fs))

    if f_start is None:
        f_start = fs / ntaps
    if f_end is None:
        f_end = fs / 2

    w_start = 2 * np.pi * f_start
    w_end = 2 * np.pi * f_end

    return T * np.log(N) / np.log(w_end / w_start)


def pink_noise(T, fs, tfade=0.1, flim=(5, 20e3)):
    """Generate pink noise.

    Parameters
    ----------
    T : float
        Length of sound.
    fs : int
        Sampling frequency.
    tfade : float
        Fade in and out time with Hann window.
    flim : array_like
        Length-2 sequence giving bandpass frequencies.

    Returns
    -------
    ndarray, floats, shape (round(T*fs),)
        Pink noise

    """
    N = int(np.round(T * fs))
    Nf = N // 2 + 1
    f = np.linspace(0, fs / 2, Nf)
    flim = np.asarray(flim)

    rand_phase = 2 * np.pi * np.random.random(Nf)
    X = np.ones(Nf) * np.exp(1j * rand_phase)
    X[1:] = X[1:] / f[1:]

    x = np.fft.irfft(X)

    # bandpass
    b, a = butter(4, flim / (fs / 2), 'bandpass')
    x = lfilter(b, a, x)

    # fade beginning and end
    n_fade = round(tfade * fs)
    fading_window = hann(2 * n_fade)
    x[:n_fade] = x[:n_fade] * fading_window[:n_fade]
    x[-n_fade:] = x[-n_fade:] * fading_window[-n_fade:]

    return x


def multichannel_serial_sound(sound, n_ch, reference=False):
    """Generate a multichannel array with sound in series on each
    channel.

    Parameters
    ----------
    sound : array, floats, shape (N,)
        Sound to be played through each channel in series.
    n_ch : int
        Number of output channels
    reference : bool, optional
        If true, add channel holding a repetition of `sound` for use as
        reference signal.

    Returns
    -------
    ndarray, shape (sound.size, n_ch(+1))
        Multichannel sound.
    """
    N = sound.size

    if reference:
        msound = np.zeros((n_ch * N, n_ch + 1))
    else:
        msound = np.zeros((n_ch * N, n_ch))

    for ch in range(n_ch):
        msound[ch * N:(ch + 1) * N, ch] = sound
        if reference:
            msound[ch * N:(ch + 1) * N, -1] = sound

    return msound


def transfer_function(ref, meas, axis=-1):
    """Compute transfer-function between time domain signals.

    Parameters
    ----------
    ref : ndarray, float
        Reference signal.
    meas : ndarray, float
        Measured signal.

    Returns
    -------
    h: ndarray, float
        Transfer-function between ref and
        meas.
    """
    R = np.fft.rfft(ref, axis=axis)   # no need for normalization because
    Y = np.fft.rfft(meas, axis=axis)  # of division
    H = Y / (R + np.spacing(1))  # avoid devided-by-zero errors
    h = frequency2time(H, axis=axis)
    return h


def multi_transfer_function(recs, ref_ch=0):
    """Transfer-function between multichannel recording and reference channels.

    Parameters
    ----------
    recs : ndarray, shape (n_ch, n_ls, n_avg, n_tap)
        Multichannel recording.
    ref_ch : int
        Index of reference channel in recs.

    Returns
    -------
    ndarray, shape (n_ch, n_ls, n_tap)
        Transfer function between reference and measured signals in time
        domain.
    """
    n_ch, n_ls, n_avg, n_tap = recs.shape
    tfs = np.zeros((n_ch, n_ls, n_tap))
    for avg in range(n_avg):
        for ch in range(n_ch):
            for ls in range(n_ls):
                tfs[ch, ls] += transfer_function(recs[ref_ch, ls, avg],
                                                 recs[ch, ls, avg]) / n_avg
    return tfs


def header_info(fname):
    """Header information of MAT-file exported by Time Data Recorder.

    Parameters
    ----------
    fname : str
        Name of the mat file. Can also pass open file-like object.

    Returns
    -------
    fs : int
        Sampling frequency.
    n_tap : int
        Number of samples per channel.
    n_ch : int
        Nuber of Channels
    """
    data = loadmat(fname, struct_as_record=False, squeeze_me=True)
    fh = data['File_Header']

    fs = int(float(fh.SampleFrequency))
    n_tap = int(fh.NumberOfSamplesPerChannel)
    n_ch = int(fh.NumberOfChannels)

    return fs, n_tap, n_ch


def load_recordings(fname, n_ls=1, n_avg=1):
    """Load multichannel Time Data Recording into ndarray.

    Parameters
    ----------
    fname : str
        Name of the mat file. Can also pass open file-like object. Reference
        channel is assumed to be recorded in first channel.
    n_ls : int, optional
        Number of simultaneously recorded output channels.
    n_avg : int, optional
        Number of recorded averages.

    Returns
    -------
    recs : ndarray, shape (n_ch, n_ls, n_avg, n_tap)
        Recordings, sliced into averages and output channels.
    fs : int
        Sampling frequency.
    """
    data = loadmat(fname, struct_as_record=False, squeeze_me=True)
    fh = data['File_Header']
    n_ch = int(fh.NumberOfChannels)
    fs = int(float(fh.SampleFrequency))  # only int() doesn't work ...
    n_tap = int(int(fh.NumberOfSamplesPerChannel) / n_ls / n_avg)

    recs = np.zeros((n_ch, n_ls, n_avg, n_tap))
    for i in range(n_ch):
        # shape (N*n_avg*n_ls, ) -> (n_avg, N*n_ls)
        temp = np.array(np.split(data['Channel_{}_Data'.format(i + 1)], n_avg))
        recs[i] = np.array(np.split(temp, n_ls, axis=1))  # (n_ls, n_avg, N)

    return recs, fs


"""FILTERING"""


def lowpass_by_frequency_domain_window(fs, x, fstart, fstop, axis=-1):
    """Summary

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
    X_windowed = time2frequency(x, axis=axis)
    X_windowed = np.moveaxis(X_windowed, axis, 0)
    X_windowed[windowed_samples] = (
        X_windowed[windowed_samples].T * half_window.T).T  # broadcasting
    X_windowed[stop:] = 0
    X_windowed = np.moveaxis(X_windowed, 0, axis)

    return frequency2time(X_windowed)


"""FFT FUNCS"""


def frequency_vector(n, fs):
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

    """
    f = np.arange(n // 2 + 1) / (n // 2)  #
    if n % 2 == 0:  # last element is Nyquist
        f *= fs / 2
    else:  # last element is not Nyquist
        f *= fs / 2 * (n - 1) / n
    return f


def time2frequency(x, axis=-1, norm=True):
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


def frequency2time(X, isEvenSampled=True, axis=-1, norm=True):
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


"""UTILS"""


def find_nearest(array, value):
    """Find nearest value in an array and its index."""
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # test pink noise
    T = 10
    fs = 44100
    n = pink_noise(T, fs)
    t = np.linspace(0, T, T * fs)
    f = np.linspace(0, fs / 2, T * fs // 2 + 1)
    plt.figure()
    plt.plot(t, n)
    plt.figure()
    plt.semilogx(f, 20 * np.log10(np.abs(np.fft.rfft(n))))
