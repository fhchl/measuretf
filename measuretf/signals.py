"""Excitation signals.

.. todo:: Collection Signals in a class Signal with common features like post_silence,
           fades, f_start and end, maxamp ...

"""

import numpy as np
from scipy.signal import hann, butter, lfilter, tukey, max_len_seq

from measuretf.fft import frequency_vector
from measuretf.utils import find_nearest


def exponential_sweep(
    T, fs, tfade=0.05, f_start=None, f_end=None, maxamp=1, post_silence=0
):
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
    post_silence : float
        Added zeros in seconds.

    Returns
    -------
    ndarray, floats, shape (round(T*fs),)
        An exponential sweep

    .. _Farina:
       A. Farina, “Simultaneous measurement of impulse response and distortion
       with a swept-sine techniqueMinnaar, Pauli,” in Proc. AES 108th conv,
       Paris, France, 2000, pp. 1–15.

    """
    # TODO: use scipy.signal.chirp instead?
    n_tap = int(np.round(T * fs))

    # start and stop frequencies
    if f_start is None:
        f_start = fs / n_tap
    if f_end is None:
        f_end = fs / 2

    assert f_start < f_end
    assert f_end <= fs / 2

    w_start = 2 * np.pi * f_start
    w_end = 2 * np.pi * f_end

    # constuct sweep
    t = np.linspace(0, T, n_tap, endpoint=False)
    sweep = np.sin(
        w_start
        * T
        / np.log(w_end / w_start)
        * (np.exp(t / T * np.log(w_end / w_start)) - 1)
    )
    sweep = sweep * maxamp  # some buffer in amplitude

    if tfade:
        # TODO: just use a tukey window here
        n_fade = round(tfade * fs)
        fading_window = hann(2 * n_fade)
        sweep[:n_fade] = sweep[:n_fade] * fading_window[:n_fade]
        sweep[-n_fade:] = sweep[-n_fade:] * fading_window[-n_fade:]

    if post_silence > 0:
        silence = np.zeros(int(round(post_silence * fs)))
        sweep = np.concatenate((sweep, silence))

    return sweep


def synchronized_swept_sine(
    Tapprox, fs, f_start=20, f_end=20e3, maxamp=1, tfade=0, post_silence=0
):
    """Generate synchronized swept sine.

    Sweep constructed in time domain as described by `Novak`_.

    Parameters
    ----------
    T : float
        length of sweep
    fs : int
        sampling frequency
    f_start, f_end: int, optional
        starting end ending frequencies
    tfade : float, optional
        Fade in and out time with Hann window.
    post_silence : float, optional
        Added zeros in seconds.

    Returns
    -------
    ndarray, floats
        The sweep. It will have a length approximately to Tapprox. If end frequency is
        integer multiple of start frequency the sweep will start and end with a 0.

    .. _Novak:
       A. Novak, P. Lotton, and L. Simon, “Transfer-Function Measurement with Sweeps *,”
       Journal of the Audio Engineering Society, vol. 63, no. 10, pp. 786–798, 2015.

    """
    assert f_start < f_end
    assert f_end <= fs / 2

    k = round(Tapprox * f_start / np.log(f_end / f_start))
    if k == 0:
        raise ValueError("Choose arguments s. t. f1 / log(f_2 / f_1) * T >= 0.5")
    L = k / f_start
    T = L * np.log(f_end / f_start)
    n = np.ceil(fs * T)
    t = np.linspace(0, T, n, endpoint=False)
    x = np.sin(2 * np.pi * f_start * L * np.exp(t / L))
    x *= maxamp

    if tfade:
        # TODO: just use a tukey window here
        n_fade = round(tfade * fs)
        fading_window = hann(2 * n_fade)
        x[:n_fade] = x[:n_fade] * fading_window[:n_fade]
        x[-n_fade:] = x[-n_fade:] * fading_window[-n_fade:]

    if post_silence > 0:
        silence = np.zeros(int(round(post_silence * fs)))
        x = np.concatenate((x, silence))

    return x


def mls(order, fs, highpass=10, fade_alpha=0.1, pink=True):
    mls = 2 * (max_len_seq(order)[0].astype(float) - 0.5)
    n = mls.size

    if pink:
        M = np.fft.rfft(mls)
        M[1:] /= frequency_vector(n, fs)[1:]
        mls = np.fft.irfft(M, n=n)

    if highpass:
        b, a = butter(2, highpass * 2 / fs, "high")
        mls = lfilter(b, a, mls)

    if fade_alpha:
        mls *= tukey(mls.size, alpha=fade_alpha)

    # normalize
    mls /= np.max(np.abs(mls))

    return mls


def pink_noise(
    T, fs, tfade=0.1, flim=None, fknee=30, post_silence=0, noise="randphase"
):
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
    if flim is None:
        flim = (10, fs / 2 - 1)

    N = int(np.round(T * fs))
    Nf = N // 2 + 1
    f = np.linspace(0, fs / 2, Nf, endpoint=False)
    flim = np.asarray(flim)

    if noise == "MLS":
        order = int(np.ceil(np.log2(N)))
        x = 2 * (max_len_seq(order)[0].astype(float) - 0.5)
        x = x[:N]
        X = np.fft.rfft(x)
    else:  # noise == "randphase"
        rand_phase = 2 * np.pi * np.random.random(Nf)
        X = np.ones(Nf) * np.exp(1j * rand_phase)

    # constant emphesis below fknee, pink above fknee
    _, ifknee = find_nearest(f, fknee)
    pink_weight = np.ones(f.shape)
    pink_weight[1:] /= f[1:]
    pink_weight[:ifknee] = pink_weight[ifknee]
    X *= pink_weight

    x = np.fft.irfft(X, n=N)

    # bandpass
    b, a = butter(4, flim / (fs / 2), "bandpass")
    x = lfilter(b, a, x)

    if tfade:
        # TODO: just use a tukey window here
        n_fade = round(tfade * fs)
        fading_window = hann(2 * n_fade)
        x[:n_fade] = x[:n_fade] * fading_window[:n_fade]
        x[-n_fade:] = x[-n_fade:] * fading_window[-n_fade:]

    if post_silence > 0:
        silence = np.zeros(int(round(post_silence * fs)))
        x = np.concatenate((x, silence))

    x /= np.abs(x).max()

    return x


def white_noise(
    T, fs, tfade=0.1, flim=(5, 20e3), post_silence=0, noise="randphase"
):
    """Generate white noise.

    Parameters
    ----------
    T : float
        Length of sound.
    fs : int
        Sampling frequency.
    tfade : float or None
        Fade in and out time with Hann window.
    flim : array_like or None
        Length-2 sequence giving bandpass frequencies.

    Returns
    -------
    ndarray, floats, shape (round(T*fs),)
        Pink noise

    """
    N = int(np.round(T * fs))
    Nf = N // 2 + 1

    if noise == "MLS":
        order = int(np.ceil(np.log2(N)))
        x = 2 * (max_len_seq(order)[0].astype(float) - 0.5)
        x = x[:N]
        X = np.fft.rfft(x)
    else:  # noise == "randphase"
        rand_phase = 2 * np.pi * np.random.random(Nf)
        X = np.ones(Nf) * np.exp(1j * rand_phase)

    x = np.fft.irfft(X, n=N)

    # bandpass
    if flim is not None:
        flim = np.asarray(flim)
        b, a = butter(4, flim / (fs / 2), "bandpass")
        x = lfilter(b, a, x)

    if tfade:
        # TODO: just use a tukey window here
        n_fade = round(tfade * fs)
        fading_window = hann(2 * n_fade)
        x[:n_fade] = x[:n_fade] * fading_window[:n_fade]
        x[-n_fade:] = x[-n_fade:] * fading_window[-n_fade:]

    if post_silence > 0:
        silence = np.zeros(int(round(post_silence * fs)))
        x = np.concatenate((x, silence))

    x /= np.abs(x).max()

    return x


def multichannel_serial_sound(sound, n_ch, reps=1, reference=False):
    """Generate a multichannel excitation that excites each channel with sound.

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
    assert sound.ndim == 1

    N = sound.size

    repsound = np.tile(sound, reps)

    multisound = np.zeros((n_ch * reps * N, n_ch))

    for ch in range(n_ch):
        multisound[ch * reps * N : (ch + 1) * reps * N, ch] = repsound

    if reference:
        multisound = np.concatenate(
            (multisound, multisound.sum(axis=-1)[:, None]), axis=-1
        )

    return multisound
