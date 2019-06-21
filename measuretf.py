"""Collection of functions for transfer-function measurements."""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from response import Response
from scipy.signal import (
    hann,
    butter,
    lfilter,
    get_window,
    convolve,
    max_len_seq,
    tukey,
    flattop,
    csd,
    welch,
    correlate,
)
from scipy.io import loadmat, wavfile


def is_ipynb():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if is_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

# EXITATION SIGNALS
# TODO: Collection Signals in a class Signal with common features like post_silence,
#       fades, f_start and end, maxamp ...


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
    T, fs, tfade=0.1, flim=(5, 20e3), fknee=30, post_silence=0, noise="randphase"
):
    """Generate white noise.

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
    flim = np.asarray(flim)

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
        msound[ch * N : (ch + 1) * N, ch] = sound
        if reference:
            msound[ch * N : (ch + 1) * N, -1] = sound

    return msound


# ESTIMATE TRANSFER FUNCTIONS


def transfer_function(
    ref, meas, ret_time=True,
    axis=-1,
    Ywindow=None,
    fftwindow=None,
    reg=0,
    reg_lim_dB=None
):
    """Compute transfer-function between time domain signals.

    Parameters
    ----------
    ref : ndarray, float
        Reference signal.
    meas : ndarray, float
        Measured signal.
    ret_time : bool, optional
        If True, return in time domain. Otherwise return in frequency domain.
    axis : integer, optional
        Time axis
    Ywindow : Tuple or None, optional
        Apply a frequency domain window to `meas`. (fs, startwindow, stopwindow) before
        the FFT to avoid numerical problems close to Nyquist frequency due to division
        by small numbers.
    fftwindow : None, optional
        Apply a Tukey time window to meas before doing the fft removing clicks at the
        end and beginning of the recording.
    reg : float
        Regularization in deconvolution
    reg_lim_dB: float
        Regularize such that reference has at least reg_lim_dB below of maximum energy
        in each bin.

    Returns
    -------
    h : ndarray, float
        Transfer-function between ref and
        meas.
    """
    # NOTE: is this used anywhere? If not remove
    if fftwindow:
        print("fftwindowing!")
        w = tukey(ref.shape[axis], alpha=0.1)
        meas = np.moveaxis(meas, axis, -1)
        meas = meas * w
        meas = np.moveaxis(meas, -1, axis)

    R = np.fft.rfft(ref, axis=axis)  # no need for normalization because
    Y = np.fft.rfft(meas, axis=axis)  # of division

    # Window the measurement before deconvolution, avoiding insane TF amplitudes near
    # the Nyquist frequency due to division by small numbers
    if Ywindow is not None:
        fs, startwindow, stopwindow = Ywindow
        W = freq_window(fs, ref.shape[axis], startwindow, stopwindow)
        Y = np.moveaxis(Y, axis, -1)
        Y *= W
        Y = np.moveaxis(Y, -1, axis)

    # FIXME: next two paragraphs are not very elegant
    R[R == 0] = np.finfo(complex).eps  # avoid devision by zero

    # Avoid large TF gains that lead to Fourier Transform numerical errors
    TOO_LARGE_GAIN = 1e9
    too_large = np.abs(Y / R) > TOO_LARGE_GAIN
    if np.any(too_large):
        warnings.warn(
            f"Some TF gains larger than {20*np.log10(TOO_LARGE_GAIN):.0f} dB. Setting to 0"
        )
        Y[too_large] = 0

    if reg_lim_dB is not None:
        # maximum of reference
        maxRdB = np.max(20 * np.log10(np.abs(R)), axis=axis)

        # power in reference should be at least
        minRdB = maxRdB - reg_lim_dB

        # 10 * log10(reg + |R|**2) = minRdB
        reg = 10**(minRdB / 10) - np.abs(R)**2
        reg[reg < 0] = 0

    H = Y * R.conj() / (np.abs(R) ** 2 + reg)

    if ret_time:
        h = np.fft.irfft(H, axis=axis, n=ref.shape[axis])
        return h
    else:
        return H


def multi_transfer_function(recs, ref_ch=0, ret_time=True):
    """Transfer-function between multichannel recording and reference channels.

    Parameters
    ----------
    recs : ndarray, shape (n_ch, n_ls, n_avg, n_tap)
        Multichannel recording.
    ref_ch : int
        Index of reference channel in recs.

    Returns
    -------
    ndarray, shape (n_ch, n_ls, n_avg, n_tap)
        Transfer function between reference and measured signals in time
        domain.
    """
    n_ch, n_ls, n_avg, n_tap = recs.shape

    if ret_time:
        tfs = np.zeros((n_ch, n_ls, n_avg, n_tap))
    else:
        tfs = np.zeros((n_ch, n_ls, n_avg, n_tap // 2 + 1))

    for avg in range(n_avg):
        for ch in range(n_ch):
            for ls in range(n_ls):
                tfs[ch, ls, avg] = transfer_function(
                    recs[ref_ch, ls, avg], recs[ch, ls, avg], ret_time=ret_time
                )

    return tfs


def transfer_function_with_reference(
        recs, ref=0, Ywindow=None, fftwindow=None, reg=0, reg_lim_dB=None
    ):
    """Transfer-function between multichannel recording and reference channels.

    Parameters
    ----------
    recs : ndarray, shape (no, ni, navg, nt)
        Multichannel recording.
    ref : int or list[int] or ndarray
        One of
            - index of reference channel
            - list of reference channels for each output
            - a reference signals of shape (..., nt)

    Returns
    -------
    ndarray, shape (no, ni, navg, nt)
        Transfer function between reference and measured signals in time
        domain.

    TODO: merge with multi_transfer_function. They are basically the same.
    """
    no, ni, navg, nt = recs.shape
    tfs = np.zeros((no, ni, navg, nt))
    for o in range(no):
        for i in range(ni):
            for avg in range(navg):
                if isinstance(ref, int):
                    # ref is reference channel
                    r = recs[o, ref, avg]
                elif isinstance(ref, list):
                    # ref is a list of reference signal channel
                    r = recs[o, ref[o], avg]
                else:
                    # ref is reference sound
                    r = ref

                tfs[o, i, avg] = transfer_function(
                    r,
                    recs[o, i, avg],
                    ret_time=True,
                    Ywindow=Ywindow,
                    fftwindow=fftwindow,
                    reg=reg,
                    reg_lim_dB=reg_lim_dB
                )

    return tfs


def transfer_function_csd(x, y, fs, compensate_delay=True, fwindow=None, **kwargs):
    """Compute transfer-function between time domain signals using Welch's method.

    Delay compensation mentioned e.g. in S. Muller, A. E. S. Member, and P. Massarani,
    “Transfer-Function Measurement with Sweeps.”


    Parameters
    ----------
    x, y : ndarray, float
        Reference and measured signal in one dimensional arrays of same length.
    fs : int
        Sampling frequency
    compensate_delay: optional, bool
        Compensate for delays in correlation estimations.
    fwindow: optional, tuple
        Frequency limits of frequency domain window to be applied after TF estimation.
    **kwargs
        Kwargs are fed to csd and welch functions.

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    H : ndarray, complex
        Transfer-function between ref and
        meas.
    """
    assert x.ndim == 1
    assert y.ndim == 1
    assert x.size == y.size

    if compensate_delay:
        x, y, dt = time_align(x, y, fs)

    f, S_xy = csd(x, y, fs=fs, **kwargs)
    _, S_xx = welch(x, fs=fs, **kwargs)

    H = S_xy / S_xx

    if compensate_delay:
        # reintroduce delay
        H *= np.exp(-1j * 2 * np.pi * f * dt)

    if fwindow is not None:
        startwindow, stopwindow = fwindow
        fs_new = 2 * int(f[-1])
        n_new = (H.shape[-1] - 1) * 2
        W = freq_window(fs_new, n_new, startwindow, stopwindow)
        H *= W

    return f, H


# B&K TIME DATA RECORDER PROCESSING


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
    fh = data["File_Header"]

    fs = int(float(fh.SampleFrequency))
    n_tap = int(fh.NumberOfSamplesPerChannel)
    n_ch = int(fh.NumberOfChannels)

    return fs, n_tap, n_ch


def load_mat_recording(fname, n_ls=1, n_avg=1, fullout=False):
    """Load multichannel Time Data Recording into ndarray.

    Parameters
    ----------
    fname : str
        Name of the mat file. Can also pass open file-like object. Reference
        channel is assumed to be recorded in first channel.
    n_ls : int, optional
        Number of simultaneously, in series recorded output channels.
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
    fh = data["File_Header"]
    n_ch = int(fh.NumberOfChannels)
    fs = int(float(fh.SampleFrequency))  # only int() doesn't work ...
    n_tap = int(int(fh.NumberOfSamplesPerChannel) / n_ls / n_avg)

    recs = np.zeros((n_ch, n_ls, n_avg, n_tap))
    for i in range(n_ch):
        # shape (N*n_avg*n_ls, ) -> (n_avg, N*n_ls)
        temp = np.array(np.split(data["Channel_{}_Data".format(i + 1)], n_avg))
        recs[i] = np.array(np.split(temp, n_ls, axis=1))  # (n_ls, n_avg, N)

    if fullout:
        return recs, fs, n_ch, n_tap

    return recs, fs


def load_npz_recording(fname, n_ls=1, n_avg=1, fullout=False):
    """Load multichannel Time Data Recording into ndarray.

    Parameters
    ----------
    fname : str
        Name of the mat file. Can also pass open file-like object. Reference
        channel is assumed to be recorded in first channel.
    n_ls : int, optional
        Number of simultaneously, in series recorded output channels.
    n_avg : int, optional
        Number of recorded averages.

    Returns
    -------
    recs : ndarray, shape (n_ch, n_ls, n_avg, n_tap)
        Recordings, sliced into averages and output channels.
    fs : int
        Sampling frequency.
    """

    with np.load(fname) as data:
        fs = data["fs"]
        n_otap = data["n_tap"]
        n_ch = data["n_ch"]
        orecs = data["recs"]  # has shape n_ch x n_taps

    n_tap = n_otap / n_ls / n_avg
    if n_tap.is_integer():
        n_tap = int(n_tap)
    else:
        raise ValueError("n_tap is not an integer")

    recs = np.zeros((n_ch, n_ls, n_avg, n_tap))
    for i in range(n_ch):
        # shape  (ntaps*n_avg*n_ls, ) -> (n_ls, ntaps*n_avg)
        temp = np.array(np.split(orecs[i], n_ls))
        temp = np.array(np.split(temp, n_avg, axis=-1))  # (n_avg, n_ls, n_taps)
        temp = np.moveaxis(temp, 0, 1)  # (n_ls, n_avg, n_taps)
        recs[i] = temp

    if fullout:
        return recs, fs, n_ch, n_tap

    return recs, fs


def load_recording(fname, n_ls=1, n_avg=1, fullout=False):
    """Load multichannel Time Data Recording into ndarray.

    Parameters
    ----------
    fname : str
        Name of the mat or npz file. Can also pass open file-like object. Reference
        channel is assumed to be recorded in first channel.
    n_ls : int, optional
        Number of simultaneously, in series recorded output channels.
    n_avg : int, optional
        Number of recorded averages.

    Returns
    -------
    recs : ndarray, shape (n_ch, n_ls, n_avg, n_tap)
        Recordings, sliced into averages and output channels.
    fs : int
        Sampling frequency.
    """
    fname = Path(fname)
    if fname.suffix == ".mat":
        recs, fs, n_ch, n_tap = load_mat_recording(
            fname, n_ls=n_ls, n_avg=n_avg, fullout=True
        )
    elif fname.suffix == ".npz":
        recs, fs, n_ch, n_tap = load_npz_recording(
            fname, n_ls=n_ls, n_avg=n_avg, fullout=True
        )

    if fullout:
        return recs, fs, n_ch, n_tap

    return recs, fs


def convert_TDRmat_recording_to_npz(fname, output_folder=None):
    """Convert .mat recording from Time Data recorder into npz format.

    Parameters
    ----------
    fname : str or Path
        File path
    n_ls : int, optional
        Number of simultaneously, in series recorded output channels.
    n_avg : int, optional
        Number of recorded averages.
    folder : None or Path, optional
        Save in this folder, instead of same folder as fname (default)
    """
    data = loadmat(fname, struct_as_record=False, squeeze_me=True)
    fh = data["File_Header"]
    n_ch = int(fh.NumberOfChannels)
    fs = int(float(fh.SampleFrequency))  # only int() doesn't work ...
    n_tap = int(int(fh.NumberOfSamplesPerChannel))

    recs = np.zeros((n_ch, n_tap))

    for i in range(n_ch):
        recs[i] = np.array(data["Channel_{}_Data".format(i + 1)])

    # remove .mat suffix
    path = Path(fname)
    parent = Path(output_folder) if output_folder is not None else path.parent
    newpath = parent / path.stem

    np.savez(newpath, recs=recs, fs=fs, n_ch=n_ch, n_tap=n_tap)


def folder_convert_TDRmat_recording_to_npz(path, output_folder=None, unlink=False):
    """Convert all .mat recording in folder into npz format.

    Parameters
    ----------
    path : str or Path
        Folder path
    n_ls : int, optional
        Number of simultaneously, in series recorded output channels.
    n_avg : int, optional
        Number of recorded averages.
    folder : None or Path, optional
        Save in this folder, instead of same folder as fname (default)
    """
    # delete content of outputfolder
    if output_folder is not None:
        for f in Path(output_folder).glob("*.npz"):
            f.unlink()

    files = Path(path).glob("*.mat")
    for f in tqdm(list(files)):
        convert_TDRmat_recording_to_npz(f, output_folder=output_folder)


def transfer_functions_from_recordings(
    fp,
    n_ls,
    n_meas,
    fformat="Recording-{}.mat",
    n_avg=1,
    ref_ch=0,
    lowpass_lim=None,
    lowpass_butt=None,
    twindow=None,
    take_T=None,
    H_comp=None,
):
    """Calculate transfer-functions from a set of recordings inside a folder.

    Parameters
    ----------
    fp : str or Path
        Path to folder.
    n_ls : int
        Number of serially measured channels.
    n_meas : int
        Total number of recordings
    H_comp : None or ndarray, shape (n_ch - 1, nf), optional
        If present, apply a compensation filter to each single recording.
    fformat : str, optional
        Recording file naming. Must include one '{}' to enumerate.
    n_avg : int, optional
        Number of averages.
    ref_ch : int, optional
        Index of reference channel
    lowpass_lim : None or Tuple, optional
        Tuple (fl, fu) that defines transition range of lowpass filter.
    twindow : None or tuple
        Specify time domain window.
    take_T : float or None, optional
        If float, only take the take_T first seconds of the impulse responses.
        if twindiow==None: Hann window in last 5%.

    Returns
    -------
    ndarray,
        Transfer-functions, shape (n_meas, n_ch - 1, n_ls, n_avg, n_tap // 2 + 1)
    """
    fpath = Path(fp)

    # read meta data from first recording
    fname = fpath / fformat.format(1)
    _, fs, n_ch, n_tap = load_recording(fname, n_ls=n_ls, n_avg=n_avg, fullout=True)

    if take_T is not None:
        # only take first take_T seconds of impulse responses
        n_tap = int(np.ceil(take_T * fs))
    else:
        n_tap = int(n_tap / n_ls / n_avg)

    shape_H = (n_meas, n_ch - 1, n_ls, n_avg, n_tap // 2 + 1)
    H = np.zeros(shape_H, dtype=complex)

    if H_comp is not None:
        # cut H to same length in time domain
        H_comp = Response.from_freq(int(fs), H_comp).ncshrink(n_tap * 1 / fs).in_freq

    for i in tqdm(np.arange(n_meas)):
        fname = fpath / fformat.format(i + 1)

        # load time domain recordings
        if Path(fname).suffix == ".mat":
            temp, fs = load_mat_recording(fname, n_ls=n_ls, n_avg=n_avg)
        elif Path(fname).suffix == ".npz":
            temp, fs = load_npz_recording(fname, n_ls=n_ls, n_avg=n_avg)
        else:
            raise ValueError()

        temp = multi_transfer_function(temp, ref_ch=ref_ch, ret_time=True)

        # exclude reference channel
        temp = np.delete(temp, 0, axis=0)

        if take_T is not None:
            # only take first take_T seconds
            temp = temp[..., :n_tap]
            if twindow is None:
                # time window the tail
                nwin = int(round(take_T * 0.05 * fs))
                w = hann(2 * nwin)[nwin:]
                temp[..., -nwin:] *= w

        if H_comp is not None:
            # convolve with compensation filter
            Temp = np.fft.rfft(temp) * H_comp[:, None, None, :]
            temp = np.fft.irfft(Temp, n=n_tap)
            del Temp

        if lowpass_lim is not None:
            # filter out HF noise with zero phase frequency domain window
            temp = lowpass_by_frequency_domain_window(fs, temp, *lowpass_lim)

        if lowpass_butt is not None:
            # filter HF noise with butterworth
            order, cutoff = lowpass_butt
            b, a = butter(order, cutoff / (fs / 2), "low")
            temp = lfilter(b, a, temp, axis=-1)

        if twindow is not None:
            T = time_window(fs, n_tap, *twindow)
            temp *= T

        H[i] = np.fft.rfft(temp)

    return H, fs, n_tap


def cut_recording(fname, cuts, names=None, remove_orig=False, outfolder=None,
    shift_before_cut=0):
    """Cut recordings at samples.
    """
    data = np.load(fname)
    recs = data["recs"]
    if shift_before_cut:
        recs = np.roll(recs, shift_before_cut, axis=-1)
    recs = np.split(recs, cuts, axis=-1)
    sounds = np.split(data["sound"].T, cuts, axis=-1)
    path = Path(fname)

    for i, rs in enumerate(zip(recs, sounds)):
        r, s = rs
        add = "-{}".format(names[i] if names is not None else i)
        parent = Path(outfolder) if outfolder is not None else path.parent
        newpath = parent / (path.stem + add)
        np.savez(
            newpath,
            recs=r,
            fs=data["fs"],
            n_ch=r.shape[0],
            n_tap=r.shape[1],
            sound=s,
            ref_ch=data["ref_ch"]
        )

    data.close()

    if remove_orig:
        # remove original files
        path.unlink()


# TF MEASUREMENT


def measure_single_output_impulse_response(
    sound,
    fs,
    out_ch=1,
    in_ch=1,
    ref_ch_indx=None,
    ref_sound_indx=None,
    calibration_gains=None,
    **sd_kwargs,
):
    """Meassure impulse response between single output and multiple inputs.

    Parameters
    ----------
    sound : ndarray, shape (nt,)
        Excitation signal
    fs : int
        Sampling rate of sound
    out_ch : int, optional
        Output channel
    in_ch : int or list, optional
        List of input channels
    ref_ch_indx : None or int, optional
        Index of reference channel in in_ch. If none, take sound as reference.

    Returns
    -------
    ndarray, shape (n_in, nt)
        Impulse response between output channel and input channels
    """
    out_ch = np.atleast_1d(out_ch)
    in_ch = np.atleast_1d(in_ch)

    if sound.ndim > 1:
        if ref_sound_indx is None and ref_ch_indx is None:
            raise ValueError("Select a reference with ref_sound_indx or ref_ch_indx.")

        if ref_sound_indx is not None:
            assert np.atleast_2d(sound).shape[1] - 1 == out_ch.size
            ref = sound[:, ref_sound_indx][:, None]
            sound = np.delete(sound, ref_sound_indx, axis=-1)
        else:
            assert np.atleast_2d(sound).shape[1] == out_ch.size

    # make copies of mapping because of
    # github.com/spatialaudio/python-sounddevice/issues/135
    rec = sd.playrec(
        sound,
        samplerate=fs,
        input_mapping=in_ch.copy(),
        output_mapping=out_ch.copy(),
        blocking=True,
        **sd_kwargs,
    )

    if calibration_gains is not None:
        calibration_gains = np.atleast_1d(calibration_gains)
        rec = rec * calibration_gains

    if ref_ch_indx is not None and ref_sound_indx is not None:
        raise ValueError("Cannot specify reference in input channel and sound file.")
    elif ref_ch_indx is not None:
        # use one of the input channels as reference
        ref = rec[:, ref_ch_indx][:, None]
        meas = np.delete(rec, ref_ch_indx, axis=-1)
    elif ref_sound_indx is not None:
        # use on of the sound file channels as reference
        meas = rec
        # ref was set above
    else:
        # just use the sound as reference
        meas = rec
        ref = sound[:, None]

    return transfer_function(ref, meas, axis=0).T


def measure_multi_output_impulse_respone(
    sound,
    fs_sound,
    out_ch,
    in_ch,
    ref_ch_indx=None,
    fs_resample=None,
    n_avg=1,
    tcut=None,
    timewindow=None,
    lowpass=None,
    calibration_gains=None,
    **sd_kwargs
):
    """Measure impulse response between multiple outputs and multiple inputs.

    Parameters
    ----------
    sound : TYPE
        Description
    fs_sound : TYPE
        Description
    out_ch : TYPE
        Description
    in_ch : TYPE
        Description
    ref_ch_indx : None or int, optional
        Index of reference channel in input channel list in_ch.
    fs_resample : None, optional
        Description
    n_avg : int, optional
        Description
    tcut : None, optional
        Do a timecut
    timewindow : None, optional
        Do a timewindow
    lowpass : None, optional
        Description
    calibration_gains : None, optional
        If not note, Apply calibration gains to measurements
    **sd_kwargs
        Description

    Return
    ------------------
    irs : ndarray, shape (n_out, n_in, n_t)
        Description
    """
    out_ch = np.atleast_1d(out_ch)
    in_ch = np.atleast_1d(in_ch)

    irs = []

    for i, oc in enumerate(out_ch):
        ir = 0
        fs = fs_sound
        for j in range(n_avg):
            ir += (
                measure_single_output_impulse_response(
                    sound,
                    fs_sound,
                    out_ch=oc,
                    in_ch=in_ch,
                    ref_ch_indx=ref_ch_indx,
                    calibration_gains=calibration_gains,
                    **sd_kwargs,
                )
                / n_avg
            )

        if tcut is not None:
            ir = Response.from_time(fs, ir).timecrop(0, tcut).in_time

        if lowpass is not None:
            ir = (
                Response.from_time(fs, ir)
                .lowpass_by_frequency_domain_window(*lowpass)
                .in_time
            )

        if timewindow is not None:
            ir = Response.from_time(fs, ir).time_window(*timewindow).in_time

        if fs_resample is not None:
            # TODO: use resample_poly
            ir = Response.from_time(fs, ir).resample(fs_resample).in_time  # (nin, nt)

        irs.append(ir)

    irs = np.stack(irs, axis=0)  # shape (nout, nin, nt)

    return irs


# RECORD EXCITATIONS


def record_single_output_excitation(sound, fs, out_ch=1, in_ch=1, **sd_kwargs):
    """Meassure impulse response between single output and multiple inputs.

    Parameters
    ----------
    sound : ndarray, shape (nt,)
        Excitation signal
    fs : int
        Sampling rate of sound
    out_ch : int, optional
        Output channel
    in_ch : int or list, optional
        List of input channels
    ref_ch_indx : None or int, optional
        Index of reference channel in in_ch. If none, take sound as reference.

    Returns
    -------
    ndarray, shape (n_in, nt)
        Impulse response between output channel and input channels
    """
    out_ch = np.atleast_1d(out_ch)
    in_ch = np.atleast_1d(in_ch)

    # make copies of mapping because of
    # github.com/spatialaudio/python-sounddevice/issues/135
    rec = sd.playrec(
        sound,
        samplerate=fs,
        input_mapping=in_ch.copy(),
        output_mapping=out_ch.copy(),
        blocking=True,
        **sd_kwargs,
    )

    return rec.T


def record_multi_output_excitation(
    sound, fs_sound, out_ch, in_ch, n_avg=1, **sd_kwargs
):
    """Measure impulse response between multiple outputs and multiple inputs.

    Parameters
    ----------
    sound : TYPE
        Description
    fs_sound : TYPE
        Description
    out_ch : TYPE
        Description
    in_ch : TYPE
        Description
    ref_ch_indx : None, optional
        Description

    Returns
    -------
    ndarray, shape (n_out, n_in, n_avg, n_tap)
        Description
    """
    out_ch = np.atleast_1d(out_ch)
    in_ch = np.atleast_1d(in_ch)

    recs = np.zeros((len(out_ch), len(in_ch), n_avg, len(sound)))
    for o, oc in enumerate(out_ch):
        for avg in range(n_avg):
            recs[o, :, avg, :] = record_single_output_excitation(
                sound, fs_sound, out_ch=oc, in_ch=in_ch, **sd_kwargs
            )
    return recs


def saverec_multi_output_excitation(
    filename,
    sound,
    fs,
    out_ch,
    in_ch,
    n_avg=1,
    ref_ch=None,
    add_datetime_to_name=False,
    description="",
    **sd_kwargs,
):
    """Record and save a multi output excitation.

    Parameters
    ----------
    sound : TYPE
        Description
    fs_sound : TYPE
        Description
    out_ch : TYPE
        Description
    in_ch : TYPE
        Description
    ref_ch : None, optional
        Description

    Returns
    -------
    ndarray, shape (n_out, n_in, n_t)
        in digital full scale (calibrate it!)
    """
    start = datetime.now()
    recs = record_multi_output_excitation(
        sound, fs, out_ch, in_ch, n_avg=n_avg, **sd_kwargs
    )

    if add_datetime_to_name:
        datetime_str = start.isoformat(" ", "seconds").replace(":", "-")
        fn = str(filename) + " - " + datetime_str
    else:
        fn = filename

    np.savez(
        fn,
        recs=recs,
        ref_ch=ref_ch,
        fs=fs,
        in_ch=in_ch,
        out_ch=out_ch,
        sound=sound,
        n_avg=n_avg,
        datetime_start=start,
        datetime_end=datetime.now(),
        description=description,
    )

    return recs


def saverec_recording(
    filename,
    fs,
    in_ch,
    T,
    ref_ch=None,
    add_datetime_to_name=False,
    description="",
    **sd_kwargs,
):
    recs = sd.rec(
        frames=int(fs * T), mapping=in_ch, blocking=True, samplerate=fs, **sd_kwargs
    )

    if add_datetime_to_name:
        datetime_str = datetime.now().isoformat(timespec="seconds").replace(":", "-")
        fn = filename + " - " + datetime_str
    else:
        fn = filename

    np.savez(
        fn,
        recs=recs.T,
        ref_ch=ref_ch,
        fs=fs,
        in_ch=in_ch,
        datetime=datetime.now(),
        description=description,
        sound=None,
    )

    return recs


# FOR KFF18


def plot_rec(fs, recs, **plot_kwargs):
    """Plot a recording."""
    recs = np.atleast_3d(recs.T).T
    fig, ax = plt.subplots(
        nrows=recs.shape[1], ncols=recs.shape[0], squeeze=False, **plot_kwargs
    )
    t = time_vector(recs.shape[-1], fs)
    for i in range(recs.shape[1]):
        for j in range(recs.shape[0]):
            ax[i, j].set_title(f"Out: {j}, In: {i}")
            ax[i, j].plot(t, recs[j, i].T)
    return fig


def load_rec(fname, plot=True, **plot_kwargs):
    """Show content of saved recording.

    Parameters
    ----------
    fname : path
        Recording in npz format
    plot : bool, optional
        If true, plot recording

    Returns
    -------
    recs, fs, ref_ch
    """
    with np.load(fname) as data:
        recs = data["recs"]  # shape (n_out, n_in, nt)
        fs = int(data["fs"])
        ref_ch = data["ref_ch"]

    if plot:
        plot_rec(fs, recs, **plot_kwargs)
    return recs, fs, ref_ch


def convert_wav_to_rec(
    wavfname,
    recname,
    ref_ch,
    n_ls_ch,
    add_datetime_to_name=False,
    description=None,
    plot=False,
):
    fs, recs = wavfile.read(wavfname)

    if recs.ndim == 1:
        recs = np.atleast_2d(recs).T

    # remove samples for clean split
    remove_samples = recs.shape[0] % n_ls_ch
    if remove_samples > 0:
        print("removing samples:", remove_samples, "nt:", recs.shape[0])
        recs = recs[:-remove_samples, :]

    if plot:
        fig = plot_rec(fs, recs.T)
        fig.suptitle("before split")

    recs = np.stack(np.split(recs.T, n_ls_ch, axis=1), axis=0)

    if plot:
        fig = plot_rec(fs, recs)
        fig.suptitle("after split")
        plt.show()

    if add_datetime_to_name:
        fn = recname + " - {}".format(datetime.now())
    else:
        fn = recname

    np.savez(
        fn,
        recs=recs,
        ref_ch=ref_ch,
        fs=fs,
        datetime=datetime.now(),
        description=description,
        sound=None,
    )
    return recs, fs


def tf_and_post_from_saved_rec(
    fname,
    tcut=None,
    twindow=None,
    fs_resample=None,
    fwindow=None,
    plot=False,
    ref=None,
    calibration_gain=None,
    fftwindow=None,
    reg_lim_dB=None,
):
    with np.load(fname) as data:
        recs = data["recs"]  # shape (n_out, n_in, navg, nt)
        fs = int(data["fs"])
        sound = data["sound"]
        ref_ch = data["ref_ch"]

    # accept legancy format without reps (n_out, n_in, nt)
    if recs.ndim == 3:
        recs = recs[:, :, None, :]

    if calibration_gain is not None:
        recs = recs * np.asarray(calibration_gain)[None, :, None, None]

    if plot:
        Response.from_time(fs, recs).plot(figsize=(10, 10))

    if fwindow is not None:
        fwindow = (fs, *fwindow)

    if ref == "sound":
        # NOTE: no calibration!
        ref = sound
    elif ref is not None:
        ref = int(ref)
    else:
        ref = int(ref_ch)

    irs = transfer_function_with_reference(
        recs, ref=ref, Ywindow=fwindow, fftwindow=fftwindow, reg_lim_dB=reg_lim_dB,
    )

    if plot:
        Response.from_time(fs, irs).plot(figsize=(10, 10))

    if twindow is not None:
        irs = Response.from_time(fs, irs).time_window(*twindow).in_time

        if plot:
            Response.from_time(fs, irs).plot(figsize=(10, 10))

    if fs_resample is not None:
        irs = (
            Response.from_time(fs, irs)
            .resample_poly(fs_resample, normalize="same_gain")
            .in_time
        )  # (nin, nt)

        fs = fs_resample
        if plot:
            Response.from_time(fs_resample, irs).plot(figsize=(10, 10))

    if tcut is not None:
        irs = Response.from_time(fs, irs).timecrop(0, tcut).in_time

        if plot:
            Response.from_time(fs, irs).plot(figsize=(10, 10))
    return fs, irs


def tf_and_post_from_saved_rec_csd(
    fname,
    tcut=None,
    twindow=None,
    rec_twindow=None,
    fs_resample=None,
    fwindow=None,
    plot=False,
    ref_ch=None,
    calibration_gain=None,
    nperseg=256,
    mode="csd",
    **csd_kwargs,
):

    assert nperseg % 2 == 0

    with np.load(fname) as data:
        recs = data["recs"]  # shape (n_out, n_in, nt)
        fs = int(data["fs"])
        ref_ch_npz = data["ref_ch"]

    if plot:
        fig = Response.from_time(fs, recs).plot(figsize=(10, 10), slce=plot)
        fig.suptitle("Recording")

    if calibration_gain is not None:
        recs = recs * np.asarray(calibration_gain)[None, :, None]

        if plot:
            fig = Response.from_time(fs, recs).plot(figsize=(10, 10), slce=plot)
            fig.suptitle("After calibration")

    # apply time window to recordings
    if rec_twindow is not None:
        recs = Response.from_time(fs, recs).time_window(*rec_twindow).in_time

        if plot:
            fig = Response.from_time(fs, recs).plot(figsize=(10, 10), slce=plot)
            fig.suptitle("After rec_twindow")

    if fs_resample is not None:
        recs = (
            Response.from_time(fs, recs)
            .resample_poly(fs_resample, normalize="same_amplitude")
            .in_time
        )
        fs = int(fs_resample)

        if plot:
            fig = Response.from_time(fs, recs).plot(figsize=(10, 10), slce=plot)
            fig.suptitle("After resampling")

    if ref_ch is None:
        ref_ch = ref_ch_npz

    # compute irs
    no, ni, nt = recs.shape
    if nperseg and mode == "csd":
        nt = nperseg
    irs = np.zeros((no, ni, nt))
    for o in range(no):
        for i in range(ni):
            if isinstance(ref_ch, list):  # NOTE: not tested!
                x = recs[o, ref_ch[o]]
            else:
                x = recs[o, ref_ch]
            y = recs[o, i]
            if mode == "csd":
                f, H = transfer_function_csd(
                    x, y, fs, nperseg=nperseg, fwindow=fwindow, **csd_kwargs
                )
                irs[o, i] = np.fft.irfft(H)
            elif mode == "naive":
                irs[o, i] = transfer_function(x, y, Ywindow=(fs, *fwindow))
            else:
                raise ValueError

    if mode == "csd":
        fs = 2 * int(f[-1])  # NOTE: assumes even nperseg

    if plot:
        fig = Response.from_time(fs, irs).plot(figsize=(10, 10), slce=plot)
        fig.suptitle("IRS")

    # apply time window to impulse response
    if twindow is not None:
        irs = Response.from_time(fs, irs).time_window(*twindow).in_time

        if plot:
            Response.from_time(fs, irs).plot(figsize=(10, 10), slce=plot)

    if tcut is not None:
        irs = Response.from_time(fs, irs).timecrop(0, tcut).in_time

        if plot:
            Response.from_time(fs, irs).plot(figsize=(10, 10), slce=plot)
    return fs, irs


def lanxi_reference_calibration_gain(
    danterec, lanxirec, ref_ch_dante=None, ref_ch_lanxi=None, ls_ch=0, plot=False
):
    """Compute calibration gain for lanxi reference channel."""
    with np.load(danterec) as data:
        fs = int(data["fs"])
        ref_ch_dante = data["ref_ch"] if ref_ch_dante is None else ref_ch_dante
        recs_dante = data["recs"][ls_ch, ref_ch_dante, :]

    with np.load(lanxirec) as data:
        fs_lanxi = int(data["fs"])
        ref_ch_lanxi = data["ref_ch"] if ref_ch_lanxi is None else ref_ch_lanxi
        recs_lanxi = data["recs"][ls_ch, ref_ch_lanxi, :]

    if fs_lanxi != fs:
        recs_lanxi = (
            Response.from_time(fs_lanxi, recs_lanxi)
            .resample_poly(fs, normalize="same_amplitude")
            .in_time
        )
        fs_lanxi = fs

    # flattop window the recording
    window = flattop(len(recs_lanxi))
    gain_window = window.mean()
    rec_lanxi_win = recs_lanxi * window / gain_window

    window = flattop(len(recs_dante))
    gain_window = window.mean()
    rec_dante_win = recs_dante * window / gain_window

    A_dante = amplitude_spectrum(rec_dante_win)
    A_lanxi = amplitude_spectrum(rec_lanxi_win)

    max_dante = np.abs(A_dante).max()
    max_lanxi = np.abs(A_lanxi).max()

    ref_cali_gain = max_dante / max_lanxi

    if plot:
        t_dante = time_vector(len(recs_dante), fs)
        t_lanxi = time_vector(len(recs_lanxi), fs_lanxi)

        plt.figure()
        plt.plot(t_dante, recs_dante)
        plt.plot(t_lanxi, recs_lanxi * ref_cali_gain)

        f_dante = frequency_vector(len(recs_dante), fs)
        f_lanxi = frequency_vector(len(recs_lanxi), fs_lanxi)

        plt.figure()
        plt.plot(f_dante, np.abs(A_dante))
        plt.plot(f_lanxi, np.abs(A_lanxi * ref_cali_gain))
        plt.xlim(995, 1005)

    return ref_cali_gain


def record_calibration(
    name, mic_channel, T_rec=5, fs_soundcard=48000, device=None, plot=False, notes=""
):
    # record the calibrated pressure
    rec = sd.rec(
        T_rec * fs_soundcard,
        samplerate=fs_soundcard,
        mapping=mic_channel,
        blocking=True,
        device=device,
    )

    rec = np.atleast_3d(rec)
    rec = np.moveaxis(rec, 0, -1)

    np.savez(
        name + " - mic_channel {}".format(mic_channel),
        recs=rec,
        mic_channel=mic_channel,
        fs=fs_soundcard,
        datetime=datetime.now(),
        notes=notes,
    )

    if plot:
        plt.figure()
        t = time_vector(T_rec * fs_soundcard, fs_soundcard)
        plt.plot(t, rec[0, 0, :])

    return rec


def calibration_gain_from_recording(fname_rec, mic_channel, L_calib=94, plot=False):
    with np.load(fname_rec) as data:
        rec_all = data["recs"]
        rec_mic = rec_all[0, mic_channel]
        fs = data["fs"]

    # flattop window the recording
    window = flattop(len(rec_mic))
    gain_window = window.mean()
    rec = rec_mic * window / gain_window

    p_calib = 10 ** (L_calib / 20) * 20e-6 * np.sqrt(2)
    A = amplitude_spectrum(rec, axis=0)
    p_meas = np.abs(A).max()

    calibration_gain = p_calib / p_meas

    if plot:
        freqs = frequency_vector(rec.shape[0], fs)
        plt.figure()
        plt.plot(
            freqs, 20 * np.log10(np.abs(A) / np.sqrt(2) / 20e-6), label="Uncalibrated"
        )
        plt.plot(
            freqs,
            20 * np.log10(np.abs(A * calibration_gain) / np.sqrt(2) / 20e-6),
            label="Calibrated",
        )
        plt.hlines(L_calib, 0, fs / 2, label="94dB")
        plt.legend()
        plt.xlim(995, 1005)
        plt.ylim(90, 96)
        plt.grid(True)

        plt.figure()
        t = time_vector(len(rec_mic), fs)
        plt.plot(t, rec_mic)

    return calibration_gain


# NON-LINEAR


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


# FILTERING


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
    """Convolve signel channel sound with multichannel impulse response.
    """
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


# FFT FUNCS


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


# UTILS


def find_nearest(array, value):
    """Find nearest value in an array and its index."""
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def time_align(x, y, fs):
    """Time align two signals, zeropad as necessary.

    If `dt` is positive `x` was delayed and `y` zeropadded for same length.
    """
    n = x.size

    # delta time array to match xcorr
    s = np.arange(1 - n, n)

    # cross correlation
    xcorr = correlate(y, x, mode="full")

    # estimate delay in time
    dt = s[xcorr.argmax()] / fs

    # match both responses in time and length
    if dt >= 0:
        x = Response.from_time(fs, x).delay(dt, keep_length=False).in_time
        y = Response.from_time(fs, y).zeropad_to_length(x.size).in_time
    else:
        x = Response.from_time(fs, x).zeropad_to_length(y.size).in_time
        y = Response.from_time(fs, y).delay(-dt, keep_length=False).in_time

    return x, y, dt


def time_align_y(x, y, fs):
    """Time align y to match x.
    """
    n = x.size

    # delta time array to match xcorr
    s = np.arange(1 - n, n)

    # cross correlation
    xcorr = correlate(x, y, mode="full")

    # estimate delay in time
    dt = s[xcorr.argmax()] / fs

    y = Response.from_time(fs, y).delay(dt, keep_length=True).in_time

    return y, dt


def coherence_csd(x, y, fs, compensate_delay=True, **csd_kwargs):
    """Compute maginitude squared coherence of two signals using Welch's method.

    Parameters
    ----------
    x, y : ndarray, float
        Reference and measured signal in one dimensional arrays of same length.
    fs : int
        Sampling frequency
    compensate_delay: optional, bool
        Compensate for delays in correlation estimations.
    **csd_kwargs
        Kwargs are fed to csd and welch functions.

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    gamma2 : ndarray
        Magnitude squared coherence
    """
    assert x.ndim == 1
    assert y.ndim == 1
    assert x.size == y.size

    if compensate_delay:
        x, y, _ = time_align(x, y, fs)

    f, S_xy = csd(x, y, fs=fs, **csd_kwargs)
    _, S_xx = welch(x, fs=fs, **csd_kwargs)
    _, S_yy = welch(y, fs=fs, **csd_kwargs)

    gamma2 = np.abs(S_xy) ** 2 / S_xx / S_yy

    return f, gamma2


def coherence_from_averages(x, y, n="pow2", avgaxis=-2, axis=-1, reg=0):
    """Compute magnitude squared coherence of from several instances of two signals.

    Parameters
    ----------
    x, y : ndarray, float
        Reference and measured signal.
    avgaxis : int, optional
        Axis over which to average.
    axis : int, optional
        Axis over which to take coherence (time axis).

    Returns
    -------
    ndarray, float
        Magnitude squared coherence

    """
    assert x.shape[axis] == y.shape[axis], "Need same nuber of datapoints"

    if n == "pow2":
        nt = x.shape[axis]
        n = 2 ** (nt - 1).bit_length()  # n of next power of 2

    X = np.fft.rfft(x, axis=axis, n=n)
    Y = np.fft.rfft(y, axis=axis, n=n)

    S_xx = (np.abs(X) ** 2).mean(axis=avgaxis)
    S_yy = (np.abs(Y) ** 2).mean(axis=avgaxis)
    S_xy = (X.conj() * Y).mean(axis=avgaxis)

    return np.abs(S_xy) ** 2 / (S_xx * S_yy + reg)
