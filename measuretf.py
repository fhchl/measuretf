"""Measurement signals."""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import hann, butter, lfilter, get_window, convolve
from scipy.io import loadmat

from tqdm import tqdm
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
    n_tap = int(np.round(T * fs))

    # start and stop frequencies
    if f_start is None:
        f_start = fs / n_tap
    if f_end is None:
        f_end = fs / 2

    w_start = 2 * np.pi * f_start
    w_end = 2 * np.pi * f_end

    # constuct sweep
    t = np.linspace(0, T, n_tap)
    sweep = np.sin(w_start * T / np.log(w_end / w_start) *
                   (np.exp(t / T * np.log(w_end / w_start)) - 1))
    sweep = sweep * 0.95  # some buffer in amplitude

    # fade beginning and end
    n_fade = round(tfade * fs)
    fading_window = hann(2 * n_fade)
    sweep[:n_fade] = sweep[:n_fade] * fading_window[:n_fade]
    sweep[-n_fade:] = sweep[-n_fade:] * fading_window[-n_fade:]

    return sweep


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


def transfer_function(ref, meas, ret_time=True, axis=-1):
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
    assert ref.shape == meas.shape
    n = ref.shape[axis]
    R = np.fft.rfft(ref, axis=axis)  # no need for normalization because
    Y = np.fft.rfft(meas, axis=axis)  # of division
    H = Y / (R + np.spacing(1))  # avoid devided-by-zero errors
    if ret_time:
        h = np.fft.irfft(H, axis=axis, n=n)
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
    ndarray, shape (n_ch, n_ls, n_tap)
        Transfer function between reference and measured signals in time
        domain.
    """
    n_ch, n_ls, n_avg, n_tap = recs.shape
    if ret_time:
        tfs = np.zeros((n_ch, n_ls, n_tap))
    else:
        tfs = np.zeros((n_ch, n_ls, n_tap // 2 + 1))
    for avg in range(n_avg):
        for ch in range(n_ch):
            for ls in range(n_ls):
                tfs[ch, ls] += transfer_function(recs[ref_ch, ls, avg],
                                                 recs[ch, ls, avg],
                                                 ret_time=ret_time) / n_avg
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
    fh = data['File_Header']
    n_ch = int(fh.NumberOfChannels)
    fs = int(float(fh.SampleFrequency))  # only int() doesn't work ...
    n_tap = int(int(fh.NumberOfSamplesPerChannel) / n_ls / n_avg)

    recs = np.zeros((n_ch, n_ls, n_avg, n_tap))
    for i in range(n_ch):
        # shape (N*n_avg*n_ls, ) -> (n_avg, N*n_ls)
        temp = np.array(np.split(data['Channel_{}_Data'.format(i + 1)], n_avg))
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
        fs = data['fs']
        n_otap = data['n_tap']
        n_ch = data['n_ch']
        orecs = data['recs']

    n_tap = n_otap / n_ls / n_avg
    if n_tap.is_integer():
        n_tap = int(n_tap)
    else:
        raise ValueError()

    recs = np.zeros((n_ch, n_ls, n_avg, n_tap))
    for i in range(n_ch):
        # shape (N*n_avg*n_ls, ) -> (n_avg, N*n_ls)
        temp = np.array(np.split(orecs[i], n_avg))
        recs[i] = np.array(np.split(temp, n_ls, axis=1))  # (n_ls, n_avg, N)

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
    fh = data['File_Header']
    n_ch = int(fh.NumberOfChannels)
    fs = int(float(fh.SampleFrequency))  # only int() doesn't work ...
    n_tap = int(int(fh.NumberOfSamplesPerChannel))

    recs = np.zeros((n_ch, n_tap))

    for i in range(n_ch):
        recs[i] = np.array(data['Channel_{}_Data'.format(i + 1)])

    # remove .mat suffix
    path = Path(fname)
    parent = Path(output_folder) if output_folder is not None else path.parent
    newpath = parent / path.stem

    np.savez(newpath, recs=recs, fs=fs, n_ch=n_ch, n_tap=n_tap)
    #import ipdb; ipdb.set_trace()


def folder_convert_TDRmat_recording_to_npz(
        path,
        output_folder=None,
):
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
    files = Path(path).glob('*.mat')
    for f in tqdm(list(files)):
        convert_TDRmat_recording_to_npz(f, output_folder=output_folder)


def transfer_functions_from_recordings(
        fp,
        n_ls,
        n_meas,
        fformat='Recording-{}.mat',
        n_avg=1,
        ref_ch=0,
        lowpass_lim=None,
        lowpass_butt=None,
        twindow=None,
        take_T=None,
        H_comp=None,
        T_comp=None,
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
    T_comp : None or float
        Length of H_comp in time domain in seconds.
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
    ndarray, shape (n_meas, n_ch - 1, n_ls, n_tap // 2 + 1)
        Transfer-functions. Possibly lowpass filtered
    """
    fpath = Path(fp)

    # read meta data from first recording
    # TODO: use load_recording funcs here
    fname = fpath / fformat.format(1)
    if fname.suffix == '.mat':
        fs, n_tap, n_ch = header_info(fname)
    elif fname.suffix == '.npz':
        with np.load(fname) as data:
            fs = data['fs']
            n_tap = data['n_tap']
            n_ch = data['n_ch']
    else:
        raise ValueError('Neither mat nor npz file.')

    if take_T is not None:
        # only take first take_T seconds of impulse responses
        n_tap = int(np.ceil(take_T * fs))
    else:
        n_tap = int(n_tap / n_ls / n_avg)

    H = np.zeros((n_meas, n_ch - 1, n_ls, n_tap // 2 + 1), dtype=complex)

    for i in tqdm(np.arange(n_meas)):
        fname = fpath / fformat.format(i + 1)

        # load time domain recordings
        if Path(fname).suffix == '.mat':
            temp, fs = load_mat_recording(fname, n_ls=n_ls, n_avg=n_avg)
        elif Path(fname).suffix == '.npz':
            temp, fs = load_npz_recording(fname, n_ls=n_ls, n_avg=n_avg)
        else:
            raise ValueError()

        # compute transfer-function in time domain
        temp = multi_transfer_function(temp, ref_ch=ref_ch, ret_time=True)

        # exclude reference channel
        temp = temp[1:]

        if H_comp is not None:
            # time crop to same length as compensation filter
            n_comp = int(T_comp * fs)
            temp = temp[:, :, :n_comp]
            Temp = np.fft.rfft(temp)
            # apply compensation filter
            Temp *= H_comp[:, None, :]
            temp = np.fft.irfft(Temp, n=n_comp)

        if take_T is not None:
            # only take first take_T seconds
            temp = temp[:, :, :n_tap]
            if twindow is None:
                # time window the tail
                nwin = int(round(take_T * 0.05 * fs))
                w = hann(2 * nwin)[nwin:]
                temp[:, :, -nwin:] *= w

        if lowpass_lim is not None:
            # filter out HF noise with zero phase frequency domain window
            temp = lowpass_by_frequency_domain_window(fs, temp, *lowpass_lim)

        if lowpass_butt is not None:
            # filter HF noise with butterworth
            order, cutoff = lowpass_butt
            b, a = butter(order, cutoff / (fs / 2), 'low')
            temp = lfilter(b, a, temp, axis=-1)

        if twindow is not None:
            T = time_window(fs, n_tap, *twindow)
            temp *= T

        H[i] = np.fft.rfft(temp)

    return H, fs, n_tap


def cut_recording(fname, cuts, names=None, remove_orig=False, outfolder=None):
    """Cut recordings at samples.
    """
    data = np.load(fname)
    recs = np.split(data['recs'], cuts, axis=-1)
    path = Path(fname)

    for i, r in enumerate(recs):
        add = '-{}'.format(names[i] if names is not None else i)
        parent = Path(outfolder) if outfolder is not None else path.parent
        newpath = parent / (path.stem + add)
        np.savez(
            newpath, recs=r, fs=data['fs'], n_ch=r.shape[0], n_tap=r.shape[1])

    data.close()

    if remove_orig:
        # remove original files
        path.unlink()


"""NON_LINEAR"""


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
       A. Farina, “Simultaneous measurement of impulse response and distortion
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
    r = np.roll(r, -np.argmax(np.abs(r)**2))

    # delays of non-linear components
    dts = exponential_sweep_harmonic_delay(T, fs, orders)
    dns = np.round((T - dts) * fs).astype(int)
    dns[0] = n

    e = np.zeros(order)

    # fundamental
    n_start = int(round((dns[1] + dns[0]) / 2))
    n_end = int(round((dns[-1] / 2)))
    e[0] = np.sum(np.abs(r[n_start:])**2)
    e[0] += np.sum(np.abs(r[:n_end])**2)

    # higher order
    for i in orders[:-2]:
        n_start = int(round((dns[i + 1] + dns[i]) / 2))
        n_end = int(round((dns[i] + dns[i - 1]) / 2))
        e[i] = np.sum(np.abs(r[n_start:n_end])**2)

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
    X_windowed = np.fft.rfft(x, axis=axis)
    X_windowed = np.moveaxis(X_windowed, axis, 0)
    X_windowed[windowed_samples] = (
        X_windowed[windowed_samples].T * half_window.T).T  # broadcasting
    X_windowed[stop:] = 0
    X_windowed = np.moveaxis(X_windowed, 0, axis)

    return np.fft.irfft(X_windowed, n=n)


def time_window(fs, n, startwindow, stopwindow, window='hann'):
    """Create a time window.
    """

    times = time_vector(n, fs)
    twindow = np.ones(n)

    if startwindow is not None:
        # start window
        samples = [find_nearest(times, t)[1] for t in startwindow]
        length = samples[1] - samples[0]
        w = get_window(window, 2 * length, fftbins=False)[:length]
        twindow[:samples[0]] = 0
        twindow[samples[0]:samples[1]] = w

    if stopwindow is not None:
        # stop window
        samples = [find_nearest(times, t)[1] for t in stopwindow]
        length = samples[1] - samples[0]
        w = get_window(window, 2 * length, fftbins=False)[length:]
        twindow[samples[0] + 1:samples[1] + 1] = w
        twindow[samples[1] + 1:] = 0

    return twindow


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


"""FFT FUNCS"""


def frequency_vector(n, fs, sided='single'):
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
    if sided == 'single':
        f = np.arange(n // 2 + 1) * fs / n
    elif sided == 'double':
        f = np.arange(n) * fs / n
    else:
        raise ValueError('Invalid value for sided.')
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
    return np.arange(n) * T


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
