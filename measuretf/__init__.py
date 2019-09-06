"""Collection of functions for transfer-function measurements."""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from response import Response
from scipy.signal import tukey, csd, welch, butter, hann, lfilter, coherence

from measuretf.utils import time_align
from measuretf.filtering import (
    freq_window,
    time_window,
    lowpass_by_frequency_domain_window,
)
from measuretf.io import load_recording
from measuretf.utils import tqdm


def transfer_function(
    ref,
    meas,
    ret_time=True,
    axis=-1,
    Ywindow=None,
    fftwindow=None,
    reg=0,
    reg_lim_dB=None,
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
        Transfer-function between ref and meas.

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
            f"TF gains larger than {20*np.log10(TOO_LARGE_GAIN):.0f} dB. Setting to 0"
        )
        Y[too_large] = 0

    if reg_lim_dB is not None:
        # maximum of reference
        maxRdB = np.max(20 * np.log10(np.abs(R)), axis=axis)

        # power in reference should be at least
        minRdB = maxRdB - reg_lim_dB

        # 10 * log10(reg + |R|**2) = minRdB
        reg = 10 ** (minRdB / 10) - np.abs(R) ** 2
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
                    reg_lim_dB=reg_lim_dB,
                )

    return tfs


def transfer_function_csd(
    x,
    y,
    fs,
    compensate_delay=True,
    compensate_range=None,
    reg=0,
    return_dt=False,
    **kwargs,
):
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

    if compensate_delay is not None:
        x, y, dt = time_align(x, y, fs, trange=compensate_range)

    f, S_xy = csd(x, y, fs=fs, **kwargs)
    _, S_xx = welch(x, fs=fs, **kwargs)

    H = S_xy / (S_xx + reg)

    if compensate_delay:
        # reintroduce delay
        H *= np.exp(-1j * 2 * np.pi * f * dt)

    if return_dt:
        return f, H, dt

    return f, H


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
        temp, fs = load_recording(fname, n_ls=n_ls, n_avg=n_avg)
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


def coherence_csd(x, y, fs, compensate_delay=True, **csd_kwargs):
    """Estimate maginitude squared coherence of two signals using Welch's method.

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
    Cxy : ndarray
        Magnitude squared coherence

    """
    assert x.ndim == 1
    assert y.ndim == 1
    assert x.size == y.size

    if compensate_delay:
        x, y, _ = time_align(x, y, fs)

    f, Cxy = coherence(x, y, fs=fs, **csd_kwargs)

    return f, Cxy


def coherence_matrix(x, fs=1, nperseg=256, compensate_delay=True, **csd_kwargs):
    """Estimate coherence matrix between series.

    Parameters
    ----------
    x : array_like
        A 2-D array containing multiple variables and observations. Each row of x
        represents a variable, and each column a single observation of all those
        variables.
    fs : float
        Samplerate
    compensate_delay : bool, optional
        If `True`, time align variables before cross-correlation.
    **csd_kwargs
        Kwargs passed to `sipy.signal.coherence`.

    """
    x = np.atleast_2d(x)
    C = np.ones((x.shape[0], x.shape[0], nperseg // 2 + 1))
    for i in range(x.shape[0]):
        for j in range(i, x.shape[0]):
            f, Cij = coherence_csd(
                x[i],
                x[j],
                fs=fs,
                compensate_delay=compensate_delay,
                nperseg=nperseg,
                **csd_kwargs,
            )
            C[i, j] = C[j, i] = Cij

    return f, C


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


def cut_recording(
    fname, cuts, names=None, remove_orig=False, outfolder=None, shift_before_cut=0
):
    """Cut recordings at samples."""
    with np.load(fname, allow_pickle=True) as data:
        recs = data["recs"]

        if shift_before_cut:
            recs = np.roll(recs, shift_before_cut, axis=-1)

        # split
        recs = np.split(recs, cuts, axis=-1)
        sounds = np.split(data["sound"].T, cuts, axis=-1)

        # split also the times
        datetime_start = data["datetime_start"]
        datetime_end = data["datetime_end"]
        duration = (datetime_end - datetime_start) / cuts

        path = Path(fname)
        for i, rs in enumerate(zip(recs, sounds)):

            add = "-{}".format(names[i] if names is not None else i)
            parent = Path(outfolder) if outfolder is not None else path.parent
            newpath = parent / (path.stem + add)

            split_start = datetime_start + i * duration
            split_end = datetime_start + (i + 1) * duration

            r, s = rs
            np.savez(
                newpath,
                recs=r,
                fs=data["fs"],
                n_ch=r.shape[0],
                n_tap=r.shape[1],
                sound=s,
                ref_ch=data["ref_ch"],
                datetime_start=split_start,
                datetime_end=split_end,
            )

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
    **sd_kwargs,
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
    """Record the excitation through a single output at multiple inputs.

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


def recordsave_single_output_excitation(
    filename,
    sound,
    fs,
    out_ch,
    in_ch,
    ref_ch=None,
    add_datetime_to_name=False,
    description="",
    **sd_kwargs,
):
    """Record and save a single sound (multichannel) excitation.

    Parameters
    ----------
    filename : str or Path
        Save recodring at this path.
    sound : ndarray, shape (nt, n_out)
        Excitation signal
    fs : int
        Sampling rate of sound
    out_ch : int or list
        Output channels
    in_ch : int or list
        Input channels
    ref_ch : None, optional
        Mark this input channel as the reference channel
    add_datetime_to_name : bool, optional
        Add a datetime to file name.
    description : str, optional
        Add a description to file.
    sd_kwargs : dict, optional
        Keyword arguments to `sounddevice.playrec`.

    Returns
    -------
    ndarray, shape (n_out, n_in, n_avg, n_t)
        Recorded signals.

    """
    start = datetime.now()
    recs = record_single_output_excitation(
        sound, fs, out_ch=out_ch, in_ch=in_ch, **sd_kwargs
    )
    end = datetime.now()

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
        datetime_start=start,
        datetime_end=end,
        description=description,
    )

    return recs


def record_multi_output_excitation(sound, fs, out_ch, in_ch, n_avg=1, **sd_kwargs):
    """Record the excitation of multiple outputs at multiple inputs.

    Parameters
    ----------
    sound : ndarray, shape (nt,)
        Excitation signal
    fs : int
        Sampling rate of sound
    out_ch : int or list
        Output channels
    in_ch : int or list
        Input channels
    n_avg : int, optinal
        Number of averages

    Returns
    -------
    ndarray, shape (n_out, n_in, n_avg, n_tap)
        Recoded signals.

    """
    out_ch = np.atleast_1d(out_ch)
    in_ch = np.atleast_1d(in_ch)

    recs = np.zeros((len(out_ch), len(in_ch), n_avg, len(sound)))
    for o, oc in enumerate(out_ch):
        for avg in range(n_avg):
            recs[o, :, avg, :] = record_single_output_excitation(
                sound, fs, out_ch=oc, in_ch=in_ch, **sd_kwargs
            )
    return recs


def recordsave_multi_output_excitation(
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
    filename : str or Path
        Save recodring at this path.
    sound : ndarray, shape (nt,)
        Excitation signal
    fs : int
        Sampling rate of sound
    out_ch : int or list
        Output channels
    in_ch : int or list
        Input channels
    n_avg : int, optinal
        Number of averages
    ref_ch : None, optional
        Mark this input channel as the reference channel
    add_datetime_to_name : bool, optional
        Add a datetime to file name.
    description : str, optional
        Add a description to file.
    sd_kwargs : dict, optional
        Keyword arguments to `sounddevice.playrec`.

    Returns
    -------
    ndarray, shape (n_out, n_in, n_avg, n_t)
        Recorded signals.

    """
    start = datetime.now()
    recs = record_multi_output_excitation(
        sound, fs, out_ch, in_ch, n_avg=n_avg, **sd_kwargs
    )
    end = datetime.now()

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
        datetime_end=end,
        description=description,
    )

    return recs


def recordsave_recording(
    filename,
    fs,
    in_ch,
    T,
    ref_ch=None,
    add_datetime_to_name=False,
    description="",
    **sd_kwargs,
):
    """Record and save signals.

    Parameters
    ----------
    filename : str or Path
        Save recodring at this path.
    fs : int
        Sampling rate of sound.
    in_ch : int or list
        Input channels.
    T : float
        Recording time in seconds.
    ref_ch : None, optional
        Mark this input channel as the reference channel
    add_datetime_to_name : bool, optional
        Add a datetime to file name.
    description : str, optional
        Add a description to file.
    sd_kwargs : dict, optional
        Keyword arguments to `sounddevice.playrec`.

    Returns
    -------
    ndarray, shape (n_in, n_t)
        Recorded signals.

    """
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
    average=False,
    return_time=False,
):
    with np.load(fname, allow_pickle=True) as data:
        recs = data["recs"]  # shape (n_out, n_in, navg, nt)
        fs = int(data["fs"])
        sound = data["sound"]
        ref_ch = data["ref_ch"]
        start = np.datetime64(data["datetime_start"].item())
        end = data["datetime_end"]

    start = np.datetime64(start.item())
    end = np.datetime64(end.item())

    # accept legancy format without reps (n_out, n_in, nt)
    if recs.ndim == 3:
        recs = recs[:, :, None, :]

    if average:
        recs = recs.mean(axis=2)[:, :, None, :]

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
        recs, ref=ref, Ywindow=fwindow, fftwindow=fftwindow, reg_lim_dB=reg_lim_dB
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

    if return_time:
        n_out, _, n_avg, _ = recs.shape
        timespan = end - start
        time_per_out = timespan / n_out
        time_per_avg = time_per_out / n_avg

        time = np.zeros((n_out, n_avg), dtype="datetime64[us]")
        for i in range(n_out):
            timeout = start + i * time_per_out
            for j in range(n_avg):
                time[i, j] = timeout + (j + 0.5) * time_per_avg

        return fs, irs, time

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


# FILTERING
