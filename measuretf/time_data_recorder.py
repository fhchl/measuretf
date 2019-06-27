"""Handle data from B&K Time Data Recorder."""

import numpy as np

from response import Response
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import hann, butter, lfilter

from measuretf import multi_transfer_function
from measuretf.filtering import lowpass_by_frequency_domain_window, time_window
from measuretf.utils import tqdm


def load_bk_mat_recording(fname, n_ls=1, n_avg=1, fullout=False):
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


def load_bk_npz_recording(fname, n_ls=1, n_avg=1, fullout=False):
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


def load_bk_recording(fname, n_ls=1, n_avg=1, fullout=False):
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
        recs, fs, n_ch, n_tap = load_bk_mat_recording(
            fname, n_ls=n_ls, n_avg=n_avg, fullout=True
        )
    elif fname.suffix == ".npz":
        recs, fs, n_ch, n_tap = load_bk_npz_recording(
            fname, n_ls=n_ls, n_avg=n_avg, fullout=True
        )

    if fullout:
        return recs, fs, n_ch, n_tap

    return recs, fs


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
    _, fs, n_ch, n_tap = load_bk_recording(fname, n_ls=n_ls, n_avg=n_avg, fullout=True)

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
        temp, fs = load_bk_recording(fname, n_ls=n_ls, n_avg=n_avg)
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
