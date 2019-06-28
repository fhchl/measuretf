"""Handle data from B&K Time Data Recorder."""

import numpy as np

from response import Response
from scipy.io import loadmat
from pathlib import Path

from measuretf.utils import tqdm
from measuretf.io import load_recording as load_npz_recording


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
    """Load multichannel B&K Time Data Recording into ndarray.

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
