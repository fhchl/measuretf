"""Load data from mat or npz recording files."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from measuretf.utils import plot_rec


def load_recording(fname, n_out=1, n_avg=1, cut=True):
    """Load recording and split according to number of sources and averages.

    Parameters
    ----------
    fname : str
        Name of the mat file. Can also pass open file-like object. Reference
        channel is assumed to be recorded in first channel.
    n_out : int, optional
        Number of simultaneously, in series recorded output channels.
    n_avg : int, optional
        Number of recorded averages.

    Returns
    -------
    recs : ndarray, shape (n_in, n_out, n_avg, n_tap)
        Recordings, sliced into averages and output channels.
    fs : int
        Sampling frequency.

    """
    fname = Path(fname)

    if fname.suffix == '.npz':
        with np.load(fname, allow_pickle=True) as data:
            fs = data["fs"]
            orecs = data["recs"]  # has shape n_in x n_tap
    else:
        orecs, fs = sf.read(str(fname))  # sf.read needs str

    if orecs.ndim == 1:
        orecs = orecs[None]

    if orecs.ndim == 2:
        if cut:
            n_in, n_otap = orecs.shape
            n_tap = n_otap / n_out / n_avg
            if n_tap.is_integer():
                n_tap = int(n_tap)
            else:
                raise ValueError("Can't split recording: n_tap is not an integer")

            recs = np.zeros((n_in, n_out, n_avg, n_tap))
            for i in range(n_in):
                # shape  (ntaps*n_avg*n_out, ) -> (n_out, ntaps*n_avg)
                temp = np.array(np.split(orecs[i], n_out))
                temp = np.array(np.split(temp, n_avg, axis=-1))  # (n_avg, n_out, n_taps)
                temp = np.moveaxis(temp, 0, 1)  # (n_out, n_avg, n_taps)
                recs[i] = temp
        else:
            recs = orecs[..., 0][None, None, None, :]

    elif orecs.ndim == 3:
        recs = orecs[None]
    elif orecs.ndim == 4:
        recs = orecs
    else:
        raise ValueError(f"orces.ndim == {orecs.ndim}!")

    return recs, fs


def convert_wav_to_recording(
    wavfname,
    recname,
    ref_ch,
    n_out,
    add_datetime_to_name=False,
    description=None,
    plot=False,
):
    """Convert WAV recording to npz recording. Split for fit.

    Parameters
    ----------
    wavfname : str or path
        Path to WAV file.
    recname : str or path
        Save at this path.
    ref_ch : int
        Channel of inputs that is reference.
    n_out : int
        Number of separate output channels that were recorded in this one WAV file.
        The WAV file will be split equally into n_out segments.
    add_datetime_to_name : bool, optional
        Description
    description : None, optional
        Description
    plot : bool, optional
        Description

    """
    recs, fs = sf.read(wavfname, always_2d=True)
    recs = recs.T  # (n_ch, n_samples) -> (n_samples, n_ch)

    # remove samples for clean split
    remove_samples = recs.shape[0] % n_out
    if remove_samples > 0:
        print("removing samples:", remove_samples, "nt:", recs.shape[0])
        recs = recs[:-remove_samples, :]

    if plot:
        fig = plot_rec(fs, recs.T)
        fig.suptitle("before split")

    recs = np.stack(np.split(recs.T, n_out, axis=1), axis=0)

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
