"""Load data from mat or npz recording files."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from datetime import datetime


def convert_wav_to_recording(
    wavfname,
    recname,
    ref_ch,
    n_out,
    add_datetime_to_name=False,
    description=None,
    plot=False,
):
    """Convert WAV recording to npz recording.

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
    fs, recs = wavfile.read(wavfname)

    if recs.ndim == 1:
        recs = np.atleast_2d(recs).T

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
