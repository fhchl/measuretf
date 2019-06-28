"""Utility functions."""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import correlate
from response import Response

from measuretf.fft import time_vector


def is_ipynb():
    """Check if environment is jupyter notebook."""
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
    """Time align y to match x."""
    n = x.size

    # delta time array to match xcorr
    s = np.arange(1 - n, n)

    # cross correlation
    xcorr = correlate(x, y, mode="full")

    # estimate delay in time
    dt = s[xcorr.argmax()] / fs

    y = Response.from_time(fs, y).delay(dt, keep_length=True).in_time

    return y, dt


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
