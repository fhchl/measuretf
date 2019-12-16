"""Utility functions."""

import matplotlib.pyplot as plt
import numpy as np
from measuretf.fft import time_vector
from response import Response
from scipy.signal import correlate


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


def time_align(x, y, fs, trange=None):
    """Time align two signals, zeropad as necessary.

    If `dt` is positive `x` was delayed and `y` zeropadded for same length.
    """
    assert len(x) == len(y)
    n = x.size

    # cross correlation
    xcorr = correlate(y, x, mode="full")

    # delta time array to match xcorr
    t = np.arange(1 - n, n) / fs

    if trange is not None:
        idx = np.logical_and(trange[0] <= t, t <= trange[1])
        t = t[idx]
        xcorr = xcorr[idx]

    # estimate delay in time
    dt = t[xcorr.argmax()]

    # match both responses in time and length
    if dt >= 0:
        x = Response.from_time(fs, x).delay(dt, keep_length=False).in_time
        y = Response.from_time(fs, y).zeropad_to_length(x.size).in_time
    else:
        y = Response.from_time(fs, y).delay(-dt, keep_length=False).in_time
        x = Response.from_time(fs, x).zeropad_to_length(y.size).in_time

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


def window_nd(a, window, steps=None, axis=None):
    """
    Create a windowed view over `n`-dimensional input that uses an
    `m`-dimensional window, with `m <= n`

    Parameters
    -------------
    a : Array-like
        The array to create the view on
    window : tuple or int
        If int, the size of the window in `axis`, or in all dimensions if
        `axis == None`

        If tuple, the shape of the desired window.  `window.size` must be:
            equal to `len(axis)` if `axis != None`, else
            equal to `len(a.shape)`, or
            1
    steps : tuple, int or None
        The offset between consecutive windows in desired dimension
        If None, offset is one in all dimensions
        If int, the offset for all windows over `axis`
        If tuple, the steps along each `axis`.
            `len(steps)` must me equal to `len(axis)`
    axis : tuple, int or None
        The axes over which to apply the window
        If None, apply over all dimensions
        if tuple or int, the dimensions over which to apply the window

    Returns
    -------
    a_view : ndarray
        A windowed view on the input array `a`, or copied list of windows

    Notes
    -----
    Source: https://stackoverflow.com/questions/45960192/using-numpy-as-strided-function-to-create-patches-tiles-rolling-or-sliding-w

    """
    ashp = np.array(a.shape)

    if axis != None:
        axs = np.array(axis, ndmin=1)
        axs[axs < 0] = a.ndim + axs[axs < 0]  # convert negative axes to positive axes
        assert np.all(np.in1d(axs, np.arange(ashp.size))), "Axes out of range"
    else:
        axs = np.arange(ashp.size)

    window = np.array(window, ndmin=1)
    assert (window.size == axs.size) | (window.size == 1), "Window dims and axes don't match"
    wshp = ashp.copy()
    wshp[axs] = window
    assert np.all(wshp <= ashp), "Window is bigger than input array in axes"

    stp = np.ones_like(ashp)
    if steps:
        steps = np.array(steps, ndmin=1)
        assert np.all(steps > 0), "Only positive steps allowed"
        assert (steps.size == axs.size) | (steps.size == 1), "Steps and axes don't match"
        stp[axs] = steps

    astr = np.array(a.strides)

    shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
    strides = tuple(astr * stp) + tuple(astr)

    as_strided = np.lib.stride_tricks.as_strided
    a_view = np.squeeze(as_strided(a, shape=shape, strides=strides))
    return a_view


def covariance(X, Y, axis=0, ddof=0):
    """Compute sample covariance."""
    mu_X = X.mean(axis=axis)
    mu_Y = X.mean(axis=axis)
    cov = 1 / (X.shape[axis] - ddof) * np.sum((X - mu_X) * (Y - mu_Y).conj(), axis=axis)
    return cov
