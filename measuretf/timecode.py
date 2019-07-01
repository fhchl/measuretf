# Align and Sync
"""
Created on Thu Jun 20 23:18:41 2019

@author: minson
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime, timezone
from scipy import signal

import matplotlib.pyplot as plt
import warnings
from joblib import Parallel, delayed

from tqdm import tqdm


def timestamps_from_irigb(irigb, fs):
    """Find time stamps in irig-B signal.

    Parameters
    ----------
    irigb : ndarray
        IRIG-B signal.
    fs : int
        Sample rate if IRIG-B signal.

    Returns
    -------
    list
        List of (timestamp, index) tuples.

    """
    # preprocess: turn into binary signal
    # TODO: remove low frequency components?
    positive = irigb >= 0
    irigb[positive] = 1
    irigb[np.logical_not(positive)] = 0

    # Compute pulse widths:
    x_edge = np.diff(irigb)

    # Indices of 1-seg start
    edge_up = np.where(x_edge == 1)[0] + 1  # 1 less than the true value

    # Indices of 0-seg start
    edge_dn = np.where(x_edge == -1)[0] + 1  # 1 less than the true value

    if edge_dn[0] < edge_up[0]:
        edge_dn = edge_dn[1:]
        if len(edge_dn) != len(edge_up):
            edge_up = edge_up[0 : len(edge_dn)]
    else:
        if len(edge_dn) != len(edge_up):
            edge_up = edge_up[:-1]

    pls_wdt = np.round((edge_dn - edge_up) / fs * 1000).astype(int)  # Pulse width in ms

    allowed_pulse_widths = reference_width, ones_width, zeros_width = (8, 5, 2)

    pls_wdt_binary = pls_wdt.copy()
    pls_wdt_binary[pls_wdt_binary == ones_width] = 1
    pls_wdt_binary[pls_wdt_binary == zeros_width] = 0

    # Decoding the Time Code (TC)
    timestamps = []  # list(array1, array2)...

    for ii in range(0, len(pls_wdt) - 58):  # Consider "Incomplete last second"
        if pls_wdt[ii] == reference_width and pls_wdt[ii + 1] == reference_width:
            # start sequence detected

            if not np.all(np.isin(pls_wdt[ii : ii + 60], allowed_pulse_widths)):
                warnings.warn(
                    f"Could not decode timestamp at {ii / 100}s: invalid pulse widths"
                )
                # timestamps.append((None, edge_up[ii + 1]))
                continue
            elif not np.all(
                [pls_wdt[ii + i] == reference_width for i in [10, 20, 30, 40, 50, 60]]
            ):
                warnings.warn(
                    f"Could not decode timestamp at {ii / 100}s: marker bits not set correctly"
                )
                # timestamps.append((None, edge_up[ii + 1]))
                continue

            Second = (
                1 * pls_wdt_binary[ii + 2]
                + 2 * pls_wdt_binary[ii + 3]
                + 4 * pls_wdt_binary[ii + 4]
                + 8 * pls_wdt_binary[ii + 5]
                + 10 * pls_wdt_binary[ii + 7]
                + 20 * pls_wdt_binary[ii + 8]
                + 40 * pls_wdt_binary[ii + 9]
            )
            Minute = (
                1 * pls_wdt_binary[ii + 11]
                + 2 * pls_wdt_binary[ii + 12]
                + 4 * pls_wdt_binary[ii + 13]
                + 8 * pls_wdt_binary[ii + 14]
                + 10 * pls_wdt_binary[ii + 16]
                + 20 * pls_wdt_binary[ii + 17]
                + 40 * pls_wdt_binary[ii + 18]
            )
            # Hours can differ by device setting. UTC is default
            Hour = (
                1 * pls_wdt_binary[ii + 21]
                + 2 * pls_wdt_binary[ii + 22]
                + 4 * pls_wdt_binary[ii + 23]
                + 8 * pls_wdt_binary[ii + 24]
                + 10 * pls_wdt_binary[ii + 26]
                + 20 * pls_wdt_binary[ii + 27]
            )
            # Day: nth day in this year
            Day = (
                1 * pls_wdt_binary[ii + 31]
                + 2 * pls_wdt_binary[ii + 32]
                + 4 * pls_wdt_binary[ii + 33]
                + 8 * pls_wdt_binary[ii + 34]
                + 10 * pls_wdt_binary[ii + 36]
                + 20 * pls_wdt_binary[ii + 37]
                + 40 * pls_wdt_binary[ii + 38]
                + 80 * pls_wdt_binary[ii + 39]
                + 100 * pls_wdt_binary[ii + 41]
                + 200 * pls_wdt_binary[ii + 42]
            )
            # Year: nth year from 2000
            Year = 2000 + (
                1 * pls_wdt_binary[ii + 51]
                + 2 * pls_wdt_binary[ii + 52]
                + 4 * pls_wdt_binary[ii + 53]
                + 8 * pls_wdt_binary[ii + 54]
                + 10 * pls_wdt_binary[ii + 56]
                + 20 * pls_wdt_binary[ii + 57]
                + 40 * pls_wdt_binary[ii + 58]
                + 80 * pls_wdt_binary[ii + 59]
            )

            # LeapYear?
            if Year % 100 == 0:  # century year
                if Year % 400 == 0:
                    #              J  F  M  A  M  J  J  A  S  O  N  D
                    DaysInMonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                else:
                    #              J  F  M  A  M  J  J  A  S  O  N  D
                    DaysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            else:
                if Year % 4 == 0:
                    #              J  F  M  A  M  J  J  A  S  O  N  D
                    DaysInMonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                else:
                    #              J  F  M  A  M  J  J  A  S  O  N  D
                    DaysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

            # Which month?
            rdays = Day
            for Month in range(1, 12):
                rdays = rdays - DaysInMonth[Month - 1]
                if rdays <= 0:
                    Date = rdays + DaysInMonth[Month - 1]
                    break

            dt = datetime(Year, Month, Date, Hour, Minute, Second, tzinfo=timezone.utc)
            timestamps.append((dt, edge_up[ii + 1]))

    return timestamps


def tc2tstamp(folder_name):
    """Read TimeCode recording *tc.wav from the given folder,
    Extracts time code in YYYY-MM-DDThh:mm:ssZ format.
    Save as .npy: [(1st sec, sample #), (2nd sec, sample #), ...]

    Parameters
    ----------
    folder_name : folder with fullpath, ex) 'D:/ASFC/Roskilde/darkzone'

    save
    -------
    saves .npy file
    filename: startT_endT_tc.npy
    """
    folder = Path(folder_name)
    # Check whether the folder actually exists. Warn User if not
    if not folder.is_dir():
        raise FileNotFoundError(folder_name)

    tc_file = next(folder.glob("*tc.wav"))
    print(f"Reading {tc_file}")

    irigb, fs = sf.read(str(tc_file))

    timestamps = timestamps_from_irigb(irigb, fs)

    # Save TC
    min_time = timestamps[0][0]  # Start time of the files
    max_time = timestamps[-1][0]  # End time of the files
    timestamps_filename = (
        min_time.isoformat() + "_" + max_time.isoformat() + "_timestamps"
    )
    timestamps_path = Path(folder_name, timestamps_filename)
    np.savez(timestamps_path, timestamps=timestamps, samplerate=fs)
    print(f"Time code saved at {timestamps_path}")

    return timestamps


def sync_to_timestamps(
    x, timestamps, samplerate, start_time=None, end_time=None, out=None
):
    """Synchronize signal according to timestamps.

    Parameters
    ----------
    x : ndarray
        Signal.
    timestamps : list
        List of (timestamp, index) tuples.
    samplerate : int
        Samplerate of output

    Returns
    -------
    out : ndarray
        Signal syncronized to `timestamps` and resampled to `samplerate`.

    """
    # beginning and end of time code recording
    min_time = timestamps[0][0]
    max_time = timestamps[-1][0]

    if start_time is None:
        start_time = min_time
    if end_time is None:
        end_time = max_time

    if not (min_time <= start_time and start_time < end_time and end_time <= max_time):
        raise ValueError("Invalid start or end times.")

    # filter time stamps according to user input times
    filtered_timestamps = timestamps[
        np.logical_and(start_time <= timestamps[:, 0], timestamps[:, 0] <= end_time)
    ]

    curser = 0
    if out is None:
        out = np.zeros(int((end_time - start_time).total_seconds()) * samplerate)

    for j in tqdm(range(len(filtered_timestamps) - 1)):
        chunk_start_time, chunk_start_sample = filtered_timestamps[j]
        chunk_end_time, chunk_end_sample = filtered_timestamps[j + 1]

        chunk = x[chunk_start_sample:chunk_end_sample]
        chunk_seconds = (chunk_end_time - chunk_start_time).total_seconds()

        if len(chunk) != samplerate:
            chunk = signal.resample(chunk, int(chunk_seconds) * samplerate)
        else:
            chunk = chunk

        out[curser : curser + len(chunk)] = chunk
        curser += len(chunk)

    return out


def sync_folder_to_timestamps(
    folder_name,
    start_time,
    end_time,
    resamplerate=48000,
    tzinfo=timezone.utc,
    out_folder=None,
    n_jobs=1
):
    """Read multiple wav recordings *rc.wav from the given folder,
    Slice files for given interval (in time).
    Save as .npz

    Parameters
    ----------
    folder_name : folder with fullpath, ex) 'D:/ASFC/Roskilde/darkzone'

    save
    -------
    saves rc_files, namelist_tc, audio chunks matrix as .npz file
    filename: startT_endT_aligned.npz

    """
    folder = Path(folder_name)
    if not folder.is_dir():
        raise FileNotFoundError(folder_name)

    if out_folder is not None:
        out_folder = Path(out_folder)
        if not out_folder.is_dir():
            raise FileNotFoundError(out_folder)
    else:
        out_folder = folder

    # load timestamps
    timestamps_file = next(folder.glob("*_timestamps.npz"))
    with np.load(timestamps_file, allow_pickle=True) as data:
        timestamps = data["timestamps"]
        samplerate = data["samplerate"]

    if resamplerate:
        samplerate = resamplerate

    print(f"Using {timestamps_file}")

    # beginning and end of time code recording
    start_time = datetime.fromisoformat(start_time).replace(tzinfo=tzinfo)
    end_time = datetime.fromisoformat(end_time).replace(tzinfo=tzinfo)

    rc_files = sorted(list(folder.glob("*rc.wav")))

    def syncer(rc_file):
        print(f"Resyncing {rc_file}...")

        # Read Audio
        y, _ = sf.read(str(rc_file))
        synced = sync_to_timestamps(y, timestamps, samplerate, start_time, end_time)
        del y

        out_name = rc_file.stem + "_synced" + rc_file.suffix
        print(f"Writing {out_name}...")
        sf.write(str(out_folder / out_name), synced, samplerate)

    if n_jobs > 1:
        Parallel(n_jobs=n_jobs)(delayed(syncer)(rc_file) for rc_file in rc_files)
    else:
        for i, rc_file in enumerate(rc_files):
            syncer(rc_file)
