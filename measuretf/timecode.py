# Align and Sync
"""
Created on Thu Jun 20 23:18:41 2019

@author: minson
"""

import sys
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
from datetime import datetime
from scipy import signal
import os
import fnmatch


def resync(folder_name, start_time, end_time):
    """Read multiple wav recordings *rc.wav from the given folder,
    Slice files for given interval (in time).
    Save as .npz

    Parameters
    ----------
    folder_name : folder with fullpath, ex) 'D:/ASFC/Roskilde/darkzone'

    save
    -------
    saves namelist_rc, namelist_tc, audio chunks matrix as .npz file
    filename: startT_endT_aligned.npz

    """

    # Check whether the folder actually exists. Warn User if not
    if os.path.isdir(folder_name) is False:
        errMsg = "Error: The following folder does not exist:{}".format(folder_name)
        sys.exit(errMsg)

    # Get Recordings from the folder
    listOfFiles = os.listdir(folder_name)

    pattern_rc = "*rc.wav"  # find file with *tc.wav
    namelist_rc = []
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern_rc):
            namelist_rc.append(entry)

    # Q. what if multiple tc inside a single folder?
    pattern_tc = "*tc.npy"  # find TC file with *tc.npy
    namelist_tc = []
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern_tc):
            namelist_tc.append(entry)
            fname_tc = Path(folder_name, entry)
            TC_list = np.load(fname_tc)
            print("Time-stamp file {} loaded.".format(entry))

    minT = TC_list[0][0]
    maxT = TC_list[-1][0]
    minT_Tobj = datetime.strptime(minT, "%Y-%m-%dT%H:%M:%SZ")
    maxT_Tobj = datetime.strptime(maxT, "%Y-%m-%dT%H:%M:%SZ")
    ST_Tobj = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%SZ")
    ET_Tobj = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%SZ")
    num_minT = int(datetime.timestamp(minT_Tobj))
    num_maxT = int(datetime.timestamp(maxT_Tobj))
    num_ST = int(datetime.timestamp(ST_Tobj))
    num_ET = int(datetime.timestamp(ET_Tobj))

    if (
        (num_ST > num_ET)
        or (num_minT > num_ST)
        or (num_ST >= num_maxT)
        or (num_minT >= num_ET)
        or (num_ET >= num_maxT)
    ):
        errMsg2 = "Invalid start_time or end_time"
        sys.exit(errMsg2)

    # Find matching sample indices
    nth_arr_ST = np.where(TC_list == start_time)[0][0]  # should be in np array
    nth_arr_ET = np.where(TC_list == end_time)[0][0]

    #    idx_ST = int(TC_list[nth_arr_ST][1])
    #    idx_ET = int(TC_list[nth_arr_ET][1])

    recs_resampled = np.zeros((48000 * (nth_arr_ET - nth_arr_ST), len(namelist_rc)))
    for k in range(len(namelist_rc)):
        temp_name = namelist_rc[k]
        temp_fname = Path(folder_name, temp_name)
        print("Now reading audio file {}\n.".format(temp_name))

        # Read Audio
        fs, yy = wavfile.read(temp_fname)
        yy_resampled = np.zeros((0, 1))

        for j in range(nth_arr_ST, nth_arr_ET):
            temp_idx = int(TC_list[j][1])
            temp_idx_last = int(TC_list[j + 1][1])
            temp_crop = yy[temp_idx:temp_idx_last]
            yy_crop_dn = signal.resample(temp_crop, 8000)
            yy_crop_up = signal.resample(yy_crop_dn, 48000)
            yy_resampled = np.append(yy_resampled, yy_crop_up)

        recs_resampled[:, k] = yy_resampled

    # Save audio chunks for given start_time, end_time
    #    ST_HHMMSSTddmmyyyy = ST_Tobj.strftime("%H%M%ST%d%m%Y")  # ST
    #    ET_HHMMSSTddmmyyyy = ET_Tobj.strftime("%H%M%ST%d%m%Y")  # ET
    #    dirname = folder_name
    #    RC_filename = ST_HHMMSSTddmmyyyy + "_" + ET_HHMMSSTddmmyyyy + "_aligned"
    #    suffix = ".npz"
    #    RC_fullname = Path(dirname, RC_filename).with_suffix(suffix)
    #    np.savez(RC_fullname, namelist_rc, namelist_tc, recs_resampled)
    #    print("Recordings aligned, {} saved.\n".format(RC_fullname))

    return namelist_rc, namelist_tc, recs_resampled


# Execution Code
# folder_name = "D:/ASFC/Roskilde/darkzone/trash/test"
# out1 = resync(folder_name, "2019-06-13T12:44:40Z", "2019-06-13T12:44:42Z")


# Decode IRIG-B signal
# Minho Song, Technical University of Denmark, DK




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

    # Check whether the folder actually exists. Warn User if not
    if os.path.isdir(folder_name) is False:
        errMsg = "Error: The following folder does not exist:{}".format(folder_name)
        sys.exit(errMsg)

    # Get Time Code recording from the folder
    listOfFiles = os.listdir(folder_name)

    # Q. what if multiple tc inside a single folder?
    pattern = "*tc.wav"  # find file with *tc.wav
    namelist = []
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            namelist.append(entry)
            print("Now reading {}\n".format(entry))

    # Read TC in wav file
    data_folder = Path(folder_name)
    file_FullName = data_folder / namelist[0]

    fs, irigB = wavfile.read(file_FullName)
    irigB_Z = irigB / 32768  # Raw TC
    irigB_Z_clean = np.ceil(irigB_Z)  # Raw TC to clean Pulse
    x_edge = np.diff(irigB_Z_clean)

    # Indices of 1-seg start from irigB_Z_clean
    edge_up = np.where(x_edge == 1)[0]  # 1 less than the true value
    edge_up = [x + 1 for x in edge_up]

    # Indices of 0-seg start from irigB_Z_clean
    edge_dn = np.where(x_edge == -1)[0]  # 1 less than the true value
    edge_dn = [x + 1 for x in edge_dn]

    if edge_dn[0] < edge_up[0]:
        edge_dn = edge_dn[1:]
        if len(edge_dn) != len(edge_up):
            edge_up = edge_up[0 : len(edge_dn)]
    else:
        if len(edge_dn) != len(edge_up):
            edge_up = edge_up[:-1]

    pls_wdt = np.subtract(edge_dn, edge_up) / fs  # Pulse width in s
    pls_wdt_sft = [y - 0.003 for y in pls_wdt]

    # Decoding the Time Code (TC)
    TC_list = []  # list(array1, array2)...

    for ii in range(0, len(pls_wdt) - 58):  # Consider "Incomplete last second"
        if pls_wdt[ii] > 0.0077 and pls_wdt[ii + 1] > 0.0077:

            Second = int(
                1 * np.ceil(pls_wdt_sft[ii + 2])
                + 2 * np.ceil(pls_wdt_sft[ii + 3])
                + 4 * np.ceil(pls_wdt_sft[ii + 4])
                + 8 * np.ceil(pls_wdt_sft[ii + 5])
                + 10 * np.ceil(pls_wdt_sft[ii + 7])
                + 20 * np.ceil(pls_wdt_sft[ii + 8])
                + 40 * np.ceil(pls_wdt_sft[ii + 9])
            )
            Minute = int(
                1 * np.ceil(pls_wdt_sft[ii + 11])
                + 2 * np.ceil(pls_wdt_sft[ii + 12])
                + 4 * np.ceil(pls_wdt_sft[ii + 13])
                + 8 * np.ceil(pls_wdt_sft[ii + 14])
                + 10 * np.ceil(pls_wdt_sft[ii + 16])
                + 20 * np.ceil(pls_wdt_sft[ii + 17])
                + 40 * np.ceil(pls_wdt_sft[ii + 18])
            )
            # Hours can differ by device setting. UTC is default
            Hour = int(
                1 * np.ceil(pls_wdt_sft[ii + 21])
                + 2 * np.ceil(pls_wdt_sft[ii + 22])
                + 4 * np.ceil(pls_wdt_sft[ii + 23])
                + 8 * np.ceil(pls_wdt_sft[ii + 24])
                + 10 * np.ceil(pls_wdt_sft[ii + 26])
                + 20 * np.ceil(pls_wdt_sft[ii + 27])
            )
            # Day: nth day in this year
            Day = int(
                1 * np.ceil(pls_wdt_sft[ii + 31])
                + 2 * np.ceil(pls_wdt_sft[ii + 32])
                + 4 * np.ceil(pls_wdt_sft[ii + 33])
                + 8 * np.ceil(pls_wdt_sft[ii + 34])
                + 10 * np.ceil(pls_wdt_sft[ii + 36])
                + 20 * np.ceil(pls_wdt_sft[ii + 37])
                + 40 * np.ceil(pls_wdt_sft[ii + 38])
                + 80 * np.ceil(pls_wdt_sft[ii + 39])
                + 100 * np.ceil(pls_wdt_sft[ii + 41])
                + 200 * np.ceil(pls_wdt_sft[ii + 42])
            )
            # Year: nth year from 2000
            Year = int(
                1 * np.ceil(pls_wdt_sft[ii + 51])
                + 2 * np.ceil(pls_wdt_sft[ii + 52])
                + 4 * np.ceil(pls_wdt_sft[ii + 53])
                + 8 * np.ceil(pls_wdt_sft[ii + 54])
                + 10 * np.ceil(pls_wdt_sft[ii + 56])
                + 20 * np.ceil(pls_wdt_sft[ii + 57])
                + 40 * np.ceil(pls_wdt_sft[ii + 58])
                + 80 * np.ceil(pls_wdt_sft[ii + 59])
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

            # Our time code is as YYYY-MM-DDTHH:MM:SSZ
            time = "{:d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}Z".format(
                2000 + Year, Month, Date, Hour, Minute, Second
            )

            TC_list.append((time, edge_up[ii + 1]))

    # Save TC
    minT = TC_list[0][0]  # Start time of the files
    min_yyyymmddTHHMMSS = datetime.strptime(minT, "%Y-%m-%dT%H:%M:%SZ")
    min_HHMMSSTddmmyyyy = min_yyyymmddTHHMMSS.strftime("%H%M%ST%d%m%Y")
    maxT = TC_list[-1][0]  # End time of the files
    max_yyyymmddTHHMMSS = datetime.strptime(maxT, "%Y-%m-%dT%H:%M:%SZ")
    max_HHMMSSTddmmyyyy = max_yyyymmddTHHMMSS.strftime("%H%M%ST%d%m%Y")
    dirname = folder_name
    TC_filename = min_HHMMSSTddmmyyyy + "_" + max_HHMMSSTddmmyyyy + "_tc"
    suffix = ".npy"
    TC_fn = Path(dirname, TC_filename).with_suffix(suffix)
    # TC_fn_nPath = TC_fn.replace(os.sep, '/')
    np.save(TC_fn, TC_list)
    print("TimeCode {} saved.\n".format(TC_fn))
    return namelist, TC_fn, TC_list


# Test Code
# folder_name = "D:/ASFC/Roskilde/darkzone/simple_validation/out_sync/dark"
# out = tc2time(folder_name)  # array
