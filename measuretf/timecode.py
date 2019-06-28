import sys
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
from datetime import datetime, timezone
import os
import fnmatch


def tc2tstamp(path_to_folder):
    """Read TimeCode recording *tc.wav from the given folder,
    Extracts time code in YYYY-MM-DDThh:mm:ssZ format.
    Save as .npy: [(1st sec, sample #), (2nd sec, sample #), ...]

    Parameters
    ----------
    path_to_folder : folder with fullpath, ex) 'D:/ASFC/Roskilde/darkzone'

    save
    -------
    saves .npy file
    filename: startT_endT_tc.npy

    Note
    ----
    By Minho Sung
    """
    folder = Path(path_to_folder)
    # Check whether the folder actually exists. Warn User if not
    if not folder.is_dir():
        raise FileNotFoundError(folder)

    tc_file = list(folder.glob("*tc.wav"))[0]
    print(f"Time code file: {tc_file}")

    # Read TC in wav file
    fs, irigB = wavfile.read(tc_file)
    irigB_Z = irigB / 32768  # Raw TC
    irigB_Z_clean = np.ceil(irigB_Z)  # Raw TC to clean Pulse
    x_edge = np.diff(irigB_Z_clean)

    # Indices of 1-seg start from irigB_Z_clean
    edge_up = np.where(x_edge == 1)[0]  # 1 less than the true value
    edge_up += 1

    # Indices of 0-seg start from irigB_Z_clean
    edge_dn = np.where(x_edge == -1)[0]  # 1 less than the true value
    edge_dn += 1

    if edge_dn[0] < edge_up[0]:
        edge_dn = edge_dn[1:]
        if len(edge_dn) != len(edge_up):
            edge_up = edge_up[0 : len(edge_dn)]
    else:
        if len(edge_dn) != len(edge_up):
            edge_up = edge_up[:-1]

    pls_wdt = (edge_dn - edge_up) / fs  # Pulse width in s
    pls_wdt_sft = pls_wdt - 0.003

    # Decoding the Time Code (TC)
    TC_list = []  # list(array1, array2)...

    for i in range(0, len(pls_wdt) - 58):  # Consider "Incomplete last second"
        if pls_wdt[i] > 0.0077 and pls_wdt[i + 1] > 0.0077:

            second = int(
                1 * np.ceil(pls_wdt_sft[i + 2])
                + 2 * np.ceil(pls_wdt_sft[i + 3])
                + 4 * np.ceil(pls_wdt_sft[i + 4])
                + 8 * np.ceil(pls_wdt_sft[i + 5])
                + 10 * np.ceil(pls_wdt_sft[i + 7])
                + 20 * np.ceil(pls_wdt_sft[i + 8])
                + 40 * np.ceil(pls_wdt_sft[i + 9])
            )
            minute = int(
                1 * np.ceil(pls_wdt_sft[i + 11])
                + 2 * np.ceil(pls_wdt_sft[i + 12])
                + 4 * np.ceil(pls_wdt_sft[i + 13])
                + 8 * np.ceil(pls_wdt_sft[i + 14])
                + 10 * np.ceil(pls_wdt_sft[i + 16])
                + 20 * np.ceil(pls_wdt_sft[i + 17])
                + 40 * np.ceil(pls_wdt_sft[i + 18])
            )
            # hours can differ by device setting. UTC is default
            hour = int(
                1 * np.ceil(pls_wdt_sft[i + 21])
                + 2 * np.ceil(pls_wdt_sft[i + 22])
                + 4 * np.ceil(pls_wdt_sft[i + 23])
                + 8 * np.ceil(pls_wdt_sft[i + 24])
                + 10 * np.ceil(pls_wdt_sft[i + 26])
                + 20 * np.ceil(pls_wdt_sft[i + 27])
            )
            # day: nth day in this year
            day = int(
                1 * np.ceil(pls_wdt_sft[i + 31])
                + 2 * np.ceil(pls_wdt_sft[i + 32])
                + 4 * np.ceil(pls_wdt_sft[i + 33])
                + 8 * np.ceil(pls_wdt_sft[i + 34])
                + 10 * np.ceil(pls_wdt_sft[i + 36])
                + 20 * np.ceil(pls_wdt_sft[i + 37])
                + 40 * np.ceil(pls_wdt_sft[i + 38])
                + 80 * np.ceil(pls_wdt_sft[i + 39])
                + 100 * np.ceil(pls_wdt_sft[i + 41])
                + 200 * np.ceil(pls_wdt_sft[i + 42])
            )
            # year: nth year from 2000
            year = int(
                1 * np.ceil(pls_wdt_sft[i + 51])
                + 2 * np.ceil(pls_wdt_sft[i + 52])
                + 4 * np.ceil(pls_wdt_sft[i + 53])
                + 8 * np.ceil(pls_wdt_sft[i + 54])
                + 10 * np.ceil(pls_wdt_sft[i + 56])
                + 20 * np.ceil(pls_wdt_sft[i + 57])
                + 40 * np.ceil(pls_wdt_sft[i + 58])
                + 80 * np.ceil(pls_wdt_sft[i + 59])
            )

            # Leapyear?
            if year % 100 == 0:  # century year
                if year % 400 == 0:
                    #              J  F  M  A  M  J  J  A  S  O  N  D
                    days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                else:
                    #              J  F  M  A  M  J  J  A  S  O  N  D
                    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            else:
                if year % 4 == 0:
                    #              J  F  M  A  M  J  J  A  S  O  N  D
                    days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                else:
                    #              J  F  M  A  M  J  J  A  S  O  N  D
                    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

            # Which month?
            rdays = day
            for month in range(1, 12):
                rdays = rdays - days_in_month[month - 1]
                if rdays <= 0:
                    date = rdays + days_in_month[month - 1]
                    break

            # Our time code is as YYYY-MM-DDTHH:MM:SSZ
            # time = "{:d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}Z".format(
            #     2000 + year, month, date, hour, minute, second
            # )
            time = datetime(2000 + year, month, date, hour, minute, second, tzinfo=timezone.utc)
            TC_list.append((time, edge_up[i + 1]))

    # Save TC
    minT = TC_list[0][0]  # Start time of the files
    maxT = TC_list[-1][0]  # End time of the files
    TC_filename = minT.isoformat() + "_" + maxT.isoformat() + "_tc"
    TC_fn = folder / TC_filename
    np.save(TC_fn, TC_list)
    print(f"Saving time stamp file: {TC_fn}")


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

    start_time = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%SZ")
    end_time = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%SZ")

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

    num_minT = int(minT.timestamp())
    num_maxT = int(maxT.timestamp())
    num_ST = int(start_time.timestamp())
    num_ET = int(end_time.timestamp())

    if (
        (num_ST > num_ET)
        or (num_minT > num_ST)
        or (num_ST >= num_maxT)
        or (num_minT >= num_ET)
        or (num_ET >= num_maxT)
    ):
        raise ValueError("Invalid start_time or end_time")

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
