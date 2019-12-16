#!/usr/bin/env python3

import argparse
from datetime import datetime
from pathlib import Path

import measuretf as mtf
import measuretf.timecode


def sync_folders(paths, start_datetime=None, end_datetime=None):
    """Create timestamps and sync to time."""
    if not isinstance(paths, list):
        paths = [paths]

    paths = [Path(path) for path in paths]

    for folder_name in paths:
        folder_index = int(folder_name.stem[-1])
        timecode = mtf.timecode.tc2tstamp(folder_name)

        if start_datetime is None or end_datetime is None:
            start_datetime = timecode[0][0]
            end_datetime = timecode[-1][0]

        print(
            "Using start and end times:",
            start_datetime.isoformat(),
            end_datetime.isoformat(),
        )
        mtf.timecode.sync_folder_to_timestamps(
            folder_name,
            start_datetime,
            end_datetime,
            out_folder=folder_name.parent / f"synced_{folder_index}",
            n_jobs=2,
        )


if __name__ == "__main__":

    # Instantiate the parser
    parser = argparse.ArgumentParser(description="Sync wav files in folder to timecode")

    # Optional positional argument
    parser.add_argument(
        "paths", type=str, nargs="+", help="At least one path to folder"
    )

    # Optional positional argument
    parser.add_argument(
        "-s",
        "--start_datetime",
        type=str,
        help="Start datetime as isoformat string, e.g. '2019-07-04T12:12:47'",
    )

    # Optional positional argument
    parser.add_argument(
        "-e",
        "--end_datetime",
        type=str,
        help="End datetime as isoformat string, e.g. '2019-07-04T12:12:47'",
    )

    args = parser.parse_args()

    try:
        print(args.start_datetime)
        start_datetime = args.start_datetime
        if start_datetime is not None:
            start_datetime = datetime.fromisoformat(start_datetime)

        end_datetime = args.end_datetime
        if end_datetime is not None:
            end_datetime = datetime.fromisoformat(args.end_datetime)
    except Exception as e:
        parser.error(str(e))

    sync_folders(args.paths, start_datetime, end_datetime)
