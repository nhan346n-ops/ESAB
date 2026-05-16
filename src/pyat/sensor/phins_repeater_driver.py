import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from pyat.sensor.nmea_parser import ChecksumError, NMEASentence, ParseError


def read_phins_repeater_as_df(phins_repeater_file: str) -> pd.DataFrame:
    """
    Reads a phins repeater file and convert it to time-indexed Dataframe.
    :param phins_repeater_file: input phins repeater file containing nmea sentences
    :return: time-indexed Dataframe with decoded phins data
    """
    # Get file starting date and time from filename
    date_phins, time_phins = get_phins_repeater_date(phins_repeater_file)

    # Create an empty list to hold parsed cycles
    data_cycles = []
    data_cycle = {}

    with open(phins_repeater_file, encoding="utf-8") as file:
        first_cycle_nmea_msg_type = None
        last_cycle_nmea_msg_type = None
        # Iterate through the NMEA lines
        for line in file.readlines():
            try:
                # try to parse NMEA data
                nmea_data = NMEASentence.parse(line)
                if not first_cycle_nmea_msg_type:
                    # store first NMEA msg id to further detect new cycle beginning
                    first_cycle_nmea_msg_type = nmea_data["type"]
            except (ParseError, ChecksumError, NotImplementedError):
                continue

            # if new cycle begins
            if nmea_data["type"] == first_cycle_nmea_msg_type and data_cycle:
                # and current cycle is timestamped, store it
                if "time" in data_cycle.keys():
                    data_cycles.append(data_cycle)
                # save last cycle NMEA messge type to further detect new cycle ending
                if not last_cycle_nmea_msg_type:
                    last_cycle_nmea_msg_type = last_nmea_msg_type
                # finaly start a new data cycle
                data_cycle = {}

            # save last read NMEA message type
            last_nmea_msg_type = nmea_data["type"]
            # remove "type" key and add data to current cycle
            nmea_data.pop("type")
            data_cycle.update(nmea_data)

            # or check if current cycle ends here (might occur in case of missing first_cycle_nmea_msg_type)
            if last_cycle_nmea_msg_type is not None and last_nmea_msg_type == last_cycle_nmea_msg_type and data_cycle:
                if "time" in data_cycle.keys():
                    # if current cycle is timestamped, store it
                    data_cycles.append(data_cycle)
                # finaly start a new data cycle
                data_cycle = {}

        # Add the last cycle
        if data_cycle:
            data_cycles.append(data_cycle)

        # Converts the list of data cycles into a DataFrame
        phins_repeater_df = pd.DataFrame(data_cycles)

        # we must get rid off every data before the first "UTC_time_in" occurence
        if "UTC_time_in" not in phins_repeater_df.columns or "time" not in phins_repeater_df.columns:
            raise IOError(
                f"File is not a valid 'phins repeater' file: {phins_repeater_file}\n"
                f"\tNo 'UTCIN_' or 'TIME' NMEA sentence found : impossible to date data"
            )

        # Slice DataFrame from that index onward
        first_utc_time_in_idx = phins_repeater_df["UTC_time_in"].first_valid_index()
        if first_utc_time_in_idx is not None:  # in case the "UTC_time_in" column is entirely missing
            phins_repeater_df = phins_repeater_df.loc[first_utc_time_in_idx:].reset_index(drop=True)
        # store 1st UTC time in as starting time
        time_phins = phins_repeater_df["UTC_time_in"].iloc[0]

        # Combine date and time data
        is_time_cols = phins_repeater_df.columns.to_series().str.contains("time")
        time_cols = phins_repeater_df.columns[is_time_cols]

        datetime_combiner = DateTimeCombiner(
            start_date=date_phins, start_time=time_phins, n_time_cols=sum(is_time_cols)
        )

        # Apply the combiner method to the DataFrame
        phins_repeater_df[time_cols] = phins_repeater_df[time_cols].apply(datetime_combiner.combine, axis=1, raw=True)

        # return a sorted datetime-indexed dataframe
        phins_repeater_df.set_index("time", inplace=True)

        return phins_repeater_df.sort_index()


def get_phins_repeater_date(phins_repeater_file: str) -> Tuple[date, time]:
    """
    Returns date of phins data deduced from the filename.
    Raises IO error if filename doesn't comply with pattern "PHINS_REPEATER_YYYY-MM-DD_HH-mm-SS_XXX" or "PHINS_REPEATER_DDMMYYYY_HHmmSS".
    """
    # regex to match filename pattern and extract date and time :
    pattern = re.compile(
        r"^PHINS_REPEATER_(?:"
        r"(?P<date1>\d{4}-\d{2}-\d{2})_"  # YYYY-MM-DD
        r"(?P<time1>\d{2}-\d{2}-\d{2})_\w+"  # HH-mm-SS + suffix
        r"|"  # or
        r"(?P<date2>\d{2}\d{2}\d{4})_"  # DDMMYYYY
        r"(?P<time2>\d{6})"  # HHmmSS
        r")$"
    )
    match = pattern.match(Path(phins_repeater_file).stem)
    if not match:
        raise IOError(
            f"Not a 'phins repeater' file: {phins_repeater_file}\n"
            f"\tFile name pattern must be 'PHINS_REPEATER_YYYY-MM-DD_HH-mm-SS_XXX' or 'PHINS_REPEATER_DDMMYYYY_HHmmSS'"
        )

    if match.group("date1") and match.group("time1"):
        return (date.fromisoformat(match.group("date1")), time.fromisoformat(match.group("time1").replace("-", ":")))
    else:  # match.group("date2") and match.group("time2"):
        return (
            datetime.strptime(match.group("date2"), "%d%m%Y").date(),
            datetime.strptime(match.group("time2"), "%H%M%S").time(),
        )


class DateTimeCombiner:
    def __init__(self, start_date: date, start_time: time, n_time_cols: int):
        # Uses properties (last_times and last_dates) to keep track of the last date and time,
        # allowing it to maintain state between calls to combine() thus handling date incrementation properly
        self.last_dates = np.repeat(start_date, n_time_cols)
        self.last_times = np.repeat(start_time, n_time_cols)

    def combine(self, time_values: np.ndarray):
        """
        Combines date, read from filename, with time, read from file content, into a datetime with date incrementation logic.
        """
        current_datetimes = np.empty_like(time_values)

        # loop on each time-like columns in current row
        for idx, (current_time, last_time, last_date) in enumerate(zip(time_values, self.last_times, self.last_dates)):
            if isinstance(current_time, time) or not np.isnan(current_time):
                # Combine last date with the current time
                current_datetimes[idx] = datetime.combine(last_date, current_time)
                # retrieve last datetime
                last_datetime = datetime.combine(last_date, last_time)

                if current_time < last_time:
                    stepback = last_datetime - current_datetimes[idx]
                    # if the stepback is more than a minute, we have crossed midnight
                    if stepback > timedelta(minutes=1):
                        # Increment the date if current time is less than the last time
                        current_datetimes[idx] += timedelta(days=1)
                        # and update the corresponding last date for the next call to combine()
                        self.last_dates[idx] += timedelta(days=1)

                # Update last times for the next call to combine()
                self.last_times[idx] = current_time
            else:
                # Convert NaN, i.e. missing, times values to NaT
                current_datetimes[idx] = pd.NaT

        return current_datetimes
