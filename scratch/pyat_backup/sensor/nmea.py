import datetime
from enum import Enum
from typing import Callable, List, NamedTuple

import numpy as np


class GPSQualityIndicator(Enum):
    """
    Enum for all gps quality indicator data, extracted from nmea specification GPS Quality Indicator
    """

    NOT_AVAILABLE = 0
    GPS_FIX = 1
    DGPS = 2
    PPS = 3
    RTK = 4
    FLOAT_RTK = 5
    ESTIMATED = 6
    MANUAL = 7
    SIMULATION = 8


class NMEADataFormat(NamedTuple):
    """
    Defines a base NMEA data format.
    """

    fieldname: List[str]  # data field names
    desc: List[str]  # data descritption
    read_formatter: List[Callable[[str], any] | None]  # data field read function


def to_float(x: str) -> float:
    """
    Converts NMEA data string to float.
    """
    return float(x) if x else np.nan


def to_int(x: str) -> int:
    """
    Converts NMEA data string to int.
    """
    return int(x) if x else np.nan


def to_time(x: str) -> datetime.time:
    """
    Converts NMEA data string to time.
    """
    return datetime.datetime.strptime(x, "%H%M%S.%f").time()


def has_subtypes(nmea_sentence_format) -> bool:
    """
    Checks if nmea sentence format has subtypes.
    """
    return not isinstance(nmea_sentence_format, NMEADataFormat)


class TalkerSentenceFormat(NMEADataFormat, Enum):
    """
    Defines NMEA 1083 standard Talker sentence data format.
    """

    HDT = NMEADataFormat(["heading", None], ["Heading Degrees, true", None], [to_float, None])


class IXSE(NMEADataFormat, Enum):
    """
    Defines IXSE subtypes (EXAIL PHINS standard) proprietary NMEA data formats.
    """

    ATITUD = NMEADataFormat(["roll", "pitch"], ["roll (degrees)", "pitch (degrees)"], [to_float, to_float])
    POSITI = NMEADataFormat(
        ["latitude", "longitude", "altitude"],
        ["latitude (degrees)", "longitude (degrees)", "altitude (meters)"],
        [to_float, to_float, to_float],
    )
    SPEED_ = NMEADataFormat(
        ["speed_east", "speed_north", "speed_up"],
        ["East speed (m/s)", "North speed (m/s)", "Up speed (m/s)"],
        [to_float, to_float, to_float],
    )
    UTMWGS = NMEADataFormat(
        ["UTM_hemisphere", "UTM_zone", "UTM_X", "UTM_Y", "UTM_Z"],
        [
            "UTM hemisphere",
            "UTM zone",
            "UTM east position (meters)",
            "UTM north position (meters)",
            "UTM altitude (meters)",
        ],
        [None, to_int, to_float, to_float, to_float],
    )
    HEAVE_ = NMEADataFormat(
        ["surge", "sway", "heave"],
        ["surge (meters)", "sway (meters)", "heave (meters)"],
        [to_float, to_float, to_float],
    )
    STDHRP = NMEADataFormat(
        ["heading_stdev", "roll_stdev", "pitch_stdev"],
        ["heading std dev (degrees)", "roll std dev (degrees)", "pitch std dev (degrees)"],
        [to_float, to_float, to_float],
    )
    STDPOS = NMEADataFormat(
        ["latitude_stdev", "longitude_stdev", "altitude_stdev"],
        ["latitude std dev (degrees)", "longitude std dev (degrees)", "altitude std dev (meters)"],
        [to_float, to_float, to_float],
    )
    STDSPD = NMEADataFormat(
        ["speed_east_stdev", "speed_north_stdev", "speed_up_stdev"],
        ["East speed std dev (m/s)", "North speed std dev (m/s)", "Up speed std dev (m/s)"],
        [to_float, to_float, to_float],
    )
    TIME__ = NMEADataFormat(["time"], ["data transmitted time (UTC)"], [to_time])
    # LOGIN_ : Last data received from the log bottom track sensor
    LOGIN_ = NMEADataFormat(
        ["DVL_ground_speed_x", "DVL_ground_speed_y", "DVL_ground speed_z", "DVL_hdng_delta", "DVL_ground_time"],
        [
            "longitudinal DVL speed (m/s)",
            "transverse DVL speed in (m/s)",
            "vertical DVL speed in (m/s)",
            "heading misalignment Kalman estimation (degrees)",
            "log data time (UTC)",
        ],
        [to_float, to_float, to_float, to_float, to_time],
    )
    # LOGDVL : Last raw data received from the log sensor
    LOGDVL = NMEADataFormat(
        ["DVL_ssp", "DVL_comp_ssp", "DVL_range"],
        [
            "DVL set sound velocity in water (m/s)",
            "DVL measured compensation sound velocity (m/s)",
            "DVL distance to bottom (meters)",
        ],
        [to_float, to_float, to_float],
    )
    # LOGWAT : Last data received from the log water track sensor
    LOGWAT = NMEADataFormat(
        [
            "DVL_water_speed_x",
            "DVL_water_speed_y",
            "DVL_water_speed_z",
            "current_speed_north",
            "current_speed_east",
            "current_speed_north_std",
            "current_speed_east_std",
            "DVL_water_time",
        ],
        [
            "longitudinal DVL speed (m/s)",
            "transverse DVL speed (m/s)",
            "vertical DVL speed (m/s)",
            "north current speed (m/s)",
            "east current speed (m/s)",
            "north current speed std dev (m/s)",
            "east current speed std dev (m/s)",
            "WT data time (UTC)",
        ],
        [to_float, to_float, to_float, to_float, to_float, to_float, to_float, to_time],
    )
    # GPSIN_ : Last data received from the GPS 1 sensor
    GPSIN_ = NMEADataFormat(
        ["GPS_latitude", "GPS_longitude", "GPS_altitude", "GPS_time", "GPS_quality"],
        ["GPS latitude (degrees)", "GPS longitude (degrees)", "GPS altitude (meters)", "GPS time (UTC)", "GPS quality"],
        [to_float, to_float, to_float, to_time, lambda x: GPSQualityIndicator(int(x))],
    )
    # DEPIN_ : Last data received from the depth sensor
    DEPIN_ = NMEADataFormat(["depth", "depth_time"], ["depth (meters)", "depth time (UTC)"], [to_float, to_time])
    # LMNIN_ : Last data received from the log EM sensor
    LMNIN_ = NMEADataFormat(
        [
            "LMN_speed_x",
            "LMN_current_speed_north",
            "LMN_current_speed_east",
            "LMN_current_speed_north_std",
            "LMN_current_speed_east_std",
            "LMN_time",
            None,
        ],
        [
            "longitudinal speed (m/s)",
            "north current speed (m/s)",
            "east current speed (m/s)",
            "north current speed std dev (m/s)",
            "east current speed std dev (m/s)",
            "EM log data time (UTC)",
            None,
        ],
        [to_float, to_float, to_float, to_float, to_float, to_time, None],
    )
    # UTCIN_ : Last UTC received
    UTCIN_ = NMEADataFormat(["UTC_time_in"], ["UTC time received (UTC)"], [to_time])

    # SORSTS : INS sensor status 1 and 2
    # Sensor status word is coded with 16 hexadecimal characters in the “$PIXSE,SORSTS,hhhhhhhh,llllllll” NMEA sentence. hhhhhhhh is the hexadecimal value of the first 32 Less Significant Bits (Sensor Status 1). llllllll is the hexadecimal value of the 32 Most Significant Bits (Sensor Status 2).
    SORSTS = NMEADataFormat(
        ["sensor_status_1", "sensor_status_2"],
        ["Sensor status word LSB (hexadecimal)", "Sensor status word MSB (hexadecimal)"],
        [str, str],
    )


# Proprietary data formats

ProprietarySentenceFormat = {
    "IXSE": IXSE,
    "TOTO": NMEADataFormat(["roll", "pitch"], ["roll in degrees", "pitch in degrees"], [to_float, to_float]),
}
