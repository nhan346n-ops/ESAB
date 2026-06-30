"""
Helpers to handle coordinates in various formats, including
Degrees, Degrees and Minutes, Degrees Minutes and Seconds, and XY/UTM.
"""

import math
import re
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numba
import numpy as np
import pyproj
from pyproj import Transformer

from pyat.utils.proj_utils import lon_lat_to_utm_proj4

LON_LAT_PROJECTION_UID = "epsg:4326"
DEFAULT_ROUNDING = 7


def _converter(
    xs: List[Union[float, str]], ys: List[Union[float, str]], *, trans, rounding
) -> Tuple[List[float], List[float]]:
    "Return a converter of two lists of lon/x lat/y using given transformation function"
    outs = trans(zip(map(float, xs), map(float, ys)))  # NB: transformer returns longitude before latitude
    nxs, nys = zip(*outs)

    def rounder(v):
        return round(v, rounding)

    return tuple(map(rounder, nxs)), tuple(map(rounder, nys))


def create_lonlat_to_xy_converter(proj: Optional[str] = None, rounding: int = DEFAULT_ROUNDING) -> Callable:
    "Return a function able to convert a given lon/lat DEGREES coordinate to X/Y in minutes"
    proj = proj or lon_lat_to_utm_proj4(0, 0)
    trans = Transformer.from_crs(LON_LAT_PROJECTION_UID, proj, always_xy=True).itransform
    return partial(_converter, trans=trans, rounding=rounding)


def reprojection_converter(
    in_proj: Optional[str], out_proj: Optional[str], rounding: int = DEFAULT_ROUNDING
) -> Callable:
    "Return a function able to convert a given UTM/LONLAT coordinate to another, with another projection"
    in_proj = in_proj or lon_lat_to_utm_proj4(0, 0)
    out_proj = out_proj or lon_lat_to_utm_proj4(0, 0)
    trans = Transformer.from_crs(in_proj, out_proj, always_xy=True).itransform
    return partial(_converter, trans=trans, rounding=rounding)


def create_xy_to_lonlat_converter(proj: Optional[str] = None, rounding: int = DEFAULT_ROUNDING) -> Callable:
    "Return a function able to convert a given lat/lon DEGREES coordinate to X/Y in minutes"
    proj = proj or lon_lat_to_utm_proj4(0, 0)
    trans = Transformer.from_crs(proj, LON_LAT_PROJECTION_UID, always_xy=True).itransform
    return partial(_converter, trans=trans, rounding=rounding)


def DEGREES_from_wildDEGREES(dd: str) -> str:
    """Return canonical representation of given decimal coordinates.

    >>> DEGREES_from_wildDEGREES("-180°")
    '-180.0'
    >>> DEGREES_from_wildDEGREES("180")
    '180.0'

    """
    dd = str(dd).replace("°", " ").strip()
    if "." not in dd:
        dd += ".0"
    return dd


def isfloat(string: str) -> bool:
    if string.count(".") == 0:
        return string.isdigit()
    elif string.count(".") == 1:
        a, b = string.split(".")
        if a.startswith("-"):
            a = a[1:]
        return a.isdigit() and b.isdigit()
    return False


def DEGREES_from_DEG_MIN_DEC(ddm: str) -> float:
    """Return decimal representation of DEG_MIN_DEC (degree decimal minutes)

    >>> DEGREES_from_DEG_MIN_DEC("45° 17,896' N")
    45.29826666666666
    >>> DEGREES_from_DEG_MIN_DEC("-45° 17,896' N")
    -45.29826666666666
    >>> DEGREES_from_DEG_MIN_DEC("-45° 17,896' S")
    45.29826666666666

    """
    ddm = re.sub(r"[°']", " ", ddm).replace(",", ".")
    sign = -1 if re.search("[swSW]", ddm) else 1
    # numbers = [*filter(len, re.split(r'\D+', ddm, maxsplit=4))]
    numbers = [s for s in map(str.strip, ddm.split()) if s and isfloat(s.lstrip("-"))]
    assert len(numbers) in range(2, 4)

    degree = int(numbers[0])
    minute_decimal = float(numbers[1])
    sign *= -1 if degree < 0 else 1

    return sign * (abs(degree) + minute_decimal / 60)


def DEGREES_from_DEG_MIN_SEC(dms: str) -> float:
    """Return decimal representation of DEG_MIN_SEC (degree minutes seconds)"""
    dms = re.sub(r"[°'\"]", " ", dms).replace(",", ".")
    sign = 1
    dms_split = dms.split()
    assert len(dms_split) in range(3, 5)
    if len(dms_split) == 3:
        D, M, S = dms_split
    else:  # len(dms_split) == 4:
        D, M, S, W = dms_split
        if W in "SWsw":  # swap direction
            sign = -1
    return sign * (int(D) + float(M) / 60 + float(S) / 3600)


def DEG_MIN_DEC_STRING_from_DEGREES(dd: Union[str, float]) -> str:
    dd = float(dd)
    degrees = int(dd)
    minutes = abs(dd - degrees) * 60
    return f"{degrees}:{minutes}"


def DEG_MIN_SEC_STRING_from_DEGREES(dd: Union[str, float], seconds_digits=3) -> str:
    dd = float(dd)
    minutes, seconds = divmod(dd * 3600, 60)
    degrees, minutes = divmod(minutes, 60)
    degrees = int(degrees)
    minutes = int(minutes)
    seconds = round(seconds, ndigits=seconds_digits)
    if seconds == 60:  # rounding bring seconds to 6°
        minutes = minutes + 1
        seconds = 0
    if minutes == 60:
        degrees = degrees + 1
        minutes = 0
    sign = ""
    return f"{degrees}:{minutes}:{seconds}"


def DEG_MIN_SEC_from_DEGREES(dd: Union[str, float]) -> Tuple[int, int, float, bool]:
    dd = float(dd)
    negative = dd < 0
    dd = abs(dd)
    minutes, seconds = divmod(dd * 3600, 60)
    degrees, minutes = divmod(minutes, 60)
    return int(degrees), int(minutes), seconds, negative


def DEG_MIN_DEC_from_DEGREES(dd: Union[str, float]):
    degrees = int(str(dd).split(".", maxsplit=1)[0]) if "." in str(dd) else int(dd)
    decimals = float("0." + str(dd).split(".")[1]) if "." in str(dd) else 0.0
    decimals *= 60
    return abs(degrees), decimals, degrees < 0


def formatted_coordinates(
    lon: str, lat: str, fmt: str, x: Optional[str] = None, y: Optional[str] = None, rounding: int = DEFAULT_ROUNDING
) -> Tuple[str, str]:
    """Format given lonlat in degrees coordinates following the given format.

    Available formats:

        {D} Degrees (integer, positive)
        {M} Minutes (integer)
        {S} Seconds (float)
        {B} Degrees (integer, signed)
        {d} Degrees (float, positive)
        {b} Degrees (float, signed)
        {p} Degrees (float, positive, padded to have the integer part fit 3 characters (or 2 for latitudes))
        {P} Degrees (integer, positive, padded to have the integer part fit 3 characters (or 2 for latitudes))
        {m} Minutes (float)
        {s} Sign marker (a dash or +)
        {w} Direction (S, N, W or E)
        {x} UTM with given projection

    >>> formatted_coordinates(-19.9128, -77.508333, '{d}')
    ('19.9128', '77.508333')
    >>> formatted_coordinates(-19, -77.508333, '{D}')
    ('19', '77')
    >>> formatted_coordinates(-19.9128, -77.508333, '{b}')
    ('-19.9128', '-77.508333')
    >>> formatted_coordinates(164.754167, -77.508333, '{D} {M} {S} {w}')
    ('164 45 20.0 W', '77 30 29.9988 S')
    >>> formatted_coordinates(164.754167, -77.508333, "{x}")
    ('-11324361.6', '933813.46')

    """
    assert isinstance(fmt, str), type(fmt)
    if "x" in fmt:
        if x is None or y is None:
            dd_to_xy = create_lonlat_to_xy_converter()
            utm_xs, utm_ys = dd_to_xy([lon], [lat])
            utm_x, utm_y = utm_xs[0], utm_ys[0]
        else:
            log = f"{lat}, {x}, {lon}, {y}"
            assert x is not None, log
            assert y is not None, log
            utm_x, utm_y = x, y

    def compute_formats(dd: float, is_longitude: bool) -> dict:
        V = {"b": dd, "d": str(dd).lstrip("-")}
        negative = float(dd) < 0.0
        if "D" in fmt or "B" in fmt or "M" in fmt or "S" in fmt or "P" in fmt:
            V["D"], V["M"], V["S"], negative_alt = DEG_MIN_SEC_from_DEGREES(dd)
            assert negative is negative_alt
        if "m" in fmt:
            Dalt, V["m"], negative_alt = DEG_MIN_DEC_from_DEGREES(dd)
            if "D" in V:
                assert Dalt == V["D"]
            if negative is not None:
                assert negative_alt == negative
        if "B" in fmt:
            V["B"] = V["D"] * (-1 if negative else 1)
        if "w" in fmt:
            V["w"] = ("W" if negative else "E") if is_longitude else ("S" if negative else "N")
        if "s" in fmt:
            V["s"] = "-" if negative else "+"
        if "x" in fmt:
            V["x"] = utm_x if is_longitude else utm_y
        if "p" in fmt:
            V["p"] = float(f"{{:{3 if is_longitude else 2}.{rounding}f}}".format(float(V["d"])))
        if "P" in fmt:
            V["P"] = f"{{:{3 if is_longitude else 2}d}}".format(V["D"])
        if rounding:
            if "S" in V:
                V["S"] = round(float(V["S"]), int(rounding))
                if str(V["S"]).endswith(".0"):
                    V["S"] = int(V["S"])
            if "D" in V:
                if str(V["D"]).endswith(".0"):
                    V["D"] = int(V["D"])
            if "m" in V:
                V["m"] = round(float(V["m"]), int(rounding))
            if "x" in V:
                V["x"] = round(float(V["x"]), int(rounding))
        return V

    return fmt.format(**compute_formats(lon, True)), fmt.format(**compute_formats(lat, False))


def compute_norm_and_radius(latNav: float, eccentricity: float, semi_major_axis: float):
    sinLN2 = math.pow(math.sin(math.radians(latNav)), 2)
    e2MoinssinLN2 = 1.0 - eccentricity * sinLN2
    norm = semi_major_axis / math.sqrt(e2MoinssinLN2)
    radius = norm * (1.0 - eccentricity) / e2MoinssinLN2
    return (norm, radius)


def compute_norm(lats: np.ndarray, semi_major_axis: float, eccentricity2: float) -> np.ndarray:
    """computes norm given an array of latitudes and ellipsoid parameters"""
    return semi_major_axis / np.sqrt(1.0 - eccentricity2 * np.sin(np.radians(lats)) ** 2)


def compute_norms_and_radii(
    lats: np.ndarray, semi_major_axis: float, eccentricity2: float
) -> tuple[np.ndarray, np.ndarray]:
    """computes norms and radii given an array of latitudes and ellipsoid squared eccentricity"""
    norm = compute_norm(lats, semi_major_axis, eccentricity2)
    radius = norm * (1.0 - eccentricity2) / (1.0 - (eccentricity2 * (np.sin(np.radians(lats)) ** 2)))
    return norm, radius


def compute_detection_position(
    along: np.ndarray, across: np.ndarray, nav_longitude: float, nav_latitude: float, heading: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute position (lon/lat) for some sounding detections."""
    semi_major_axis = 6378137.0
    semi_minor_axis = 6356752.314245179
    eccentricity = 1.0 - (semi_minor_axis / semi_major_axis) * (semi_minor_axis / semi_major_axis)
    norm, radius = compute_norm_and_radius(nav_latitude, eccentricity, semi_major_axis)

    sin_heading = math.sin(math.radians(heading))
    cos_heading = math.cos(math.radians(heading))

    lon = (
        np.degrees(along * sin_heading + across * cos_heading) / norm / math.cos(math.radians(nav_latitude))
        + nav_longitude
    )
    lat = np.degrees(along * cos_heading - across * sin_heading) / radius + nav_latitude

    return (lon, lat)


def compute_distance(longitudes: np.ndarray, latitudes: np.ndarray):
    """
    Returns an array with the distance between two consecutive positions
    """
    shifted_longitudes, shifted_latitudes = np.empty_like(longitudes), np.empty_like(latitudes)
    shifted_longitudes[1:] = longitudes[:-1]
    shifted_longitudes[0] = longitudes[0]
    shifted_latitudes[1:] = latitudes[:-1]
    shifted_latitudes[0] = latitudes[0]

    geodesic = pyproj.Geod(ellps="WGS84")
    _, _, distances = geodesic.inv(longitudes, latitudes, shifted_longitudes, shifted_latitudes)
    return distances


@numba.njit(cache=True, fastmath=True)
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the distance between two georeferenced points using the Haversine formula.

    Args:
        lat1 (float): Latitude of the first point (in degrees).
        lon1 (float): Longitude of the first point (in degrees).
        lat2 (float): Latitude of the second point (in degrees).
        lon2 (float): Longitude of the second point (in degrees).

    Returns:
        float: Distance between the two points in meters.
    """
    # Earth's average radius in meters
    R = 6371000.0

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Differences in latitude and longitude
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    distance = R * c

    return distance
