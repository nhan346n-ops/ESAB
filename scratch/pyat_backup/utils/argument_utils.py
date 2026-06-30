#! /usr/bin/env python3
# coding: utf-8

import argparse
import datetime as dt
import errno
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy
from dateutil.parser import parse as dateutil_parse
from osgeo import osr

import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.utils.pyat_logger as log
from pyat.utils.type_conf import cdi, coord, filters, layers
from pyat.utils.example_action import ExampleAction
from pyat.common.geo_file import SR_WGS_84
from pyat.utils.coords import DEG_MIN_DEC_STRING_from_DEGREES, DEG_MIN_SEC_STRING_from_DEGREES
from pyat.utils.number_utils import normalize_latitude, normalize_longitude

osr.UseExceptions()


class Geobox:
    @property
    def spatial_reference(self):
        return self._spatial_reference

    @spatial_reference.setter
    def spatial_reference(self, spatial_reference: osr.SpatialReference):
        self._spatial_reference = spatial_reference

    def __init__(self, upper, lower, left, right, spatial_reference: osr.SpatialReference = SR_WGS_84):
        self.upper = upper
        self.lower = lower
        self.right = right
        self.left = left
        self.spatial_reference = spatial_reference

    def extend(self, upper, lower, left, right) -> None:
        self.upper = max(upper, self.upper)
        self.lower = min(lower, self.lower)
        self.right = max(right, self.right)
        self.left = min(left, self.left)

    def is_empty(self) -> bool:
        return self.upper == 0.0 and self.lower == 0.0 and self.right == 0.0 and self.left == 0.0

    def fix_if_180th_meridian(self):
        """
        Swap West and East if spanning the 180th meridian
        """
        if self.spatial_reference.IsGeographic() and self.left - self.right + 360.0 < self.right - self.left:
            self.right, self.left = self.left, self.right

    def expand_to_arcmin(self):
        """
        Expand the geobox, if in geographic coordinates to match an integer number of arcmin

        """
        if self.spatial_reference.IsGeographic():
            upper_arcmin = math.ceil(self.upper * 60) / 60  # swith to arcmin then back to degrees
            self.upper = min(90.0, upper_arcmin)
            lower_arcmin = math.floor(self.lower * 60) / 60
            self.lower = max(-90.0, lower_arcmin)
            self.right = math.ceil(self.right * 60) / 60
            self.left = math.floor(self.left * 60) / 60
            self.fix_if_180th_meridian()

    def realign(self, x_modulo: float = 1 / 60, y_modulo: float = 1 / 60):
        """
        Realigns the bounds so that they are an exact multiple of the given modulo
        By default, modulo is equals to one arcmin
        In case of projected Geobox, modulo is expressed in meters
        """
        self.upper = self.__upperBound(self.upper, y_modulo)
        self.lower = self.__lowerBound(self.lower, y_modulo)
        self.right = self.__upperBound(self.right, x_modulo)
        self.left = self.__lowerBound(self.left, x_modulo)
        if self.spatial_reference.IsGeographic():
            self.upper = min(90.0, self.upper)
            self.lower = max(-90.0, self.lower)
            self.right = self.right if self.right >= -180.0 else 360.0 + self.right
            self.left = self.left if self.left <= 180.0 else self.left - 360.0

    def __upperBound(self, value: float, modulo: float):
        valMod = value / modulo
        # Round when decimals are negligible
        return modulo * (math.ceil(valMod) if valMod % 1 > 1e-4 else round(valMod))

    def __lowerBound(self, value: float, modulo: float):
        valMod = value / modulo
        # Round when decimals are negligible
        return modulo * (math.floor(valMod) if valMod % 1 > 1e-4 else round(valMod))

    def is_spanning_180th_meridian(self) -> bool:
        """
        return true when SR is latlon and this geobox spans the 180th meridian
        """
        return self.spatial_reference.IsGeographic() and self.left > self.right

    def get_delta_x(self) -> float:
        """
        return the distance between right and left
        """
        if self.is_spanning_180th_meridian():
            return self.right - self.left + 360.0
        return abs(self.right - self.left)

    def get_delta_y(self) -> float:
        """
        return the distance between up and down
        """
        return abs(self.upper - self.lower)

    def normalize_degrees(self) -> None:
        """
        Normalize latitudes [-90, 90] and longitudes [-180, 180]
        """
        if self.spatial_reference.IsGeographic():
            self.upper = normalize_latitude(self.upper)
            self.lower = normalize_latitude(self.lower)
            self.left = normalize_longitude(self.left)
            self.right = normalize_longitude(self.right)

    def to_dict(self):
        return {
            "north": self.upper,
            "south": self.lower,
            "west": self.left,
            "east": self.right,
        }

    def __str__(self):
        return f"{self:DMD}"

    def __format__(self, format_spec: str):
        """
        Called by the format() built-in function, and by extension, evaluation of formatted string literals
        and the str.format() method, to produce a 'formatted' string representation of a Geobox.
        format_spec is one of the string DMS or DMD (default)
        """
        if self.spatial_reference.IsGeographic():
            formater = DEG_MIN_SEC_STRING_from_DEGREES if format_spec == "DMS" else DEG_MIN_DEC_STRING_from_DEGREES
            return f"Upper Left(E[{formater(self.left)}],N[{formater(self.upper)}])  Lower right(S[{formater(self.lower)}], W[{formater(self.right)}])"
        else:
            return f"Upper left( {self.upper} , {self.left} ) Lower right( {self.lower} , {self.right} )"


class GeoBoxBuilder:
    def __init__(self, spatial_reference: osr.SpatialReference):
        self.spatial_reference = spatial_reference

        # Min latitude of the resulting GeoBox
        self.min_y = math.inf
        # Max latitude of the resulting GeoBox
        self.max_y = -math.inf
        # Min longidude of the resulting GeoBox centered on 0th meridian
        self.min_x_0 = math.inf
        # Max longidude of the resulting GeoBox centered on 0th meridian
        self.max_x_0 = -math.inf
        # Min longidude of the resulting GeoBox centered on 180th meridian
        self.min_x_180 = math.inf
        # Max longidude of the resulting GeoBox centered on 180th meridian
        self.max_x_180 = -math.inf

    def add_lon_lat(self, longitude: float, latitude: float):
        """Add a point to the geobox"""
        self.min_y = min(self.min_y, latitude)
        self.max_y = max(self.max_y, latitude)

        self.min_x_0 = min(self.min_x_0, longitude)
        self.max_x_0 = max(self.max_x_0, longitude)

        if self.spatial_reference.IsGeographic():
            self.min_x_180 = min(self.min_x_180, longitude - 360.0 if longitude >= 0.0 else longitude)
            self.max_x_180 = max(self.min_x_180, longitude - 360.0 if longitude >= 0.0 else longitude)

    def add_lons_lats(self, longitudes: numpy.ndarray, latitudes: numpy.ndarray):
        """Add some points to the geobox"""
        self.min_y = min(self.min_y, numpy.nanmin(latitudes))
        self.max_y = max(self.max_y, numpy.nanmax(latitudes))

        self.min_x_0 = min(self.min_x_0, numpy.nanmin(longitudes))
        self.max_x_0 = max(self.max_x_0, numpy.nanmax(longitudes))

        if self.spatial_reference.IsGeographic():
            longidudes180 = numpy.where(longitudes >= 0.0, longitudes - 360.0, longitudes)
            self.min_x_180 = min(self.min_x_180, numpy.nanmin(longidudes180))
            self.max_x_180 = max(self.max_x_180, numpy.nanmax(longidudes180))

    def build(self) -> Geobox:
        """return the resulting geobox"""
        result = Geobox(self.max_y, self.min_y, self.min_x_0, self.max_x_0)
        result.spatial_reference = self.spatial_reference

        # Check if spanning 180th meridian
        if self.spatial_reference.IsGeographic() and abs(self.max_x_0 - self.min_x_0) > abs(
            self.max_x_180 - self.min_x_180
        ):
            result.right = self.max_x_180 + 360.0 if self.max_x_180 < -180.0 else self.max_x_180
            result.left = self.min_x_180 + 360.0 if self.min_x_180 < -180.0 else self.min_x_180

        return result


def parse_int(
    arg_name: str, arg_value: Union[str, int], default: int = 0, min_value: int = 0, max_value: int = sys.maxsize
) -> int:
    """Parse a string to int"""
    if arg_value is None:
        return default
    result = default
    try:
        result = int(arg_value)
    except ValueError as exc:
        raise ValueError(f"Invalid value '{arg_value}' for argument {arg_name}") from exc
    if result < min_value or result > max_value:
        raise ValueError(f"Value of {arg_name} argument must be in the range [{min_value}, {max_value}]")
    return result


def parse_float(arg_name: str, arg_value: Union[str, float], default: float = 0.0) -> float:
    """Parse a string to float"""
    if arg_value is None:
        return default
    try:
        return float(arg_value)
    except ValueError as exc:
        raise ValueError(f"Invalid value '{arg_value}' for argument {arg_name}") from exc


def parse_coord(arg_name: str, arg_value: dict) -> dict:
    """Parse a dict with keys north/south/west/east to a dict with keys lat/lon"""
    return {
        DtmConstants.DIM_LAT: [
            parse_float(arg_name + "[south]", arg_value["south"]),
            parse_float(arg_name + "[north]", arg_value["north"]),
        ],
        DtmConstants.DIM_LON: [
            parse_float(arg_name + "[west]", arg_value["west"]),
            parse_float(arg_name + "[east]", arg_value["east"]),
        ],
    }


def parse_geobox(arg_name: str, arg_value: dict) -> Geobox:
    """Parse a dict with keys north/south/west/east to a geobox"""
    return Geobox(
        upper=parse_float(arg_name + "[north]", arg_value["north"]),
        lower=parse_float(arg_name + "[south]", arg_value["south"]),
        left=parse_float(arg_name + "[west]", arg_value["west"]),
        right=parse_float(arg_name + "[east]", arg_value["east"]),
    )


def check_output_paths(i_paths: list, o_paths: list) -> None:
    """Check if the number of input path = number of output path.

    Arguments:
        i_paths {list} -- List of input paths.
        o_paths {list} -- List of output paths.

    Raises:
        AttributeError: The number of input/output paths must be equal.
    """
    if not o_paths is None and len(o_paths) != len(i_paths):
        raise AttributeError(f"{len(o_paths)} != {len(i_paths)}: The number of input/output paths must be equals.")


def create_output_path(
    i_path: str, suffix: str = "", extension: str = DtmConstants.EXTENSION_NC, o_path: str = None, overwrite=False
) -> str:
    """Generate (when o_path = None) or only check (when o_path != None) the name of output path with suffix and a extension.

    Arguments:
        i_path -- input path.
        suffix -- suffix of the generated output path.
        extension -- extension of the generated output path.
        o_path -- output path to check.

    Returns:
        [str] -- the generated or checked output path
    Raises:
        FileExistsError: File already exists.
    """
    if o_path is None:
        root, ext = os.path.splitext(i_path)
        o_path = root + suffix + extension

    # Check if there is a point in output file.
    if o_path.rfind(".") == -1:
        o_path += extension
    elif o_path[o_path.rfind(".") :] != extension:
        # Add or change the format
        o_path = o_path[: o_path.rfind(".")] + extension

    check_output_path(o_path, overwrite)

    return o_path


def check_output_path(o_path: str, overwrite: bool) -> None:
    """Raise a FileExistsError when file exists and overwrite is not allowed"""
    if os.path.exists(o_path) and not overwrite:
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), o_path)


def parse_list_of_files(arg_name: str, arg_value: Any, check_exist: bool = True) -> list:
    """Parse a list of files. each file must exists"""

    if isinstance(arg_value, str):  # only one path
        arg_value = [arg_value]

    if not check_exist:
        return arg_value

    result = []
    if arg_value:
        for file in arg_value:
            if not file == "[]":  # ignore special case
                if not os.path.exists(file):
                    raise ValueError(f"Invalid value for argument {arg_name} : file {file} does not exist")
                result.append(file)
    return result


def parse_datetime(arg_value: Any) -> Optional[dt.datetime]:
    """Parse arg_value in a datetime."""

    if arg_value is None:
        return None

    if isinstance(arg_value, dt.datetime):
        return arg_value

    return dateutil_parse(arg_value)


def parse_list_of_str(arg_value: Union[List[str], str, None]) -> List[str]:
    """Parse a list of str."""

    if arg_value is None:
        return []

    if isinstance(arg_value, list):  # already a list
        return arg_value

    return arg_value.split(",")


def parse_layers(arg_value) -> dict:
    if arg_value:
        result: Dict[str, bool] = dict.fromkeys(DtmConstants.LAYERS, False)
        for layer, activated in arg_value.items():
            if layer in DtmConstants.LAYERS:
                result[layer] = bool(activated)
            else:
                raise ValueError(f"Invalid layer name '{layer}'")
        return result
    else:
        return dict.fromkeys(DtmConstants.LAYERS, True)


def create_argv_parser(process_name: str, json_config_file_path: str) -> argparse.ArgumentParser:
    """
    Create a ArgumentParser to parse a command line and check the arguments according to the json configuration file

    Arguments:
        process_name -- name of the process.
        json_config_file_path -- path to the json configuration file.

    Returns:
        [argparse.ArgumentParser] -- the created parser

    """
    # Init
    parser = argparse.ArgumentParser(description=f"{process_name}.")

    # Read parameters from the configuration file
    logger = log.logging.getLogger(process_name)
    logger.debug("Accepted parameters : ")
    conf = None
    with open(json_config_file_path, "r", encoding="utf-8") as json_config_file:
        conf = json.load(json_config_file)

    # Add command "-e"
    parser.add_argument(
        "-e",
        "--example",
        nargs=0,
        help="Generate a json example file.",
        default="parameters.json",
        action=ExampleAction,
    )

    for param in conf["parameters"]:
        if not "choices" in param:
            param["choices"] = None

        if "type" in param:
            if param["type"] == "int":
                param["type"] = int
            elif param["type"] == "float":
                param["type"] = float
            elif param["type"] == "cdi_filter#filter":
                param["type"] = filters
            elif param["type"] == "geobox#coords":
                param["type"] = coord
            elif param["type"] == "layers":
                param["type"] = layers
            elif param["type"] == "cdi#modify":
                param["type"] = cdi
            else:
                param["type"] = str
        else:
            param["type"] = str

        if not "default" in param:
            param["default"] = None
        elif param["default"] == "[]":
            param["default"] = []

        if not "nargs" in param:
            param["nargs"] = None
        elif param["nargs"] not in ["?", "+", "*"]:
            # Handle composite nargs values like '+|1' by extracting the
            # argparse-compatible part (before '|')
            nargs_val = str(param["nargs"]).split("|")[0].strip()
            if nargs_val in ["?", "+", "*"]:
                param["nargs"] = nargs_val
            else:
                try:
                    param["nargs"] = int(nargs_val)
                except ValueError:
                    param["nargs"] = None

        if not "help" in param:
            param["help"] = None
        if not "name" in param:
            param["name"] = ""
        if not "action" in param:
            if "long_key" in param:
                parser.add_argument(
                    param["key"],
                    param["long_key"],
                    nargs=param["nargs"],
                    type=param["type"],
                    choices=param["choices"],
                    help=param["help"],
                    default=param["default"],
                )
            else:
                parser.add_argument(
                    param["key"],
                    nargs=param["nargs"],
                    type=param["type"],
                    choices=param["choices"],
                    help=param["help"],
                    default=param["default"],
                )
        else:
            if "long_key" in param:
                parser.add_argument(param["key"], param["long_key"], help=param["help"], action=param["action"])
            else:
                parser.add_argument(param["key"], help=param["help"], action=param["action"])
        logger.debug(param["key"] + " " + param["name"])

    return parser
