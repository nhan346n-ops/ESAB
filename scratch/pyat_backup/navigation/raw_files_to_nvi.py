#! /usr/bin/env python3
# coding: utf-8

import logging
import os
from typing import Dict, List, NamedTuple

import numpy as np
import pandas
import xarray as xr

from pynvi.version_2 import export_nvi as nvi_e
from pynvi.version_2.nvi_groups import NavigationGrp

__logger = logging.getLogger("Export NVI")


class RawFilesToNviArg(NamedTuple):
    """
    Class representing all arguments for configuring the process
    """

    # Source files
    i_paths: List[str]

    # Path of Nvi file to create
    o_paths: List[str]

    # Mandatory layers
    time: str
    latitude: str
    longitude: str
    heading: str
    height_above_reference_ellipsoid: str
    vertical_offset: str

    # Optional layers
    pitch: str | None = None
    roll: str | None = None
    heading_rate: str | None = None
    quality_flag: str | None = None
    speed_over_ground: str | None = None
    course_over_ground: str | None = None
    pitch_rate: str | None = None
    roll_rate: str | None = None
    speed_relative: str | None = None
    position_quality_indicator: str | None = None

    # Slicing
    o_value_count: List[int] = [-1]
    overwrite: bool = False

    # Time sampling
    time_interval: int = 0
    # Number sampling
    sounding_interval: int = 0


def exports(**kwargs) -> None:
    """
    Function accepting all arguments of the process as a dict. Possible arguments are listed in "RawFilesToNviArg" class
    """
    exports_with_ExportNviArg(RawFilesToNviArg(**kwargs))


def exports_with_ExportNviArg(args: RawFilesToNviArg) -> None:
    """
    Main function
    Use arguments to create a NVI
    """
    offset = 0
    for o_path, i_path, count in zip(args.o_paths, args.i_paths, args.o_value_count):
        if not os.path.exists(o_path) or args.overwrite:
            _process_export(args, o_path, i_path, offset, count)
        else:
            __logger.warning(f"{o_path} exists and cannot be overwritten")
        offset += count


def _process_export(args: RawFilesToNviArg, o_path: str, i_path: str, offset: int, count: int) -> None:
    # Build a XR dataset
    nav_dataset = _make_xr_dataset(args, offset, count)
    # Sampling
    nav_dataset = apply_sampling(args, count, nav_dataset)

    # Cast time to uint64 to fit NviArgs specification
    nav_dataset["time"] = nav_dataset["time"].astype("uint64")

    # Exporting
    nvi_e.exports(
        o_path=o_path,
        time=_get_variable(nav_dataset, "time"),
        latitude=_get_variable(nav_dataset, "latitude"),
        longitude=_get_variable(nav_dataset, "longitude"),
        heading=_get_variable(nav_dataset, "heading"),
        height_above_reference_ellipsoid=_get_variable(nav_dataset, "height_above_reference_ellipsoid"),
        vertical_offset=_get_variable(nav_dataset, "vertical_offset"),
        pitch=_get_variable(nav_dataset, "pitch"),
        roll=_get_variable(nav_dataset, "roll"),
        heading_rate=_get_variable(nav_dataset, "heading_rate"),
        quality_flag=_get_variable(nav_dataset, "quality_flag"),
        speed_over_ground=_get_variable(nav_dataset, "speed_over_ground"),
        course_over_ground=_get_variable(nav_dataset, "course_over_ground"),
        pitch_rate=_get_variable(nav_dataset, "pitch_rate"),
        roll_rate=_get_variable(nav_dataset, "roll_rate"),
        speed_relative=_get_variable(nav_dataset, "speed_relative"),
        position_quality_indicator=_get_variable(nav_dataset, "position_quality_indicator"),
        source_filenames=[i_path],
        overwrite=args.overwrite,
    )


def _make_xr_dataset(args: RawFilesToNviArg, offset: int, count: int) -> xr.Dataset:
    """
    Make numpy array with mapped files
    For each of them, prepare the definition of xarray Variable
    Group all xarray variables in a xarray dataset
    """
    var_defs: Dict = {}
    _add_data_var_definition(var_defs, "latitude", args.latitude, np.float64, offset, count)
    _add_data_var_definition(var_defs, "longitude", args.longitude, np.float64, offset, count)
    _add_data_var_definition(var_defs, "heading", args.heading, np.float32, offset, count)
    _add_data_var_definition(
        var_defs, "height_above_reference_ellipsoid", args.height_above_reference_ellipsoid, np.float32, offset, count
    )
    _add_data_var_definition(var_defs, "vertical_offset", args.vertical_offset, np.float32, offset, count)
    _add_data_var_definition(var_defs, "pitch", args.pitch, np.float32, offset, count)
    _add_data_var_definition(var_defs, "roll", args.roll, np.float32, offset, count)
    _add_data_var_definition(var_defs, "heading_rate", args.heading_rate, np.float32, offset, count)
    _add_data_var_definition(var_defs, "quality_flag", args.quality_flag, np.uint8, offset, count)
    _add_data_var_definition(var_defs, "speed_over_ground", args.speed_over_ground, np.float32, offset, count)
    _add_data_var_definition(var_defs, "course_over_ground", args.course_over_ground, np.float32, offset, count)
    _add_data_var_definition(var_defs, "pitch_rate", args.pitch_rate, np.float32, offset, count)
    _add_data_var_definition(var_defs, "roll_rate", args.roll_rate, np.float32, offset, count)
    _add_data_var_definition(var_defs, "speed_relative", args.speed_relative, np.float32, offset, count)
    _add_data_var_definition(
        var_defs, "position_quality_indicator", args.position_quality_indicator, np.uint8, offset, count
    )

    # Build a XR dataset
    return xr.Dataset(
        data_vars=var_defs,
        coords={
            "time": pandas.to_datetime(
                _make_np_array(args.time, np.uint64, offset, count),
            )
        },
    )


def apply_sampling(args: RawFilesToNviArg, count: int, nav_dataset: xr.Dataset) -> xr.Dataset:
    """
    Apply sampling of soundings :
     - Keep one navigation sounding per args.sounding_interval navigation soundings
     - Keep one navigation soundings every "args.time_interval" seconds
    """
    if args.sounding_interval > 0:
        nav_dataset = nav_dataset.isel(time=slice(0, count, args.sounding_interval))
    elif args.time_interval > 0:
        # Use 'unwrap' method to properly interpolate periodic variables (like heading...).
        _unwrap_periodic_variable(nav_dataset, NavigationGrp.LONGITUDE_VNAME, 180)
        _unwrap_periodic_variable(nav_dataset, NavigationGrp.HEADING_VNAME, 360)
        _unwrap_periodic_variable(nav_dataset, NavigationGrp.COURSE_OVER_GROUND_VNAME, 360)

        # Resample with interpolation, set "origin = 'start'" to ensure there is no shift at the beginning.
        nav_dataset = nav_dataset.resample(time=f"{args.time_interval}s", origin="start").interpolate()

        # apply modulo on heading/gndCourse (because heading has been unwraped before interpolation).
        _wrap_periodic_variable(nav_dataset, NavigationGrp.LONGITUDE_VNAME, -180, 180)
        _wrap_periodic_variable(nav_dataset, NavigationGrp.HEADING_VNAME, 0, 360)
        _wrap_periodic_variable(nav_dataset, NavigationGrp.COURSE_OVER_GROUND_VNAME, 0, 360)
    return nav_dataset


def _unwrap_periodic_variable(ds: xr.Dataset, variable_name: str, period: int) -> xr.Dataset:
    """
    Uses numpy 'unwrap' method to prepare periodic data for interpolation.
    """
    if variable_name in ds:
        unwrapped_var = ds[variable_name]
        unwrapped_var[~np.isnan(unwrapped_var)] = np.unwrap(unwrapped_var[~np.isnan(unwrapped_var)], period=period)
        ds[variable_name] = unwrapped_var
    return ds


def _wrap_periodic_variable(ds: xr.Dataset, variable_name: str, start_period: int, end_period: int) -> xr.Dataset:
    """
    Returns 'unwrapped' variable to periodic bound (from example, moves data back to [0:360] or [-180:180]).
    """
    if variable_name in ds:
        ds[variable_name] = (ds[variable_name] + start_period) % (end_period - start_period) + start_period
    return ds


def _get_variable(nav_dataset: xr.Dataset, variable_name: str) -> np.ndarray | None:
    """
    Convert and return a numpy array from a xarray variable when exists
    """
    return nav_dataset[variable_name].to_numpy() if variable_name in nav_dataset else None


def _add_data_var_definition(
    var_defs: Dict, variable_name: str, file: str | None, np_dtype, offset: int, count: int
) -> None:
    """
    Create a xarray variable definition and add it to the var_defs dictionnary
    """
    if file is not None:
        var_defs[variable_name] = (["time"], _make_np_array(file, np_dtype, offset, count))


def _make_np_array(data_file: str | None, data_type, offset: int, count: int) -> np.ndarray | None:
    """
    Create a numpy array, read data from the specified file
    """
    result = None
    if data_file is not None:
        __logger.debug(f"Reading {os.path.basename(data_file)}")
        if count > 0:
            result = np.fromfile(
                file=data_file,
                dtype=data_type,
                offset=offset * np.dtype(data_type).itemsize,
                count=count,
            )
        else:
            result = np.fromfile(file=data_file, dtype=data_type)
    return result
