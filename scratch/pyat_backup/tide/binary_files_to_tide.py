#! /usr/bin/env python3
# coding: utf-8

import json
import logging
import os
from typing import Dict, List, NamedTuple

import numpy as np
import pandas
import xarray as xr
from pytide.driver import tide_driver
from pytide.driver.export_tide_args import ExportTideArgs

__logger = logging.getLogger("Export Tide")


class BinaryFilesToTideArgs(NamedTuple):
    """
    Class representing all arguments for configuring the process
    """

    # Path of Tide file to create
    o_paths: List[str]

    # Mandatory layers
    time: str
    tide: str

    # Source files
    i_paths: List[str] | None = None

    # Optional layers
    latitude: str | None = None
    longitude: str | None = None

    # Slicing
    o_value_count: List[int] = [-1]
    overwrite: bool = False

    # Metadata (from JSON string)
    metadata: str | None = None

    # Tide metadata (from export wizard individual fields)
    tide_type: str | None = None
    tide_source: str | None = None
    vertical_reference: str | None = None
    surge_corrected: str | None = None
    vertical_datum: str | None = None
    prediction_model: str | None = None
    reference_height_above_ellipsoid: str | None = None

    tide_gauge_latitude: str | None = None
    tide_gauge_longitude: str | None = None
    tide_gauge_name: str | None = None
    tide_gauge_id: str | None = None

    comment: str | None = None


def exports(**kwargs) -> None:
    """
    Function accepting all arguments of the process as a dict. Possible arguments are listed in "BinaryFilesToTideArgs" class
    """
    exports_with_BinaryFilesToTideArgs(BinaryFilesToTideArgs(**kwargs))


def exports_with_BinaryFilesToTideArgs(args: BinaryFilesToTideArgs) -> None:
    """
    Main function
    Use arguments to create a Tide file
    """
    offset = 0
    for o_path, count in zip(args.o_paths, args.o_value_count):
        if not os.path.exists(o_path) or args.overwrite:
            _process_export(args, o_path, offset, count)
        else:
            __logger.warning(f"{o_path} exists and cannot be overwritten")
        offset += count


def _process_export(args: BinaryFilesToTideArgs, o_path: str, offset: int, count: int) -> None:
    # Build a XR dataset
    tide_dataset = _make_xr_dataset(args, offset, count)

    # Cast time to uint64 to fit TtbArgs specification
    tide_dataset["time"] = tide_dataset["time"].astype("uint64")

    # Convert metadata string to dictionary
    metadata_dict: Dict[str, str] = {}
    if args.metadata is not None:
        metadata_dict = json.loads(args.metadata)
    else:
        if args.tide_type is not None:
            metadata_dict["tide_type"] = args.tide_type
        if args.tide_source is not None:
            metadata_dict["tide_source"] = args.tide_source
        if args.vertical_reference is not None:
            metadata_dict["vertical_reference"] = args.vertical_reference
        if args.surge_corrected is not None:
            metadata_dict["surge_corrected"] = args.surge_corrected
        if args.vertical_datum is not None:
            metadata_dict["vertical_datum"] = args.vertical_datum
        if args.prediction_model is not None:
            metadata_dict["prediction_model"] = args.prediction_model
        if args.reference_height_above_ellipsoid is not None:
            metadata_dict["reference_height_above_ellipsoid"] = args.reference_height_above_ellipsoid
        if args.tide_gauge_latitude is not None:
            metadata_dict["tide_gauge_latitude"] = args.tide_gauge_latitude
        if args.tide_gauge_longitude is not None:
            metadata_dict["tide_gauge_longitude"] = args.tide_gauge_longitude
        if args.tide_gauge_name is not None:
            metadata_dict["tide_gauge_name"] = args.tide_gauge_name
        if args.tide_gauge_id is not None:
            metadata_dict["tide_gauge_id"] = args.tide_gauge_id
        if args.comment is not None:
            metadata_dict["comment"] = args.comment

    # Exporting
    export_args = ExportTideArgs(
        time=_get_variable(tide_dataset, "time"),
        latitude=_get_variable(tide_dataset, "latitude"),
        longitude=_get_variable(tide_dataset, "longitude"),
        tide=_get_variable(tide_dataset, "tide"),
        o_path=o_path,
        overwrite=args.overwrite,
        metadata=metadata_dict,
    )
    tide_driver.exports_with_ExportTideArgs(export_args)


def _make_xr_dataset(args: BinaryFilesToTideArgs, offset: int, count: int) -> xr.Dataset:
    """
    Make numpy array with mapped files
    For each of them, prepare the definition of xarray Variable
    Group all xarray variables in a xarray dataset
    """
    var_defs: Dict = {}
    _add_data_var_definition(var_defs, "latitude", args.latitude, np.float64, offset, count)
    _add_data_var_definition(var_defs, "longitude", args.longitude, np.float64, offset, count)
    _add_data_var_definition(var_defs, "tide", args.tide, np.float32, offset, count)

    # Build a XR dataset
    return xr.Dataset(
        data_vars=var_defs,
        coords={
            "time": pandas.to_datetime(
                _make_np_array(args.time, np.uint64, offset, count),
            )
        },
    )


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
