#! /usr/bin/env python3
# coding: utf-8
import os

import pandas as pd
from pygws.service.progress_monitor import DefaultMonitor

from pyat.tide.sensor_tide_app import NavTide
from tests.generator.gps_generator import make_gps_netcdf_with_data


def generate_gps_netcdf_file() -> str:
    """
    Create a netcdf3 gps file
    Returns:

    """
    _gps_file = make_gps_netcdf_with_data(start_time=43899.00012627, cycle_count=3600)
    return _gps_file


def open_file(input_file) -> pd.DataFrame:
    """
    open the file and parse it to panda Dataframe
    Args:
        input_file: csv
    Returns:
        panda Dataframe
    """
    return pd.read_csv(input_file)


def test_gps_netcdf_tide():
    _gps_file = make_gps_netcdf_with_data(start_time=43899.00012627, cycle_count=3600)
    _dir_path = os.path.dirname(os.path.realpath(_gps_file))
    _output_file = _dir_path + "/test_gps_export_tide.csv"
    _gps_netcdf_tide = NavTide(
        input_files=[_gps_file],
        output_file=_output_file,
        positioning_type_filter=False,
        reference_surface="",
        interval_minutes=10,
        monitor=DefaultMonitor,
    )
    _gps_netcdf_tide.__call__()
    _result = open_file(_output_file)
    assert _result.size == 9  # one value every 10 minutes
    assert os.path.isfile(_output_file) is True


def test_gps_netcdf_tide_with_shipping_type_filter():
    _gps_file = make_gps_netcdf_with_data(start_time=43899.00012627, cycle_count=3600)
    _dir_path = os.path.dirname(os.path.realpath(_gps_file))
    _output_file = _dir_path + "/test_gps_export_tide.csv"
    _gps_netcdf_tide = NavTide(
        input_files=[_gps_file],
        output_file=_output_file,
        positioning_type_filter=True,
        reference_surface="WGS84",
        interval_minutes=10,
        monitor=DefaultMonitor,
    )
    _gps_netcdf_tide.__call__()
    _result = open_file(_output_file)
    assert _result.size == 9  # one value every 10 minutes
    assert os.path.isfile(_output_file) is True


if __name__ == "__main__":
    gps_file1 = make_gps_netcdf_with_data(start_time=43899.00012627, cycle_count=3600)
    dir_path = os.path.dirname(os.path.realpath(gps_file1))
    output_file = dir_path + "test_gps_export_tide.csv"

    gps_netcdf_tide = NavTide(
        input_files=[gps_file1],
        output_file=output_file,
        positioning_type_filter=False,
        reference_surface="",
        monitor=DefaultMonitor,
    )
    gps_netcdf_tide.__call__()
    result = open_file(output_file)
