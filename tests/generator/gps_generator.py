#! /usr/bin/env python3
# coding: utf-8
import datetime

from typing import Optional, Tuple

import netCDF4 as nc
import numpy as np
import tempfile as tmp


def make_gps_netcdf_with_data(
    start_time: float,
    cycle_count: int,
    temp_dir: Optional[str] = None,
):
    """
    Produce multiple GPS netcdf files and return its path.
    cycle_count : number of navigation positions
    start_time : starting days (days)
    """
    path_gps = tmp.mktemp(suffix=".gps", dir=temp_dir)
    with nc.Dataset(path_gps, "w", format="NETCDF3_CLASSIC") as dataset:
        # NetCdf file
        dataset.source = "Acquisition of test sensor"
        dataset.conventions = "CF-1.0."
        dataset.creationtime = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        dataset.frame_period = 1.0
        # Dimensions
        # dataset.createDimension("measureTS", 1)
        # dataset.createDimension("gndspeed", 1)
        # dataset.createDimension("gndcourse", 1)
        dataset.createDimension("alt", cycle_count)
        dataset.createDimension("long", cycle_count)
        # dataset.createDimension("prec", 1)
        dataset.createDimension("lat", cycle_count)
        dataset.createDimension("mode", cycle_count)
        dataset.createDimension("time", None)
        # Variables
        # dataset.createVariable("measureTS", "f4", ("measureTS",))
        # dataset.createVariable("gndspeed", "f4", ("gndspeed",))
        alts = dataset.createVariable("alt", "f4", ("alt",))
        alts.units = "m"
        longs = dataset.createVariable("long", "f8", ("long",))
        longs.units = "degree_east"
        lats = dataset.createVariable("lat", "f8", ("lat",))
        lats.units = "degree_north"
        modes = dataset.createVariable("mode", "i4", ("mode",))
        modes.units = "dimensionless"
        times = dataset.createVariable("time", "f8", ("time",))
        times.units = "days since 1899-12-30 00:00:00 UTC"
        times.calendar = "gregorian"
        alts[:] = np.linspace(14.000, 16.000, num=cycle_count)
        longs[:] = np.linspace(-5.0000000, -6.0000000, num=cycle_count)
        lats[:] = np.linspace(48.0000000, 49.0000000, num=cycle_count)
        modes[:] = np.linspace(5, 5, num=cycle_count)
        modes[5] = 1
        times[:] = np.linspace(start_time, start_time + cycle_count / (24 * 3600), num=cycle_count)
    return path_gps
