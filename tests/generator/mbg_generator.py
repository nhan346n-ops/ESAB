#! /usr/bin/env python3
# coding: utf-8

import tempfile as tmp
from typing import Optional, Tuple

import netCDF4 as nc
import numpy as np


def make_mbg_with_data(
    cycle_count: int,
    julian_date: int,
    min_max_hours: Tuple[int, int],
    min_max_longitudes: Tuple[float, float],
    min_max_latitudes: Tuple[float, float],
    temp_dir: Optional[str] = None,
):
    """
    Produce a the MBG and return its path.
    cycle_count : number of navigation positions
    julian_date : starting date (22/11/2015 = 2457349)
    hour : starting hour (ms)
    longitude/latitude : first navigation position
    """
    path_mbg = tmp.mktemp(suffix=".mbg", dir=temp_dir)
    with nc.Dataset(path_mbg, "w", format="NETCDF3_CLASSIC") as dataset:
        dataset.mbClasse = "MB_SWATH"
        dataset.mbVersion = 210
        dataset.mbLevel = 0
        dataset.mbMinDepth = 198.0
        dataset.mbMaxDepth = 512.0
        dataset.createDimension("mbCycleNbr", cycle_count)
        dataset.createDimension("mbBeamNbr", 1)
        dataset.createDimension("mbAntennaNbr", 1)
        dataset.createDimension("mbVelocityProfilNbr", 1)
        dataset.mbNorthLatitude = min_max_longitudes[1]
        dataset.mbSouthLatitude = min_max_longitudes[0]
        dataset.mbEastLongitude = min_max_latitudes[1]
        dataset.mbWestLongitude = min_max_latitudes[0]
        dates = dataset.createVariable("mbDate", "i4", ("mbCycleNbr",))
        dates.add_offset = 2440588
        dataset.mbStartDate = julian_date
        dataset.mbEndDate = julian_date
        dates[:] = np.full((cycle_count), dataset.mbStartDate)

        dataset.mbStartTime = min_max_hours[0]
        dataset.mbEndTime = min_max_hours[1]
        hours = dataset.createVariable("mbTime", "i4", ("mbCycleNbr",))
        hours[:] = np.linspace(dataset.mbStartTime, dataset.mbEndTime + (cycle_count * 100), cycle_count)

        longitudes = dataset.createVariable("mbAbscissa", "i4", ("mbCycleNbr",))
        longitudes.scale_factor = 1e-7
        longitudes[:] = np.linspace(dataset.mbWestLongitude, dataset.mbEastLongitude, cycle_count)

        latitudes = dataset.createVariable("mbOrdinate", "i4", ("mbCycleNbr",))
        latitudes.scale_factor = 5e-8
        latitudes[:] = np.linspace(dataset.mbSouthLatitude, dataset.mbNorthLatitude, cycle_count)

    return path_mbg


if __name__ == "__main__":
    print(
        make_mbg_with_data(
            1296,
            2457349,
            (67803129, 69602009),
            (38.28327886547648, 38.201800778531485),
            (-17.692696033303296, -17.64389017109131),
        )
    )
