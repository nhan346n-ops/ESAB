#! /usr/bin/env python3
# coding: utf-8


import tempfile as tmp

import numpy as np

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.generator.dtm_generator as dtm_generator
from pyat.dtm.merge.merge_fill import MergeFillProcess
from pyat.dtm.merge.merge_simple import MergeSimpleProcess


def make_dtm_1(temp_dir: str) -> str:
    """
    Generate a DTM with no elevation at 2 corners (SW and NE)
    Value count are all 5
    Filtered count are all 7

           -9.995°              -9.0°
              |                   |
     49°      +---------------+
              |               |
              |               +---+
              |                   |
              | elev = -90        |
              +---+               |
                  |               |
     48.005°      +---------------+

    """
    nb_cell = 200
    elevations = np.full((nb_cell, nb_cell), -90.0)
    row, col = np.indices((40, 40))
    elevations[row, col] = dtm_driver.get_missing_value(DtmConstants.ELEVATION_NAME)
    elevations[row - 40, col - 40] = dtm_driver.get_missing_value(DtmConstants.ELEVATION_NAME)
    value_count = np.full_like(elevations, 5, dtype=int)
    value_count[np.isnan(elevations)] = dtm_driver.get_missing_value(DtmConstants.VALUE_COUNT)
    filtered_count = np.full_like(elevations, 7, dtype=int)
    filtered_count[np.isnan(elevations)] = dtm_driver.get_missing_value(DtmConstants.FILTERED_COUNT)
    return dtm_generator.make_dtm_from_NW(
        (-9.0, 49.0),
        0.005,
        {
            DtmConstants.ELEVATION_NAME: elevations,
            DtmConstants.VALUE_COUNT: value_count,
            DtmConstants.FILTERED_COUNT: filtered_count,
        },
        temp_dir,
    )


def make_dtm_2(temp_dir) -> str:
    """
    Generate a DTM with no elevation at 2 corners (SW and NE)
    Value count are all 3
    Filtered count are all 10

         -9.195°        -9.0°
            |             |
      49.0° +-------------+
            |             |
            |             |
            |  elev = -20 |
            |             |
    48.805° +-------------+

    """
    nb_cell = 50
    # Generations from -90 to -100
    elevations = np.full((nb_cell, nb_cell), -20.0)
    return dtm_generator.make_dtm_from_NW(
        (-9.0, 49.0),
        0.005,
        {
            DtmConstants.ELEVATION_NAME: elevations,
            DtmConstants.VALUE_COUNT: np.full_like(elevations, 3, int),
            DtmConstants.FILTERED_COUNT: np.full_like(elevations, 10, int),
        },
        temp_dir,
    )


def test_merge_simple_without_smoothing():
    """
    Merge this 2 DTMs :
                          +-------------+
                          |  elev = -20 |
              +-----------|---+         |                +-----------+---+---+
              |           |   |         |                |           |   |-20|
              | dtm_base  |   +---+     |                |           |-55+---+
              |           +-------|-----+    merge =>    |           +-------+
              | elev = -90        |                      | elev = -90        |
              +---+               |                      +---+               |
                  |               |                          |               |
                  +---------------+                          +---------------+

    """
    with tmp.TemporaryDirectory() as temp_dir:
        dtm_base_path = make_dtm_1(temp_dir)
        dtm_path2 = make_dtm_2(temp_dir)
        path_o_dtm = tmp.mktemp(suffix=".dtm.nc", dir=temp_dir)

        geobox = {"north": 49.0025, "south": 48.002500000000005, "west": -9.997499999999999, "east": -8.9975}
        merge = MergeSimpleProcess(i_paths=[dtm_base_path, dtm_path2], coord=geobox, o_path=path_o_dtm)
        merge()

        with dtm_driver.open_dtm(path_o_dtm) as o_driver:
            # Check elevations
            elevs = o_driver[DtmConstants.ELEVATION_NAME][:].data
            value_count = o_driver[DtmConstants.VALUE_COUNT][:].data
            filtered_count = o_driver[DtmConstants.FILTERED_COUNT][:].data
            # All elevations at NE (-9°/49°) must come from dtm_path2 (-20.0)
            # VALUE_COUNT and FILTERED_COUNT come from dtm_path2 too (3 and 10)
            for row in range(-40, 0):
                for col in range(-40, 0):
                    assert elevs[row, col] == -20.0
                    assert value_count[row, col] == 3
                    assert filtered_count[row, col] == 10
            # Check mean elevations on the covering surface
            # VALUE_COUNT and FILTERED_COUNT are the sum from both DTM (8 and 17)
            for row in range(-50, 0):
                for col in range(-50, -40):
                    assert elevs[row, col] == -55.0
                    assert value_count[row, col] == 8
                    assert filtered_count[row, col] == 17
            for row in range(-50, -40):
                for col in range(-50, 0):
                    assert elevs[row, col] == -55.0
                    assert value_count[row, col] == 8
                    assert filtered_count[row, col] == 17


def test_merge_fill_without_smoothing():
    """
    Merge this 2 DTMs :
                          +-------------+
                          |  elev = -20 |
              +-----------|---+         |                +-----------+---+---+
              |           |   |         |                |               |-20|
              | dtm_base  |   +---+     |                |               +---+
              |           +-------|-----+    merge =>    |                   |
              | elev = -90        |                      | elev = -90        |
              +---+               |                      +---+               |
                  |               |                          |               |
                  +---------------+                          +---------------+

    """
    with tmp.TemporaryDirectory() as temp_dir:
        dtm_base_path = make_dtm_1(temp_dir)
        dtm_path2 = make_dtm_2(temp_dir)
        path_o_dtm = tmp.mktemp(suffix=".dtm.nc", dir=temp_dir)

        geobox = {"north": 49.0025, "south": 48.002500000000005, "west": -9.997499999999999, "east": -8.9975}
        merge = MergeFillProcess(i_paths=[dtm_base_path, dtm_path2], coord=geobox, o_path=path_o_dtm)
        merge()

        with dtm_driver.open_dtm(path_o_dtm) as o_driver:
            # Check elevations
            elevs = o_driver[DtmConstants.ELEVATION_NAME][:]
            value_count = o_driver[DtmConstants.VALUE_COUNT][:]
            filtered_count = o_driver[DtmConstants.FILTERED_COUNT][:]
            # All elevations at NE (-9°/49°) must come from dtm_path2 (-20.0)
            # VALUE_COUNT and FILTERED_COUNT come from dtm_path2 too (3 and 10)
            for row in range(-40, 0):
                for col in range(-40, 0):
                    assert elevs[row, col] == -20.0
                    assert value_count[row, col] == 3
                    assert filtered_count[row, col] == 10
            # Check elevations on covering surface : must come from dtm_path1 (-90.0)
            # VALUE_COUNT and FILTERED_COUNT come from dtm_path1 too (5 and 7)
            for row in range(-50, 0):
                for col in range(-50, -40):
                    assert elevs[row, col] == -90.0
                    assert value_count[row, col] == 5
                    assert filtered_count[row, col] == 7
            for row in range(-50, -40):
                for col in range(-50, 0):
                    assert elevs[row, col] == -90.0
                    assert value_count[row, col] == 5
                    assert filtered_count[row, col] == 7


def test_merge_fill_with_smoothing():
    """
    Merge this 2 DTMs :                          smooth elevations -----+
                          +-------------+                               |
                          |  elev = -20 |                               |
              +-----------|---+         |                +-------------+-+---+
              |           |   |         |                |             | |-20|
              |           |   +---+     |                |             | +---+
              |           +-------|-----+    merge =>    |             +-----+
              | elev = -90        |                      | elev=-90          |
              +---+               |                      +---+               |
                  |               |                          |               |
                  +---------------+                          +---------------+

    """
    with tmp.TemporaryDirectory() as temp_dir:
        dtm_base_path = make_dtm_1(temp_dir)
        dtm_path2 = make_dtm_2(temp_dir)
        path_o_dtm = tmp.mktemp(suffix=".dtm.nc", dir=temp_dir)

        geobox = {"north": 49.0025, "south": 48.002500000000005, "west": -9.997499999999999, "east": -8.9975}
        merge = MergeFillProcess(
            i_paths=[dtm_base_path, dtm_path2], coord=geobox, o_path=path_o_dtm, smoothing_border=3
        )
        merge()

        with dtm_driver.open_dtm(path_o_dtm) as o_driver:
            # Check elevations
            elevs = o_driver[DtmConstants.ELEVATION_NAME][:]
            value_count = o_driver[DtmConstants.VALUE_COUNT][:]
            filtered_count = o_driver[DtmConstants.FILTERED_COUNT][:]
            # All elevations at NE (-9°/49°) values must come from dtm_path2
            for row in range(-37, 0):
                for col in range(-37, 0):
                    assert elevs[row, col] == -20.0
                    assert value_count[row, col] == 3
                    assert filtered_count[row, col] == 10

            # Smoothing from -90 to -20
            # VALUE_COUNT and FILTERED_COUNT come from dtm_path2 (3 and 10)
            nb_count_eq_3 = 0
            nb_filtered_eq_10 = 0
            for row in range(-40, 1):
                for col in range(-40, -37):
                    # Gap filling let some cells in border with no value
                    assert -90.0 <= elevs[row, col] <= -20.0 or elevs[row, col] is np.ma.masked
                    if value_count[row, col] == 3:
                        nb_count_eq_3 += 1
                    if filtered_count[row, col] == 10:
                        nb_filtered_eq_10 += 1
            assert nb_count_eq_3 == 120
            assert nb_filtered_eq_10 == 120

            nb_count_eq_3 = 0
            nb_filtered_eq_10 = 0
            for row in range(-40, -37):
                for col in range(-40, 0):
                    assert -90.0 <= elevs[row, col] <= -20.0 or elevs[row, col] is np.ma.masked
                    if value_count[row, col] == 3:
                        nb_count_eq_3 += 1
                    if filtered_count[row, col] == 10:
                        nb_filtered_eq_10 += 1
            assert nb_count_eq_3 == 120
            assert nb_filtered_eq_10 == 120

            # Check elevations on covering surface : values must come from dtm_path1
            for row in range(-50, 0):
                for col in range(-50, -40):
                    assert elevs[row, col] == -90.0
                    assert value_count[row, col] == 5
                    assert filtered_count[row, col] == 7
            for row in range(-50, -40):
                for col in range(-50, 0):
                    assert elevs[row, col] == -90.0
                    assert value_count[row, col] == 5
                    assert filtered_count[row, col] == 7
