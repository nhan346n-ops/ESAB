#! /usr/bin/env python3
# coding: utf-8

import tempfile

import numpy as np
import pytest

import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.generator.dtm_generator as dtm_generator
from pyat.dtm import dtm_driver
from pyat.dtm.transform.geobox_shrink import ShrinkProcess
from pyat.utils.coords import DEG_MIN_SEC_STRING_from_DEGREES, DEGREES_from_DEG_MIN_SEC

west = DEGREES_from_DEG_MIN_SEC("030°48'48'' E")
south = DEGREES_from_DEG_MIN_SEC("43°54'55'' N")
row_count = 1420
col_count = 2560
spatial_res = 1.0 / 3600.0 * 10.0  # 10''


def make_dtm(empty_cell_count: int, temp_dir: str) -> str:
    """
    Generate a DTM with no elevation at the border
    """
    elevations = np.full((row_count, col_count), dtm_driver.get_missing_value(DtmConstants.ELEVATION_NAME))
    row, col = np.indices((row_count - 2 * empty_cell_count, col_count - 2 * empty_cell_count))
    elevations[row + empty_cell_count, col + empty_cell_count] = (
        100 * np.random.default_rng().random((row_count - 2 * empty_cell_count, col_count - 2 * empty_cell_count))
        - 1000
    )

    value_count = np.full_like(elevations, dtm_driver.get_missing_value(DtmConstants.VALUE_COUNT))
    value_count[row + empty_cell_count, col + empty_cell_count] = 1

    return dtm_generator.make_dtm_with_data(
        (west, south),
        (west + spatial_res * (col_count - 1), south + spatial_res * (row_count - 1)),
        {
            DtmConstants.ELEVATION_NAME: elevations,
            DtmConstants.VALUE_COUNT: value_count,
        },
        temp_dir=temp_dir,
    )


def test_no_shrink_dtm():
    """
    Define a DTM without any empty border
    Process the shrink process
    Check that the resulting DTM is the same than the created one
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        empty_cell_count = 0
        path_i_dtm = make_dtm(empty_cell_count, temp_dir)
        path_o_dtm = tempfile.mktemp(suffix=".dtm.nc", dir=temp_dir)

        # Launch the process
        shinker = ShrinkProcess(i_paths=[path_i_dtm], o_paths=[path_o_dtm])
        shinker()

        with dtm_driver.open_dtm(path_o_dtm) as o_driver:
            o_dtm_file = o_driver.dtm_file
            assert o_dtm_file.col_count == col_count
            assert o_dtm_file.row_count == row_count
            assert o_dtm_file.spatial_resolution_x == pytest.approx(spatial_res)
            assert o_dtm_file.spatial_resolution_y == pytest.approx(spatial_res)
            lons = o_dtm_file.compute_x_axis()
            assert DEG_MIN_SEC_STRING_from_DEGREES(lons[0]) == DEG_MIN_SEC_STRING_from_DEGREES(west)
            lats = o_dtm_file.compute_y_axis()
            assert DEG_MIN_SEC_STRING_from_DEGREES(lats[0]) == DEG_MIN_SEC_STRING_from_DEGREES(south)


def test_plain_shrink_dtm():
    """
    Define a DTM with an empty border of 5 cells
    Process the shrink process
    Check that the empty border has been remove in the resulting DTM
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # 5 consecutive empty cells around the dtm
        empty_cell_count = 5
        path_i_dtm = make_dtm(empty_cell_count, temp_dir)
        path_o_dtm = tempfile.mktemp(suffix=".dtm.nc", dir=temp_dir)

        # Launch the process
        shinker = ShrinkProcess(i_paths=[path_i_dtm], o_paths=[path_o_dtm])
        shinker()

        with dtm_driver.open_dtm(path_i_dtm) as i_driver, dtm_driver.open_dtm(path_o_dtm) as o_driver:
            o_dtm_file = o_driver.dtm_file
            assert o_dtm_file.col_count == col_count - 2 * empty_cell_count
            assert o_dtm_file.row_count == row_count - 2 * empty_cell_count
            assert o_dtm_file.spatial_resolution_x == pytest.approx(spatial_res)
            assert o_dtm_file.spatial_resolution_y == pytest.approx(spatial_res)

            o_lons = o_driver[DtmConstants.LON_NAME]
            o_lats = o_driver[DtmConstants.LAT_NAME]
            i_lons = i_driver[DtmConstants.LON_NAME]
            i_lats = i_driver[DtmConstants.LAT_NAME]

            # First cell's lon/lat is the same than the first cell with elevation of the input file (col = 5 and row = 5)
            assert o_lons[0] == pytest.approx(i_lons[5], abs=1e-14)
            assert o_lats[0] == pytest.approx(i_lats[5], abs=1e-14)
            # Last cell's lon/lat is the same than the Last cell with elevation of the input file (col = -5 and row = -5)
            assert o_lons[-1] == pytest.approx(i_lons[-6], abs=1e-14)
            assert o_lats[-1] == pytest.approx(i_lats[-6], abs=1e-14)
