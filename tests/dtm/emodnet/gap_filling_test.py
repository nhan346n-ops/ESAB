#! /usr/bin/env python3
# coding: utf-8

import tempfile as tmp

import netCDF4 as nc
import numpy as np

import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.generator.dtm_generator as dtm_generator
from pyat.dtm import dtm_driver
from pyat.dtm.transform.interpolation.gap_filling import GapFillingProcess
from pyat.dtm.numba.gap_filling_functions import find_coord, find_distance
from tests.generator.dtm_generator import DtmGenerator


def make_dtm(temp_dir: str) -> str:
    """
    Generate a DTM with no elevation at the centrer (16 cells)
            -10°                 -9°
              |                   |
         49°  +-------------------+
              |Elev=[-900, -1000] |
              |    +---------+    |
              |    |   NaN   |    |
              |    +---------+    |
              |                   |
         48°  +-------------------+

    """
    elevations = 100 * np.random.default_rng().random((10, 10)) - 1000
    row, col = np.indices((4, 4))
    elevations[row + 3, col + 3] = dtm_driver.get_missing_value(DtmConstants.ELEVATION_NAME)

    value_count = np.full_like(elevations, 2, dtype=dtm_driver.get_type(DtmConstants.VALUE_COUNT))
    value_count[row + 3, col + 3] = dtm_driver.get_missing_value(DtmConstants.VALUE_COUNT)
    # Nb of missing values in value_count :
    unique, counts = np.unique(value_count, return_counts=True)
    frequencies = dict(zip(unique, counts))
    assert frequencies[dtm_driver.get_missing_value(DtmConstants.VALUE_COUNT)] == 16
    assert frequencies[2] == 84

    # Set a cdi to 2 where cells are empty
    cdi_index = np.ones_like(elevations, dtype=dtm_driver.get_type(DtmConstants.CDI_INDEX))
    cdi_index[row + 3, col + 3] = 2

    return dtm_generator.make_dtm_with_data(
        (-9.0, 49.0),
        (-10, 48.0),
        {
            DtmConstants.ELEVATION_NAME: elevations,
            DtmConstants.VALUE_COUNT: value_count,
            DtmConstants.CDI_INDEX: cdi_index,
        },
        temp_dir=temp_dir,
    )


def do_test_gap_filling_value_count_to_0_not_nan(path_i_dtm: str, path_o_dtm: str):
    """
    Check that interpolated cell has 0 as value count rather than Missing Value
    """
    process = GapFillingProcess(i_paths=[path_i_dtm], o_paths=[path_o_dtm], mask_size=3, overwrite=True)
    process()

    with dtm_driver.open_dtm(path_o_dtm) as o_driver:
        assert DtmConstants.VALUE_COUNT in o_driver
        value_count = o_driver[DtmConstants.VALUE_COUNT][:].data

        # In that test case, gap filling computes elevation for 8 cells and let the 8 others to NaN. Expected value_count is 8
        unique, counts = np.unique(value_count, return_counts=True)
        frequencies = dict(zip(unique, counts))
        assert frequencies[-1] == 8  # Not interpolated cells => value_count must be missing_value
        assert frequencies[0] == 8  # Interpolated cells => value_count must be 0
        assert frequencies[2] == 84  # Unchanged cells

        # Gap filling must reset the CDI for celles where interpolation is not possible
        assert DtmConstants.CDI in o_driver
        assert DtmConstants.CDI_INDEX in o_driver
        cdi_index = o_driver[DtmConstants.CDI_INDEX][:].data

        unique, counts = np.unique(cdi_index, return_counts=True)
        frequencies = dict(zip(unique, counts))
        assert frequencies[-1] == 8  # Not interpolated cells
        assert frequencies[1] == 92  # 84 unchanged cells + 8 interpolated cells


def test_gap_filling_value_count_to_0_not_nan():
    """
    Check that interpolated cell has 0 as value count rather than Missing Value
    """
    with tmp.TemporaryDirectory() as temp_dir:
        path_i_dtm = make_dtm(temp_dir)
        path_o_dtm = tmp.mktemp(suffix=".dtm.nc", dir=temp_dir)
        do_test_gap_filling_value_count_to_0_not_nan(path_i_dtm, path_o_dtm)


def test_gap_filling_outfile_is_infile():
    """
    Same test than previously but o_dtm == i_dtm
    """
    with tmp.TemporaryDirectory() as temp_dir:
        path_i_dtm = make_dtm(temp_dir)
        do_test_gap_filling_value_count_to_0_not_nan(path_i_dtm, path_i_dtm)


def test_find_distance():
    # Parameters
    size = 3

    # Process
    m_test = find_distance(size)

    # Verify
    m = np.array(
        [
            [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0], [7.0, 8.0, 9.0]],
            [[5.0, 6.0, 9.0], [2.0, 4.0, 8.0], [1.0, 3.0, 7.0]],
            [[9.0, 8.0, 7.0], [6.0, 4.0, 3.0], [5.0, 2.0, 1.0]],
            [[7.0, 3.0, 1.0], [8.0, 4.0, 2.0], [9.0, 6.0, 5.0]],
        ]
    )
    for q in range(m_test.shape[0]):
        for row in range(m_test.shape[1]):
            for col in range(m_test.shape[2]):
                # Vérification du premier quadran
                assert m_test[q][row, col] == m[q][row, col]


def test_find_coord():
    # Parameters
    mask_size = 3

    # Process
    index = find_distance(mask_size)
    coord = find_coord(index)

    # Verify the first quadran
    ind = 0
    for cell in coord[0]:
        r = cell[0]
        c = cell[1]
        ind += 1
        if ind == 1:
            assert int(r) == 0
            assert int(c) == 0
        if ind == 2:
            assert int(r) == 0
            assert int(c) == 1
        if ind == 3:
            assert int(r) == 1
            assert int(c) == 0
        if ind == 4:
            assert int(r) == 1
            assert int(c) == 1
        if ind == 5:
            assert int(r) == 0
            assert int(c) == 2
        if ind == 6:
            assert int(r) == 1
            assert int(c) == 2
        if ind == 7:
            assert int(r) == 2
            assert int(c) == 0
        if ind == 8:
            assert int(r) == 2
            assert int(c) == 1
        if ind == 9:
            assert int(r) == 2
            assert int(c) == 2


def test_gap_filling():
    with tmp.TemporaryDirectory() as temp_dir:

        # Parameters
        generator = DtmGenerator(temp_dir)
        i_path = generator.create_pattern_interpolation(value=10, value_2=20)
        i_paths = [i_path]
        mask_size = 3
        o_path = i_path[:-3] + "-gap_fill" + DtmConstants.EXTENSION
        params = {"i_paths": i_paths, "mask_size": mask_size, "o_paths": [o_path], "overwrite": True}

        # Process
        gapFilling = GapFillingProcess(**params)
        gapFilling()

        # Verify
        with nc.Dataset(o_path) as o_data, nc.Dataset(i_path) as i_data:
            o_elev = o_data[DtmConstants.ELEVATION_NAME][:]
            i_elev = i_data[DtmConstants.ELEVATION_NAME][:]

            rowSize = o_elev.shape[0]
            colSize = o_elev.shape[1]

            for row in range(rowSize):
                for col in range(colSize):
                    if mask_size <= row < rowSize - mask_size and mask_size <= col < colSize - mask_size:
                        if np.ma.is_masked(i_elev[row, col]):
                            assert np.ma.is_masked(o_elev[row, col]) is False

            # Check CDI
            o_cdi_ref = o_data[DtmConstants.CDI]
            assert 1 == o_cdi_ref.size
            assert "10" == o_cdi_ref[0]
