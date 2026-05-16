#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile as tmp

import numpy as np

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.generator.dtm_generator as dtm_generator
import tests.generator.kml_generator as kml_generator
from pyat.dtm.cdi.set_cdi_process import SetCdiProcess


def make_dtm_with_no_cdi() -> str:
    """
    Generate a DTM with a hole at [0,0] holes and a STDEV layer
    Cell's longitude = [-10°, -9.9°, -9.8°, ..., -9°],
    Cell's latitude = [39°, 39.1°, 39.2°, ..., 40°]
    Resolution = 6' (0,1°)
    Elevation[-10°, 39°] = NaN
    """
    # Generations from -90 to -100
    elevations = 10 * np.random.default_rng().random((10, 10)) - 100
    elevations[0, 0] = np.nan
    stdev = np.random.default_rng().random((10, 10))
    stdev[0, 0] = np.nan

    return dtm_generator.make_dtm_with_data(
        (-10.0, 39.0), (-9.0, 40.0), {DtmConstants.ELEVATION_NAME: elevations, DtmConstants.STDEV: stdev}
    )


def make_dtm_with_1_cdi(cdi: str) -> str:
    """
    Generate a DTM with a hole at [0,0] holes and a STDEV layer
    Cell's longitude = [-10°, -9.9°, -9.8°, ..., -9°],
    Cell's latitude = [39°, 39.1°, 39.2°, ..., 40°]
    Resolution = 6' (0,1°)
    Elevation[-10°, 39°] = NaN
    """
    result = make_dtm_with_no_cdi()
    with dtm_driver.open_dtm(result, "r+") as o_driver:
        o_driver.create_cdi_reference_variable([cdi])
        o_driver.add_layer(DtmConstants.CDI_INDEX)
        o_driver[DtmConstants.CDI_INDEX][:] = 0
        # elevation is NaN at [0,0]
        o_driver[DtmConstants.CDI_INDEX][0, 0] = -1
    return result


def check_only_1_cdi(driver: dtm_driver.DtmDriver, expected_cdi: str, expected_cell_count: int = 99):
    """
    Check that all elevations have the same CDI
    """
    # Check CDI is has been created
    o_cdi_ref = driver[DtmConstants.CDI]
    assert len(list(filter(None, o_cdi_ref[:]))) == 1
    assert o_cdi_ref[0] == expected_cdi

    # Check cells have expected CDI but cell[0,0] (no elevation at this point)
    cdi_index, count = np.unique(driver[DtmConstants.CDI_INDEX][:], return_counts=True)
    assert cdi_index[0] == 0
    assert count[0] == expected_cell_count


def remove_tmp_file(path: str):
    if not path is None and os.path.exists(path):
        os.remove(path)


def test_nominal_set_cdi():
    """
    Invoke process SetCdiProcess to set the same CDI on all cell
    """
    CDI = test_nominal_set_cdi.__name__
    try:
        path_i_dtm = make_dtm_with_no_cdi()
        path_o_dtm = tmp.mktemp(suffix=".dtm.nc")

        # Process
        setCDI = SetCdiProcess(i_paths=[path_i_dtm], o_paths=[path_o_dtm], cdi={os.path.basename(path_i_dtm): CDI})
        setCDI()

        with dtm_driver.open_dtm(path_i_dtm) as i_driver, dtm_driver.open_dtm(path_o_dtm) as o_driver:
            # Check elevations and stddev
            assert np.allclose(i_driver[DtmConstants.ELEVATION_NAME][:], o_driver[DtmConstants.ELEVATION_NAME][:])
            assert np.allclose(i_driver[DtmConstants.STDEV][:], o_driver[DtmConstants.STDEV][:])
            # Check CDI
            check_only_1_cdi(o_driver, CDI)

    finally:
        remove_tmp_file(path_i_dtm)
        remove_tmp_file(path_o_dtm)


def test_replace_cdi():
    """
    Invoke process SetCdiProcess to replace the same CDI on all cell
    """
    OLD_CDI = "OLD_" + test_replace_cdi.__name__
    NEW_CDI = "NEW_" + test_replace_cdi.__name__
    path_i_dtm = make_dtm_with_1_cdi(OLD_CDI)
    path_o_dtm = tmp.mktemp(suffix=".dtm.nc")
    try:
        # Process
        setCDI = SetCdiProcess(i_paths=[path_i_dtm], o_paths=[path_o_dtm], cdi={os.path.basename(path_i_dtm): NEW_CDI})
        setCDI()

        with dtm_driver.open_dtm(path_o_dtm) as o_driver:
            # Check CDI
            check_only_1_cdi(o_driver, NEW_CDI)

    finally:
        remove_tmp_file(path_i_dtm)
        remove_tmp_file(path_o_dtm)


def test_set_cdi_with_geo_filter():
    """
    Invoke process SetCdiProcess to
      - create a DTM with no CDI
      - set a CDI over a zone (the southern half)
      - set an other CDI on all empty cells
    """
    FIRST_CDI = "1_" + test_set_cdi_with_geo_filter.__name__
    SECOND_CDI = "2_" + test_set_cdi_with_geo_filter.__name__
    path_i_dtm = make_dtm_with_no_cdi()
    path_o_dtm1 = tmp.mktemp(suffix=".dtm.nc")
    path_o_dtm2 = tmp.mktemp(suffix=".dtm.nc")
    kml_path1 = kml_path2 = None
    try:
        # Generate a KML, covering the southern half
        coord = [[-10.5, 38.5], [-8.5, 38.5], [-8.5, 39.5], [-10.5, 39.5]]
        kml_path1 = kml_generator.create_kml(tmp.gettempdir(), {"zone": coord})

        # Set the FIRST_CDI to the southern cells
        setCDI = SetCdiProcess(
            i_paths=[path_i_dtm], o_paths=[path_o_dtm1], cdi={os.path.basename(path_i_dtm): FIRST_CDI}, mask=[kml_path1]
        )
        setCDI()
        with dtm_driver.open_dtm(path_o_dtm1) as o_driver:
            # Check CDI (49 only because 1 cell has NaN as elevation)
            check_only_1_cdi(o_driver, FIRST_CDI, 49)

        # Generate a KML, covering the all the DTM
        coord = [[-10.5, 38.5], [-8.5, 38.5], [-8.5, 40.5], [-10.5, 40.5]]
        kml_path2 = kml_generator.create_kml(tmp.gettempdir(), {"zone": coord})

        # Set the SECOND_CDI to the northern cells
        setCDI = SetCdiProcess(
            i_paths=[path_o_dtm1],
            o_paths=[path_o_dtm2],
            cdi={os.path.basename(path_o_dtm1): SECOND_CDI},
            mask=[kml_path2],
            cell_without_cdi=True,
        )
        setCDI()
        with dtm_driver.open_dtm(path_o_dtm2) as o_driver:
            # Check SECOND_CDI is has been created
            o_cdi_ref = o_driver[DtmConstants.CDI]
            assert len(list(filter(None, o_cdi_ref[:]))) == 2

            # Check cells have expected CDI
            cdi_index, count = np.unique(o_driver[DtmConstants.CDI_INDEX][:], return_counts=True)
            # Still 49 cells with FIRST_CDI
            assert o_cdi_ref[0] == FIRST_CDI
            assert cdi_index[0] == 0
            assert count[0] == 49
            # An now, 50 cells with SECOND_CDI
            assert o_cdi_ref[1] == SECOND_CDI
            assert cdi_index[1] == 1
            assert count[1] == 50

    finally:
        remove_tmp_file(path_i_dtm)
        remove_tmp_file(path_o_dtm1)
        remove_tmp_file(path_o_dtm2)
        remove_tmp_file(kml_path1)
        remove_tmp_file(kml_path2)
