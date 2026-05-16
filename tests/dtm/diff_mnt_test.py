#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile as tmp

import numpy as np
from osgeo import gdal

import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.generator.dtm_generator as dtm_generator
from pyat.dtm.analyse.dtm_diff import DiffMnt
from tests.generator.kml_generator import create_kml


def make_dtm(lon: float, lat: float, min_elev: float, max_elev: float) -> str:
    """
    Generate a DTM with min_elev in the first column to max_elev at the last column
             lon°
              |
              +---------------------+
              | min_elev...max_elev |
              | min_elev...max_elev |
              | min_elev...max_elev |
              | min_elev...max_elev |
              | min_elev...max_elev |
        lat°  +---------------------+

    """
    elevations = np.array([np.linspace(min_elev, max_elev, 5)] * 5, dtype=float)
    return dtm_generator.make_dtm_from_SW((lon, lat), 0.05, {DtmConstants.ELEVATION_NAME: elevations})


def test_same_file_diff_mnt():
    """
    Check the diff mnt on the same dtm. Expected difference == 0.0
    """
    path_i_dtm = make_dtm(-10, 48, -100, -50)
    diff_mnt = DiffMnt(reference_file=path_i_dtm, second_file=path_i_dtm, output_dir=tmp.gettempdir())
    diff_mnt()
    dataset = gdal.Open(diff_mnt.output_file)
    assert dataset is not None
    try:
        band = dataset.GetRasterBand(1)
        diff = band.ReadAsArray()
        np.testing.assert_array_equal(diff, 0.0)
    finally:
        del dataset
        try:
            os.remove(diff_mnt.output_file)
        finally:
            pass

        os.remove(path_i_dtm)


def test_nominal_diff_mnt():
    """
    Check the diff mnt on 2 DTMs
    """
    path_ref_dtm = make_dtm(-10, 48, -100, -50)
    path_sec_dtm = make_dtm(-9.9, 48.1, -95, -50)
    diff_mnt = DiffMnt(reference_file=path_ref_dtm, second_file=path_sec_dtm, output_dir=tmp.gettempdir())
    diff_mnt()
    dataset = gdal.Open(diff_mnt.output_file)
    assert dataset is not None
    try:
        band = dataset.GetRasterBand(1)
        diff = band.ReadAsArray()
        # Expected diff :
        # [[  nan   nan   20.  21.25  22.5 ]
        #  [  nan   nan   20.  21.25  22.5 ]
        #  [  nan   nan   20.  21.25  22.5 ]
        #  [  nan   nan   nan   nan   nan]
        #  [  nan   nan   nan   nan   nan]]
        for row in range(3):
            assert diff[row, 2] == 20.0
            assert diff[row, 3] == 21.25
            assert diff[row, 4] == 22.5
        # Other cells must have no value
        unique, counts = np.unique(diff[~np.isnan(diff)], return_counts=True)
        assert len(unique) == 3  # 20.0: 3, 21.25: 3, 22.5: 3, nan: 16
    finally:
        del dataset
        os.remove(diff_mnt.output_file)
        os.remove(path_ref_dtm)
        os.remove(path_sec_dtm)


def test_diff_mnt_with_shape():
    """
    Check the diff mnt on 2 DTMs and a specific zone
    """
    path_ref_dtm = make_dtm(-10, 48, -100, -50)
    path_sec_dtm = make_dtm(-9.9, 48.1, -95, -50)
    coord = [[-9.87, 48.14], [-9.87, 48.30], [-9.7, 48.30], [-9.7, 48.14]]
    path_kml = create_kml(tmp.gettempdir(), {"zone": coord})

    diff_mnt = DiffMnt(
        reference_file=path_ref_dtm, second_file=path_sec_dtm, output_dir=tmp.gettempdir(), mask=[path_kml]
    )
    diff_mnt()
    dataset = gdal.Open(diff_mnt.output_file)
    assert dataset is not None
    try:
        band = dataset.GetRasterBand(1)
        diff = band.ReadAsArray()
        print(diff)
        # Expected diff :
        # [[  nan   nan   nan  21.25  22.5 ]
        #  [  nan   nan   nan  21.25  22.5 ]
        #  [  nan   nan   nan   nan   nan]
        #  [  nan   nan   nan   nan   nan]
        #  [  nan   nan   nan   nan   nan]]
        for row in range(2):
            assert diff[row, 3] == 21.25
            assert diff[row, 4] == 22.5
        # Other cells must have no value
        unique, counts = np.unique(diff[~np.isnan(diff)], return_counts=True)
        assert len(unique) == 2  # 21.25: 2, 22.5: 2, nan: 21

    finally:
        del dataset
        os.remove(diff_mnt.output_file)
        os.remove(path_ref_dtm)
        os.remove(path_sec_dtm)
        os.remove(path_kml)
