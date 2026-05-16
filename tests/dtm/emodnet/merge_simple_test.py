#! /usr/bin/env python3
# coding: utf-8

import os
import unittest

import netCDF4 as nc
import numpy as np

import pyat.common.geo_file as gf
import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.directory_utils as dir_util
from pyat.dtm.merge.merge_simple import MergeSimpleProcess
from tests.generator.dtm_generator import DtmGenerator


class TestMergeSimple(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")

    def setUp(self):
        print(f"Start of {self.__class__.__name__}.")
        self.directory = dir_util.get_test_directory()
        self.generator = DtmGenerator(self.directory)
        # Input paths
        self.paths = []
        # Output File Path
        self.o_path = os.path.join(self.directory, "merged_simple" + DtmConstants.EXTENSION_NC)

    def test_simple_merge(self):
        self.paths = [
            self.generator.create_pattern(value=10, spatial_reference=gf.SR_WGS_84),
            self.generator.create_pattern(
                value=20, spatial_reference=gf.SR_WGS_84, pair_impair=1, line_col=1, number=2, allValue=False
            ),
            self.generator.create_pattern(
                value=30, spatial_reference=gf.SR_WGS_84, pair_impair=0, line_col=1, number=2, allValue=False
            ),
        ]

        # Parameters
        params = {"i_paths": self.paths, "o_path": self.o_path, "overwrite": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        merge = MergeSimpleProcess(coord=geobox, **params)
        merge()

        with nc.Dataset(self.o_path) as o_file:
            self.assertEqual(o_file[DtmConstants.CRS_NAME].__dict__["grid_mapping_name"], "latitude_longitude")
            self.check_merged_dtm(o_file)

    def test_projected_simple_merge(self):
        self.paths = [
            self.generator.create_pattern(value=10, spatial_reference=gf.SR_PSEUDO_MERCATOR),
            self.generator.create_pattern(
                value=20, spatial_reference=gf.SR_PSEUDO_MERCATOR, pair_impair=1, line_col=1, number=2, allValue=False
            ),
            self.generator.create_pattern(
                value=30, spatial_reference=gf.SR_PSEUDO_MERCATOR, pair_impair=0, line_col=1, number=2, allValue=False
            ),
        ]

        # Parameters
        params = {"i_paths": self.paths, "o_path": self.o_path, "overwrite": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        merge = MergeSimpleProcess(coord=geobox, **params)
        merge()

        with nc.Dataset(self.o_path) as o_file:
            self.assertEqual(o_file[DtmConstants.CRS_NAME].__dict__["grid_mapping_name"], "mercator")
            self.check_merged_dtm(o_file)

    def check_merged_dtm(self, o_file: nc.Dataset):
        layer_elevation = o_file[DtmConstants.ELEVATION_NAME][:]
        layer_elevation_min = o_file[DtmConstants.ELEVATION_MIN][:]
        layer_elevation_max = o_file[DtmConstants.ELEVATION_MAX][:]
        layer_cdi_index = o_file[DtmConstants.CDI_INDEX][:]
        layer_stdev = o_file[DtmConstants.STDEV][:]
        layer_value_count = o_file[DtmConstants.VALUE_COUNT][:]
        layer_cdi = o_file[DtmConstants.CDI][:]

        cdi_30 = np.where(layer_cdi[:] == "30")
        cdi_20 = np.where(layer_cdi[:] == "20")

        self.assertTrue(cdi_30)
        self.assertTrue(cdi_20)
        self.assertNotEqual(cdi_30, cdi_20)

        for r in range(layer_elevation.shape[0]):
            for c in range(layer_elevation.shape[1]):
                if not c % 2:
                    self.assertEqual(40, layer_value_count[r, c])
                    self.assertEqual((10 + 30) / 2, layer_elevation[r, c])
                    self.assertEqual(31, layer_elevation_max[r, c])
                    self.assertEqual(9, layer_elevation_min[r, c])
                    self.assertLessEqual(((1**2 + 3**2) / 2.0) ** 0.5, layer_stdev[r, c], 1e-3)
                    self.assertEqual(cdi_30[0], layer_cdi_index[r, c])
                else:
                    self.assertEqual(30.0, layer_value_count[r, c])
                    self.assertEqual((10 + 20) / 2, layer_elevation[r, c])
                    self.assertEqual(21, layer_elevation_max[r, c])
                    self.assertEqual(9, layer_elevation_min[r, c])
                    self.assertLessEqual(((1**2 + 2**2) / 2) ** 0.5, layer_stdev[r, c], 1e-3)
                    self.assertEqual(cdi_20[0], layer_cdi_index[r, c])

    def test_merge_simple_cdi(self):
        self.paths = [
            self.generator.create_pattern_for_cdi_test(value=6, mode=0, value_count=10, cdi="test"),
            self.generator.create_pattern_for_cdi_test(value=5, mode=0, value_count=20, cdi="test2"),
        ]
        # Parameters
        params = {"i_paths": self.paths, "overwrite": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        merge = MergeSimpleProcess(coord=geobox, **params)
        merge()

        # Verify merge
        with nc.Dataset(self.o_path, "r") as o_file:
            o_cdi = o_file[DtmConstants.CDI_INDEX][:]
            o_cdi_ref = o_file[DtmConstants.CDI][:]

            self.assertEqual("test2", o_cdi_ref[0])
            self.assertEqual("", o_cdi_ref[1])

            for row in range(o_cdi.shape[0]):
                for col in range(o_cdi.shape[1]):
                    # Because the value count of the second file is bigger than the first one.
                    self.assertEqual(0, o_cdi[row, col])

    def tearDown(self):
        if os.path.exists(self.o_path):
            os.remove(self.o_path)

        for path in self.paths:
            if os.path.exists(path):
                os.remove(path)

    @classmethod
    def tearDownClass(cls):
        print(f"End of {cls.__name__}.")


if __name__ == "__main__":
    unittest.main()
