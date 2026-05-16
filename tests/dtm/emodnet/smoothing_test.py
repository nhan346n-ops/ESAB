#! /usr/bin/env python3
# coding: utf-8

import os
import unittest

import netCDF4 as nc

import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.directory_utils as dir_util
from pyat.dtm.transform.smoothing import SmoothingProcess
from tests.generator.dtm_generator import DtmGenerator
from tests.generator.kml_generator import create_kml


class TestSmoothingProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")
        cls.directory = dir_util.get_test_directory()
        generator = DtmGenerator(cls.directory)
        cls.path = generator.create_pattern_smoothing(value=20, value_2=30)

    def test_smoothing_process(self):
        # Parameters
        i_paths = [self.path]
        params = {"i_paths": i_paths, "overwrite": True}

        # Process
        Smoothing = SmoothingProcess(**params)
        Smoothing.run()

        # Verify
        o_path = self.path[:-3] + "-smoothed" + DtmConstants.EXTENSION_NC

        with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:

            o_elevation = o_dataset[DtmConstants.ELEVATION_NAME][:]
            o_smoothing = o_dataset[DtmConstants.ELEVATION_SMOOTHED_NAME][:]

            i_elevation = i_dataset[DtmConstants.ELEVATION_NAME][:]

            # Verify size
            self.assertEqual(o_elevation.shape[0], i_elevation.shape[0])
            self.assertEqual(o_elevation.shape[1], i_elevation.shape[1])

            # Compare layers
            for row in range(o_elevation.shape[0]):
                for col in range(o_elevation.shape[1]):
                    if col in (0, 15):
                        self.assertEqual(o_smoothing[row, col], (3 * 30 + 3 * 20) / 6)
                    elif col % 2 != 0:
                        self.assertLessEqual(o_smoothing[row, col] - (30 * 6 + 20 * 3) / 9, 1e-4)
                    else:
                        self.assertLessEqual(o_smoothing[row, col] - (30 * 3 + 20 * 6) / 9, 1e-4)

        os.remove(o_path)

    def test_smoothing_zone(self):
        # Parameters
        i_paths = [self.path]
        coord = [[-3.995, 47.005], [-3.987, 47.005], [-3.987, 47.013], [-3.995, 47.013]]
        kml_path = create_kml(self.directory, {"zone": coord})
        params = {"i_paths": i_paths, "mask": [kml_path], "overwrite": True}

        # Process
        Smoothing = SmoothingProcess(**params)
        Smoothing.run()

        # Verify
        o_path = self.path[:-3] + "-smoothed" + DtmConstants.EXTENSION_NC

        with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:

            o_elevation = o_dataset[DtmConstants.ELEVATION_NAME][:]
            o_smoothing = o_dataset[DtmConstants.ELEVATION_SMOOTHED_NAME][:]

            i_elevation = i_dataset[DtmConstants.ELEVATION_NAME][:]

            lat = o_dataset[DtmConstants.DIM_LAT]
            lon = o_dataset[DtmConstants.DIM_LON]

            # Verify size
            self.assertEqual(o_elevation.shape[0], i_elevation.shape[0])
            self.assertEqual(o_elevation.shape[1], i_elevation.shape[1])

            # Compare layers
            for row in range(o_elevation.shape[0]):
                for col in range(o_elevation.shape[1]):
                    if 47.005 < lat[row] < 47.013 and -3.995 < lon[col] < -3.987:
                        if col % 2 != 0:
                            self.assertLessEqual(o_smoothing[row, col] - (30 * 6 + 20 * 3) / 9, 1e-4)
                        else:
                            self.assertLessEqual(o_smoothing[row, col] - (30 * 3 + 20 * 6) / 9, 1e-4)
                    else:
                        self.assertEqual(o_elevation[row, col], i_elevation[row, col])

        os.remove(o_path)
        os.remove(kml_path)

    @classmethod
    def tearDownClass(cls):
        print(f"End of {cls.__name__}.")
        os.remove(cls.path)


if __name__ == "__main__":
    unittest.main()
