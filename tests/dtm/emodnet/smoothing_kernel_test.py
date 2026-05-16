#! /usr/bin/env python3
# coding: utf-8

import os
import unittest

import netCDF4 as nc
import numpy as np

import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.directory_utils as dir_util
from pyat.dtm.transform.smoothing_kernel import SmoothingProcess
from tests.generator.dtm_generator import DtmGenerator
from tests.generator.kml_generator import create_kml


class TestSmoothingProcessKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")
        cls.directory = dir_util.get_test_directory()
        generator = DtmGenerator(cls.directory)
        cls.path = generator.create_pattern_smoothing(value=20, value_2=30)

    def test_smoothing_process_flat(self):
        # Parameters
        i_paths = [self.path]
        params = {"i_paths": i_paths, "kernel_choice": "3x3 flat", "overwrite": True}

        # Process
        smoothing = SmoothingProcess(**params)
        smoothing()

        # Verify
        o_path = self.path[:-3] + "-smoothed" + DtmConstants.EXTENSION_NC

        with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:

            o_elevation = o_dataset[DtmConstants.ELEVATION_NAME][:]
            i_elevation = i_dataset[DtmConstants.ELEVATION_NAME][:]

            # Verify size
            self.assertEqual(o_elevation.shape[0], i_elevation.shape[0])
            self.assertEqual(o_elevation.shape[1], i_elevation.shape[1])

            # Compare layers
            for row in range(o_elevation.shape[0]):
                for col in range(o_elevation.shape[1]):
                    # 3x3 flat computation
                    elevations = [self.get_elevation(i_elevation, row - 1, col - 1),
                                  self.get_elevation(i_elevation, row - 1, col),
                                  self.get_elevation(i_elevation, row - 1, col + 1),
                                  self.get_elevation(i_elevation, row, col - 1),
                                  self.get_elevation(i_elevation, row, col) * 4,
                                  self.get_elevation(i_elevation, row, col + 1),
                                  self.get_elevation(i_elevation, row + 1, col - 1),
                                  self.get_elevation(i_elevation, row + 1, col),
                                  self.get_elevation(i_elevation, row + 1, col + 1)]
                    elevation_sum = np.nansum(elevations)
                    elevation_kernel = np.count_nonzero(np.nan_to_num(elevations)) + 3
                    self.assertAlmostEqual(
                        o_elevation[row, col],
                        elevation_sum / elevation_kernel,
                        places=5,
                    )

        os.remove(o_path)

    def test_smoothing_process_x_with_mask(self):
        # Parameters
        coord = [[-3.995, 47.005], [-3.987, 47.005], [-3.987, 47.013], [-3.995, 47.013]]
        kml_path = create_kml(self.directory, {"zone": coord})
        params = {"i_paths": [self.path], "kernel_choice": "3x3 X", "mask": [kml_path], "overwrite": True}

        # Process
        smoothing = SmoothingProcess(**params)
        smoothing()

        # Verify
        o_path = self.path[:-3] + "-smoothed" + DtmConstants.EXTENSION_NC

        with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:

            o_elevation = o_dataset[DtmConstants.ELEVATION_NAME][:]
            i_elevation = i_dataset[DtmConstants.ELEVATION_NAME][:]

            # Verify size
            self.assertEqual(o_elevation.shape[0], i_elevation.shape[0])
            self.assertEqual(o_elevation.shape[1], i_elevation.shape[1])

            lat = o_dataset[DtmConstants.DIM_LAT]
            lon = o_dataset[DtmConstants.DIM_LON]

            # Compare layers
            for row in range(o_elevation.shape[0]):
                for col in range(o_elevation.shape[1]):
                    elevations = [  # 3x3 X computation
                        self.get_elevation(i_elevation, row - 1, col - 1),
                        self.get_elevation(i_elevation, row - 1, col + 1),
                        self.get_elevation(i_elevation, row, col) * 4,
                        self.get_elevation(i_elevation, row + 1, col - 1),
                        self.get_elevation(i_elevation, row + 1, col + 1)]
                    elevation_sum = np.nansum(elevations)
                    elevation_kernel = np.count_nonzero(np.nan_to_num(elevations)) + 3

                    if 47.005 <= lat[row] <= 47.013 and -3.995 <= lon[col] <= -3.987:
                        self.assertAlmostEqual(
                            o_elevation[row, col],
                            elevation_sum / elevation_kernel,
                            places=5,
                        )

        os.remove(o_path)
        os.remove(kml_path)

    def get_elevation(self, elevation, row: int, col: int) -> float:
        if row < 0 or row >= elevation.shape[0] or col < 0 or col >= elevation.shape[1]:
            return 0.0
        return elevation[row, col]

    @classmethod
    def tearDownClass(cls):
        print(f"End of {cls.__name__}.")
        os.remove(cls.path)


if __name__ == "__main__":
    unittest.main()
