#! /usr/bin/env python3
# coding: utf-8

import os
import unittest

import netCDF4 as nc
import numpy as np

import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.directory_utils as dir_util
from pyat.dtm.transform.reduction import ReductionProcess
from tests.generator.dtm_generator import DtmGenerator, geoBox1


class TestReduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")
        cls.directory = dir_util.get_test_directory()
        generator = DtmGenerator(cls.directory)
        cls.path = []
        cls.path.append(generator.create_reduction_file())
        cls.path.append(generator.create_1(value=10))
        cls.path.append(
            generator.create_long_lat(
                geoBox=geoBox1, zones=np.array([[3, 5, 1, 3], [8, 15, 0, 7]]), values=[20, 10], opt="zone"
            )
        )
        cls.path.append(
            generator.create_long_lat(
                geoBox=geoBox1, zones=np.array([[1, 3, 1, 3], [10, 15, 10, 12]]), values=[10, 20], opt="zone"
            )
        )
        cls.path.append(generator.create_1_without_value_count(value=10))

    def test_reduction_1(self):
        # Parameters
        name = "generated_16x16_10"
        i_paths = [self.path[1]]
        o_path = os.path.join(self.directory, name + "-reduced_4" + DtmConstants.EXTENSION_NC)
        params = {"i_paths": i_paths, "overwrite": True}

        # Process
        reduction = ReductionProcess(**params)
        reduction()

        # Verify
        with nc.Dataset(o_path) as dataset:
            elevation = dataset[DtmConstants.ELEVATION_NAME]
            self.assertEqual(elevation.shape[0], 4)
            self.assertEqual(elevation.shape[1], 4)

        if os.path.exists(o_path):
            os.remove(o_path)

    def test_reduction_2(self):
        # Parameters
        name = "generated_longlat_16x16_zone_[3 5 1 3]_20_[ 8 15  0  7]_10"
        path = os.path.join(self.directory, name + ".nc")
        i_paths = [path]
        factor = 2
        params = {"i_paths": i_paths, "factor": factor, "overwrite": True}

        # Output file path
        o_path = os.path.join(self.directory, name + "-reduced_" + str(factor) + DtmConstants.EXTENSION_NC)

        # Process
        reduction = ReductionProcess(**params)
        reduction()

        # Verify
        with nc.Dataset(o_path) as dataset:
            elevation = dataset[DtmConstants.ELEVATION_NAME]

            self.assertEqual(elevation.shape[0], 8)
            self.assertEqual(elevation.shape[1], 8)

            for r in range(elevation.shape[0]):
                for c in range(elevation.shape[1]):

                    # Check zone 1-2 / 0-1
                    if r in (1, 2) and c in (0, 1):
                        self.assertEqual(20, elevation[r, c])

                    # Check zone 4-7 / 0-3
                    elif r in (4, 5, 6, 7) and c in (0, 1, 2, 3):
                        self.assertEqual(10, elevation[r, c])

        if os.path.exists(o_path):
            os.remove(o_path)

    def test_reduction_3(self):
        # Parameters
        name = "generated_longlat_16x16_zone_[1 3 1 3]_10_[10 15 10 12]_20"
        path = os.path.join(self.directory, name + ".nc")
        i_paths = [path]
        factor = 2
        params = {"i_paths": i_paths, "factor": factor, "overwrite": True}

        # Output file path
        o_path = os.path.join(self.directory, name + "-reduced_" + str(factor) + DtmConstants.EXTENSION_NC)

        # Process
        reduction = ReductionProcess(**params)
        reduction()

        # Verify
        with nc.Dataset(o_path) as dataset:
            elevation = dataset[DtmConstants.ELEVATION_NAME]

            self.assertEqual(elevation.shape[0], 8)
            self.assertEqual(elevation.shape[1], 8)

            for r in range(elevation.shape[0]):
                for c in range(elevation.shape[1]):

                    # Check zone 0-1 / 0-1
                    if r in (0, 1) and c in (0, 1):
                        self.assertEqual(10, elevation[r, c])

                    # Check zone 5-7 / 5-6
                    elif r in (5, 6, 7) and c in (5, 6):
                        self.assertEqual(20, elevation[r, c])

        if os.path.exists(o_path):
            os.remove(o_path)

    def test_reduction_4(self):
        # Parameters
        name = "generated_16x16_reduction_file"
        path = os.path.join(self.directory, name + ".nc")
        i_paths = [path]
        factor = 2
        params = {"i_paths": i_paths, "factor": factor, "overwrite": True}

        # Output file path
        o_path = os.path.join(self.directory, name + "-reduced_" + str(factor) + DtmConstants.EXTENSION_NC)

        # Process
        reduction = ReductionProcess(**params)
        reduction()

        # Verify
        precision = 1e-4
        with nc.Dataset(o_path) as dataset:
            elevation = dataset[DtmConstants.ELEVATION_NAME]
            elevation_max = dataset[DtmConstants.ELEVATION_MAX]
            elevation_min = dataset[DtmConstants.ELEVATION_MIN]
            cdi_index = dataset[DtmConstants.CDI_INDEX]
            cdi_ref = dataset[DtmConstants.CDI]
            stdev = dataset[DtmConstants.STDEV]
            value_count = dataset[DtmConstants.VALUE_COUNT]

            self.assertEqual(elevation.shape[0], 8)
            self.assertEqual(elevation.shape[1], 8)

            self.assertEqual(206, value_count[0, 0])
            self.assertLessEqual((1**2 + 2**2 + 101**2 + 102**2) / 206.0 - elevation[0, 0], precision)
            self.assertEqual(0, elevation_min[0, 0])
            self.assertEqual(103, elevation_max[0, 0])

            cdi_102 = np.where(cdi_ref[:] == "102")
            self.assertTrue(cdi_102)

            self.assertEqual(cdi_102, cdi_index[0, 0])

            self.assertLessEqual(((1**3 + 2**3 + 101**3 + 102**3) / 206) ** 0.5 - stdev[0, 0], precision)

        if os.path.exists(o_path):
            os.remove(o_path)

    def test_reduction_without_value_count(self):
        # Parameters
        name = "generated_16x16_10_without_value_count"
        i_paths = [self.path[4]]
        o_path = os.path.join(self.directory, name + "-reduced_4" + DtmConstants.EXTENSION_NC)

        # Remove output file if it already exists
        if os.path.exists(o_path):
            os.remove(o_path)

        params = {"i_paths": i_paths, "overwrite": True}

        # Process
        reduction = ReductionProcess(**params)
        reduction()

        # Verify
        self.assertTrue(os.path.exists(o_path))
        with nc.Dataset(o_path) as dataset:
            elevation = dataset[DtmConstants.ELEVATION_NAME]
            self.assertEqual(elevation.shape[0], 4)
            self.assertEqual(elevation.shape[1], 4)
            self.assertFalse(np.isnan(np.sum(elevation)))
            self.assertNotIn(DtmConstants.VALUE_COUNT, dataset.variables)


if __name__ == "__main__":
    unittest.main()
