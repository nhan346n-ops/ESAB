#! /usr/bin/env python3
# coding: utf-8

import os
import unittest

import netCDF4 as nc
import numpy as np

import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.directory_utils as dir_util
from pyat.dtm.transform.linear_transform import LinearTransformProcess
from tests.generator.dtm_generator import DtmGenerator
from tests.generator.kml_generator import create_kml


class TestLinearTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")
        cls.directory = dir_util.get_test_directory()
        generator = DtmGenerator(cls.directory)
        cls.path = generator.create_pattern(value=10)

    def test_linear_transform_positif(self):
        # Parameters
        a = 2
        b = -5
        i_paths = [self.path]
        params = {"i_paths": i_paths, "a": a, "b": b, "overwrite": True}

        o_path = self.path[:-3] + "-linear_transform" + DtmConstants.EXTENSION_NC

        # Process
        linearTransform = LinearTransformProcess(**params)
        linearTransform()

        # Verify
        with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:

            o_elevation = o_dataset[DtmConstants.ELEVATION_NAME][:]
            o_max = o_dataset[DtmConstants.ELEVATION_MAX][:]
            o_min = o_dataset[DtmConstants.ELEVATION_MIN][:]
            o_stdev = o_dataset[DtmConstants.STDEV][:]
            o_value_count = o_dataset[DtmConstants.VALUE_COUNT][:]

            i_elevation = i_dataset[DtmConstants.ELEVATION_NAME][:]
            i_max = i_dataset[DtmConstants.ELEVATION_MAX][:]
            i_min = i_dataset[DtmConstants.ELEVATION_MIN][:]
            i_stdev = i_dataset[DtmConstants.STDEV][:]
            i_value_count = i_dataset[DtmConstants.VALUE_COUNT][:]

            # Verify size
            self.assertEqual(o_elevation.shape[0], i_elevation.shape[0])
            self.assertEqual(o_elevation.shape[1], i_elevation.shape[1])

            # Compare layers
            for row in range(o_elevation.shape[0]):
                for col in range(o_elevation.shape[1]):
                    if not np.isnan(o_elevation[row, col]):
                        self.assertEqual(o_elevation[row, col], i_elevation[row, col] * a + b)
                        if a > 0:
                            self.assertEqual(o_max[row, col], i_max[row, col] * a + b)
                            self.assertEqual(o_min[row, col], i_min[row, col] * a + b)
                        else:
                            self.assertEqual(o_max[row, col], i_min[row, col] * a + b)
                            self.assertEqual(o_min[row, col], i_max[row, col] * a + b)
                        self.assertEqual(o_stdev[row, col], i_stdev[row, col] * abs(a))
                        self.assertEqual(o_value_count[row, col], i_value_count[row, col])
                    else:
                        self.assertTrue(np.isnan(i_elevation[row, col]))
                        self.assertTrue(np.isnan(i_max[row, col]))
                        self.assertTrue(np.isnan(i_min[row, col]))
                        self.assertTrue(np.isnan(i_stdev[row, col]))
                        self.assertTrue(np.isnan(i_value_count[row, col]))

        os.remove(o_path)

    def test_linear_transform_zone(self):
        # Parameters
        a = 2
        b = -5
        i_paths = [self.path]
        lat_max = 47.0141
        lat_min = 47.0027
        lon_max = -3.987
        lon_min = -3.995
        coord = [[lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max]]
        kml_path = create_kml(self.directory, {"zone": coord})
        params = {"i_paths": i_paths, "a": a, "b": b, "mask": [kml_path]}

        o_path = self.path[:-3] + "-linear_transform" + DtmConstants.EXTENSION_NC

        # Process
        linearTransform = LinearTransformProcess(**params)
        linearTransform()

        # Verify
        with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
            o_elevation = o_dataset[DtmConstants.ELEVATION_NAME][:]
            o_max = o_dataset[DtmConstants.ELEVATION_MAX][:]
            o_min = o_dataset[DtmConstants.ELEVATION_MIN][:]
            o_stdev = o_dataset[DtmConstants.STDEV][:]
            o_value_count = o_dataset[DtmConstants.VALUE_COUNT][:]
            lat = o_dataset[DtmConstants.DIM_LAT][:]
            lon = o_dataset[DtmConstants.DIM_LON][:]

            i_elevation = i_dataset[DtmConstants.ELEVATION_NAME][:]
            i_max = i_dataset[DtmConstants.ELEVATION_MAX][:]
            i_min = i_dataset[DtmConstants.ELEVATION_MIN][:]
            i_stdev = i_dataset[DtmConstants.STDEV][:]
            i_value_count = i_dataset[DtmConstants.VALUE_COUNT][:]

            # Verify size
            self.assertEqual(o_elevation.shape[0], i_elevation.shape[0])
            self.assertEqual(o_elevation.shape[1], i_elevation.shape[1])

            for row in range(o_elevation.shape[0]):
                for col in range(o_elevation.shape[1]):
                    cell_lat = lat[row]
                    cell_lon = lon[col]
                    if lat_min < cell_lat < lat_max and lon_min < cell_lon < lon_max:
                        if not np.isnan(o_elevation[row, col]):
                            self.assertEqual(
                                o_elevation[row, col],
                                i_elevation[row, col] * a + b,
                                msg=f"Error while checking cell[{row},{col}]",
                            )
                            if a > 0:
                                self.assertEqual(
                                    o_max[row, col],
                                    i_max[row, col] * a + b,
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                                self.assertEqual(
                                    o_min[row, col],
                                    i_min[row, col] * a + b,
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                            else:
                                self.assertEqual(
                                    o_max[row, col],
                                    i_min[row, col] * a + b,
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                                self.assertEqual(
                                    o_min[row, col],
                                    i_max[row, col] * a + b,
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                            self.assertEqual(
                                o_stdev[row, col],
                                i_stdev[row, col] * abs(a),
                                msg=f"Error while checking cell[{row},{col}]",
                            )
                            self.assertEqual(
                                o_value_count[row, col],
                                i_value_count[row, col],
                                msg=f"Error while checking cell[{row},{col}]",
                            )
                        else:
                            self.assertTrue(
                                np.isnan(i_elevation[row, col], msg=f"Error while checking cell[{row},{col}]")
                            )
                            self.assertTrue(np.isnan(i_max[row, col], msg=f"Error while checking cell[{row},{col}]"))
                            self.assertTrue(np.isnan(i_min[row, col], msg=f"Error while checking cell[{row},{col}]"))
                            self.assertTrue(np.isnan(i_stdev[row, col], msg=f"Error while checking cell[{row},{col}]"))
                            self.assertTrue(
                                np.isnan(i_value_count[row, col], msg=f"Error while checking cell[{row},{col}]")
                            )
                    else:
                        if not np.isnan(i_elevation[row, col]):
                            self.assertEqual(o_elevation[row, col], i_elevation[row, col])
                            self.assertEqual(o_max[row, col], i_max[row, col])
                            self.assertEqual(o_min[row, col], i_min[row, col])
                            self.assertEqual(o_stdev[row, col], i_stdev[row, col])
                            self.assertEqual(o_value_count[row, col], i_value_count[row, col])
                        else:
                            self.assertTrue(np.isnan(o_elevation[row, col]))
                            self.assertTrue(np.isnan(o_max[row, col]))
                            self.assertTrue(np.isnan(o_min[row, col]))
                            self.assertTrue(np.isnan(o_stdev[row, col]))
                            self.assertTrue(np.isnan(o_value_count[row, col]))

        os.remove(o_path)
        os.remove(kml_path)

    def test_linear_transform_negatif(self):
        # Parameters
        a = -2
        b = -5
        i_paths = [self.path]
        params = {"i_paths": i_paths, "a": a, "b": b, "overwrite": True}

        o_path = self.path[:-3] + "-linear_transform" + DtmConstants.EXTENSION_NC

        # Process
        linearTransform = LinearTransformProcess(**params)
        linearTransform()

        # Verify
        with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:

            o_elevation = o_dataset[DtmConstants.ELEVATION_NAME][:]
            o_max = o_dataset[DtmConstants.ELEVATION_MAX][:]
            o_min = o_dataset[DtmConstants.ELEVATION_MIN][:]
            o_stdev = o_dataset[DtmConstants.STDEV][:]
            o_value_count = o_dataset[DtmConstants.VALUE_COUNT][:]

            i_elevation = i_dataset[DtmConstants.ELEVATION_NAME][:]
            i_max = i_dataset[DtmConstants.ELEVATION_MAX][:]
            i_min = i_dataset[DtmConstants.ELEVATION_MIN][:]
            i_stdev = i_dataset[DtmConstants.STDEV][:]
            i_value_count = i_dataset[DtmConstants.VALUE_COUNT][:]

            # Verify size
            self.assertEqual(o_elevation.shape[0], i_elevation.shape[0])
            self.assertEqual(o_elevation.shape[1], i_elevation.shape[1])

            # Compare layers
            for row in range(o_elevation.shape[0]):
                for col in range(o_elevation.shape[1]):
                    if not np.isnan(o_elevation[row, col]):
                        self.assertEqual(o_elevation[row, col], i_elevation[row, col] * a + b)
                        if a > 0:
                            self.assertEqual(o_max[row, col], i_max[row, col] * a + b)
                            self.assertEqual(o_min[row, col], i_min[row, col] * a + b)
                        else:
                            self.assertEqual(o_max[row, col], i_min[row, col] * a + b)
                            self.assertEqual(o_min[row, col], i_max[row, col] * a + b)
                        self.assertEqual(o_stdev[row, col], i_stdev[row, col] * abs(a))
                        self.assertEqual(o_value_count[row, col], i_value_count[row, col])
                    else:
                        self.assertTrue(np.isnan(i_elevation[row, col]))
                        self.assertTrue(np.isnan(i_max[row, col]))
                        self.assertTrue(np.isnan(i_min[row, col]))
                        self.assertTrue(np.isnan(i_stdev[row, col]))
                        self.assertTrue(np.isnan(i_value_count[row, col]))

        os.remove(o_path)

    @classmethod
    def tearDownClass(cls):
        print(f"End of {cls.__name__}.")
        os.remove(cls.path)


if __name__ == "__main__":
    unittest.main()
