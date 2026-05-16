#! /usr/bin/env python3
# coding: utf-8

import os
import unittest

import netCDF4 as nc
import numpy as np

import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.directory_utils as dir_util
from pyat.dtm.transform.set_interpolated import SetInterpolatedProcess
from tests.generator.dtm_generator import DtmGenerator
from tests.generator.kml_generator import create_kml


class TestSetInterpolated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")
        cls.directory = dir_util.get_test_directory()
        generator = DtmGenerator(cls.directory)
        cls.path = generator.create_set_interpolation_file()

    def test_set_interpolated_no_mask(self):
        # Parameters
        i_paths = [self.path]
        params = {"i_paths": i_paths}

        # Process
        setInterpolated = SetInterpolatedProcess(**params)
        setInterpolated()

        # Verify
        o_path = self.path[:-3] + "-interpolated" + DtmConstants.EXTENSION_NC
        try:
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                for layer in [DtmConstants.INTERPOLATION_FLAG]:
                    data = o_dataset[layer][:]
                    i_data = i_dataset[layer][:]
                    for row in range(data.shape[0]):
                        for col in range(data.shape[1]):
                            if np.ma.is_masked(data[row, col]):
                                self.assertTrue(
                                    np.ma.is_masked(i_data[row, col]),
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                            else:
                                self.assertEqual(
                                    data[row, col],
                                    1,
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
        finally:
            os.remove(o_path)

    def test_set_interpolated_geo_zone(self):
        # Parameters
        i_paths = [self.path]
        suffix = "-interpolated_single_geo_zone"
        o_path = self.path[:-3] + suffix + DtmConstants.EXTENSION_NC
        lat_max = 47.1
        lat_min = 47
        lon_max = -3.9
        lon_min = -4
        coord = [[lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max]]
        kml_path = create_kml(self.directory, {"z": coord})
        params = {"i_paths": i_paths, "mask": [kml_path], "o_paths": [o_path]}

        # Process
        setInterpolated = SetInterpolatedProcess(**params)
        setInterpolated()

        # Verify
        try:
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                for layer in [DtmConstants.INTERPOLATION_FLAG]:
                    data = o_dataset[layer][:]
                    i_data = i_dataset[layer][:]
                    lat = o_dataset[DtmConstants.DIM_LAT][:]
                    lon = o_dataset[DtmConstants.DIM_LON][:]

                    for row in range(data.shape[0]):
                        for col in range(data.shape[1]):
                            cell_lat = lat[row]
                            cell_lon = lon[col]
                            if np.ma.is_masked(data[row, col]):
                                self.assertTrue(
                                    np.ma.is_masked(i_data[row, col]),
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                            elif lon_min <= cell_lon <= lon_max and lat_min <= cell_lat <= lat_max:
                                self.assertEqual(
                                    data[row, col],
                                    1,
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                            else:
                                self.assertEqual(
                                    data[row, col],
                                    i_data[row, col],
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
        finally:
            os.remove(o_path)
            os.remove(kml_path)

    def test_set_interpolated_geo_zone_kml_multiple(self):
        # Parameters
        i_paths = [self.path]
        o_path = self.path[:-3] + "-interpolated-multiple-kml" + DtmConstants.EXTENSION_NC
        coord_1 = [[-4.001, 47.017], [-3.992, 47.017], [-3.992, 47.010], [-4.001, 47.010]]
        coord_2 = [[-3.989, 47.006], [-3.982, 47.006], [-3.982, 46.999], [-3.989, 46.999]]
        kml_1 = create_kml(self.directory, {"A": coord_1})
        kml_2 = create_kml(self.directory, {"B": coord_2})
        kmls = [kml_1, kml_2]
        params = {"i_paths": i_paths, "o_paths": [o_path], "mask": kmls}

        # Process
        setInterpolated = SetInterpolatedProcess(**params)
        setInterpolated()
        try:
            # Verify
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                for layer in [DtmConstants.INTERPOLATION_FLAG]:
                    data = o_dataset[layer][:]
                    i_data = i_dataset[layer][:]
                    lat = o_dataset[DtmConstants.DIM_LAT][:]
                    lon = o_dataset[DtmConstants.DIM_LON][:]

                    for row in range(data.shape[0]):
                        for col in range(data.shape[1]):
                            cell_lat = lat[row]
                            cell_lon = lon[col]
                            if np.ma.is_masked(data[row, col]):
                                self.assertTrue(
                                    np.ma.is_masked(i_data[row, col]),
                                    msg=f"Error while checking cell[{row},{col}] coord(lon,lat)=({cell_lon},{cell_lat})",
                                )
                            elif -4.001 <= cell_lon <= -3.992 and 47.010 <= cell_lat <= 47.017:
                                self.assertEqual(
                                    data[row, col],
                                    1,
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                            elif -3.989 <= cell_lon <= -3.982 and 46.999 <= cell_lat <= 47.006:
                                self.assertEqual(
                                    data[row, col],
                                    1,
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                            else:
                                self.assertEqual(
                                    data[row, col],
                                    i_data[row, col],
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
        finally:
            os.remove(o_path)
            os.remove(kml_1)
            os.remove(kml_2)

    def test_set_interpolated_geo_kml_zone_multiple(self):
        # Parameters
        i_paths = [self.path]
        o_path = self.path[:-3] + "-interpolated-multiple-zone" + DtmConstants.EXTENSION_NC
        coord_1 = [[-4.001, 47.017], [-3.992, 47.017], [-3.992, 47.010], [-4.001, 47.010]]
        coord_2 = [[-3.989, 47.006], [-3.982, 47.006], [-3.982, 46.999], [-3.989, 46.999]]
        coords = {"zone1": coord_1, "zone2": coord_2}
        kml_1 = create_kml(dir=self.directory, coords=coords)
        kml = [kml_1]
        params = {"i_paths": i_paths, "o_paths": [o_path], "mask": kml}

        # Process
        setInterpolated = SetInterpolatedProcess(**params)
        setInterpolated()

        # Verify
        try:
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                for layer in [DtmConstants.INTERPOLATION_FLAG]:
                    data = o_dataset[layer][:]
                    i_data = i_dataset[layer][:]
                    lat = o_dataset[DtmConstants.DIM_LAT][:]
                    lon = o_dataset[DtmConstants.DIM_LON][:]

                    for row in range(data.shape[0]):
                        for col in range(data.shape[1]):
                            cell_lat = lat[row]
                            cell_lon = lon[col]
                            # check zone1
                            if np.ma.is_masked(data[row, col]):
                                self.assertTrue(
                                    np.ma.is_masked(i_data[row, col]),
                                    msg=f"Error while checking cell[{row},{col}] coord(lon,lat)=({cell_lon},{cell_lat})",
                                )
                            elif -4.001 <= cell_lon <= -3.992 and 47.010 <= cell_lat <= 47.017:
                                self.assertEqual(
                                    data[row, col],
                                    1,
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                            elif -3.989 <= cell_lon <= -3.982 and 46.999 <= cell_lat <= 47.006:
                                self.assertEqual(
                                    data[row, col],
                                    1,
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                            else:
                                self.assertEqual(
                                    data[row, col],
                                    i_data[row, col],
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
        finally:
            os.remove(o_path)
            os.remove(kml_1)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.path)
        print(f"End of {cls.__name__}.")


if __name__ == "__main__":
    unittest.main()
