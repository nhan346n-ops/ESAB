#! /usr/bin/env python3
# coding: utf-8

import os
import unittest

import netCDF4 as nc
import numpy as np
import pandas

import pyat.xyz.xyz_constants as XyzConstants
import tests.directory_utils as dir
from pyat.xyz.xyz2dtm import Xyz2Dtm
from pyat.xyz.xyz_file import XyzFile
from tests.generator.xyz_generator import GEOBOX_1, XyzGenerator, RESOLUTION


class TestXyz2Dtm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")
        generator = XyzGenerator(dir.get_output_directory())
        cls.n_files = 5
        cls.paths = []
        for i in range(cls.n_files):
            cls.paths.append(generator.create_file("test_" + str(i), GEOBOX_1))

    def test_xyz2dtm_unique(self):
        # Parameters
        i_path = self.paths[0]
        o_path = self.paths[0] + ".dtm.nc"
        params = {"i_paths": [i_path], "o_paths": [o_path],"target_resolution": RESOLUTION}

        # Process
        process = Xyz2Dtm(**params)
        process()

        # Verify
        data = pandas.read_csv(
            self.paths[0], names=XyzFile.ColumnNames, delimiter=";", header=None, usecols=XyzFile.ColumnNames
        )

        with nc.Dataset(o_path, mode="r") as nc_data:
            self.assertTrue("elevation" in nc_data.variables)
            for name, value in nc_data.variables.items():
                if not name in ["lon", "lat", "crs", "cdi_reference"]:
                    for row in range(value.shape[0]):
                        for col in range(value.shape[1]):
                            if value.shape[0] == 1:
                                element = col
                            else:
                                element = row * value.shape[1] + col

                            if not np.isnan(data[XyzConstants.COL_DEPTH].values[element]):
                                self.assertLessEqual(
                                    nc_data[name][row, col] - data[XyzConstants.COL_DEPTH].values[element], 1e-3
                                )
        os.remove(o_path)

    def test_xyz2dtm_multiple(self):
        # Parameters
        i_paths = self.paths[1:]
        o_paths = [i_path + ".dtm.nc" for i_path in i_paths]
        params = {"i_paths": i_paths, "o_paths": o_paths, "target_resolution": RESOLUTION}

        process = Xyz2Dtm(**params)
        process()

        # Verify
        for i_path, o_path in zip(i_paths, o_paths):
            print(f"Verify {o_path} from {i_path}")
            data = pandas.read_csv(
                i_path, names=XyzFile.ColumnNames, delimiter=";", header=None, usecols=XyzFile.ColumnNames
            )
            with nc.Dataset(o_path, mode="r") as nc_data:
                self.assertTrue("elevation" in nc_data.variables)
                for name, value in nc_data.variables.items():
                    if not name in ["lon", "lat", "crs", "cdi_reference"]:
                        for row in range(value.shape[0]):
                            for col in range(value.shape[1]):
                                if value.shape[0] == 1:
                                    element = col
                                else:
                                    element = row * value.shape[1] + col

                                if not np.isnan(data[XyzConstants.COL_DEPTH].values[element]):
                                    self.assertLessEqual(
                                        nc_data[name][row, col] - data[XyzConstants.COL_DEPTH].values[element], 1e-3
                                    )
            os.remove(o_path)

    @classmethod
    def tearDownClass(cls):
        for path in cls.paths:
            os.remove(path)
        print(f"End of {cls.__name__}.")


if __name__ == "__main__":
    unittest.main()
