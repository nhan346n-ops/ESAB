#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile
import unittest

import netCDF4 as nc
import numpy as np

import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.directory_utils as dir_util
from pyat.dtm.transform.add_default_layers import DefaultLayersProcess
from tests.generator.dtm_generator import DtmGenerator


class TestDefaultLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")
        cls.directory = dir_util.get_test_directory()
        generator = DtmGenerator(cls.directory)
        cls.path = generator.create_pattern(
            value=10, pair_impair=0, line_col=1, number=2, allValue=False, except_layer=[-1, -2, -3, -4, -5, -6]
        )
        cls.path2 = generator.create_pattern(
            value=20,
            pair_impair=0,
            line_col=1,
            number=2,
            allValue=False,
            except_layer=[1, 2, 3, 4, 5, 6, 7, 8, -1],
        )

    def test_default_layer(self):
        # Input File Path
        path = self.path
        output = tempfile.mktemp(suffix=DtmConstants.EXTENSION)

        # Parameters
        params = {"i_paths": [path], "o_paths": [output]}

        # Process
        defaultLayers = DefaultLayersProcess(**params)
        defaultLayers()

        # Verify
        with nc.Dataset(output) as o_dataset:
            o_interp = o_dataset[DtmConstants.INTERPOLATION_FLAG]
            o_elevation = o_dataset[DtmConstants.ELEVATION_NAME]

            for i in range(o_elevation.shape[0]):
                for j in range(o_elevation.shape[1]):
                    if np.ma.is_masked(o_elevation[i, j]):
                        self.assertTrue(np.ma.is_masked(o_interp[i, j]))
                    else:
                        self.assertEqual(0, o_interp[i, j])

            # Check CDI
            o_cdi_ref = o_dataset[DtmConstants.CDI]
            self.assertEqual(1, o_cdi_ref.size)
            self.assertEqual("10", o_cdi_ref[0])

        os.remove(output)

    def test_default_layer_2(self):
        # Input File Path
        path = self.path2
        output = tempfile.mktemp(suffix=DtmConstants.EXTENSION)

        # Parameters
        params = {"i_paths": [path], "o_paths": [output]}

        # Process
        defaultLayers = DefaultLayersProcess(**params)
        defaultLayers()

        # Verify
        with nc.Dataset(output) as o_dataset:
            o_interp = o_dataset[DtmConstants.INTERPOLATION_FLAG]
            o_min = o_dataset[DtmConstants.ELEVATION_MIN]
            o_max = o_dataset[DtmConstants.ELEVATION_MAX]
            o_v_c = o_dataset[DtmConstants.VALUE_COUNT]
            o_f_c = o_dataset[DtmConstants.FILTERED_COUNT]
            o_stdev = o_dataset[DtmConstants.STDEV]
            o_cdi = o_dataset[DtmConstants.CDI_INDEX]
            o_elevation = o_dataset[DtmConstants.ELEVATION_NAME]

            for i in range(o_elevation.shape[0]):
                for j in range(o_elevation.shape[1]):
                    if np.ma.is_masked(o_elevation[i, j]):
                        self.assertTrue(np.ma.is_masked(o_interp[i, j]))
                        self.assertTrue(np.ma.is_masked(o_min[i, j]))
                        self.assertTrue(np.ma.is_masked(o_max[i, j]))
                        self.assertTrue(np.ma.is_masked(o_v_c[i, j]))
                        self.assertTrue(np.ma.is_masked(o_stdev[i, j]))
                        self.assertTrue(np.ma.is_masked(o_cdi[i, j]))
                        self.assertTrue(np.ma.is_masked(o_f_c[i, j]))
                    else:
                        self.assertEqual(0, o_interp[i, j])
                        self.assertEqual(20, o_min[i, j])
                        self.assertEqual(20, o_max[i, j])
                        self.assertEqual(1, o_v_c[i, j])
                        self.assertEqual(0, o_stdev[i, j])
                        self.assertTrue(np.ma.is_masked(o_cdi[i, j]))
                        self.assertEqual(0, o_f_c[i, j])

            # Verify history
            self.assertTrue(", process with Python DefaultLayersProcess from " in o_dataset.history)

        os.remove(output)

    @classmethod
    def tearDownClass(cls):
        print(f"End of {cls.__name__}.")
        os.remove(cls.path)
        os.remove(cls.path2)


if __name__ == "__main__":
    unittest.main()
