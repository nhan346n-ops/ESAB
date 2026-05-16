#! /usr/bin/env python3
# coding: utf-8

import os
import unittest

import netCDF4 as nc
import numpy as np

import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.directory_utils as dir_util
from pyat.dtm.analyse.sanity_check_process import SanityCheckProcess
from tests.generator.dtm_generator import DtmGenerator
import pyat.dtm.cdi.cdi_layer_util as cdi_util

class TestSanityCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")
        cls.directory = dir_util.get_test_directory()
        generator = DtmGenerator(cls.directory)
        # mode 1: zeros end ones cdis
        cls.path = generator.create_pattern_sanity_check(size=16, value=20)
        # mode 2: only zeros cdis
        cls.path2 = generator.create_pattern_sanity_check(size=16, value=20, mode=2)
        # mode 3: zeros and undef cdis
        cls.path3 = generator.create_pattern_sanity_check(size=16, value=20, mode=3)

    def test_sanity_check_interp(self):
        # Parameters
        i_paths = [self.path]
        interp = True
        params = {"i_paths": i_paths, "interp": interp, 'overwrite': True}

        # Process
        sanityCheck = SanityCheckProcess(**params)
        sanityCheck()

        # Verify
        o_path = self.path[:-3] + "-cleaned" + DtmConstants.EXTENSION_NC

        with nc.Dataset(o_path) as o_ds, nc.Dataset(self.path) as i_ds:
            o_elev = o_ds[DtmConstants.ELEVATION_NAME][:]
            o_interp = o_ds[DtmConstants.INTERPOLATION_FLAG][:]
            i_interp = i_ds[DtmConstants.INTERPOLATION_FLAG][:]

            for i in range(o_elev.shape[0]):
                for j in range(o_elev.shape[1]):
                    if np.ma.is_masked(o_elev[i, j]):
                        self.assertTrue(np.ma.is_masked(o_interp[i, j]))
                    elif not np.ma.is_masked(i_interp[i, j]):
                        self.assertFalse(np.ma.is_masked(o_interp[i, j]))
                    else:
                        self.assertEqual(0, o_interp[i, j])

        os.remove(o_path)

    def test_sanity_check_cdi_force(self):
        # Parameters2
        i_paths = [self.path]
        cdi = True
        params = {"i_paths": i_paths, "cdi": cdi, 'overwrite': True}

        # Process
        sanityCheck = SanityCheckProcess(**params)
        sanityCheck()

        # Verify
        o_path = self.path[:-3] + "-cleaned" + DtmConstants.EXTENSION_NC

        with nc.Dataset(o_path) as o_ds:
            o_elev = o_ds[DtmConstants.ELEVATION_NAME][:]
            o_cdi_ref = o_ds[DtmConstants.CDI][:]
            o_cdi_index = o_ds[DtmConstants.CDI_INDEX][:]

            #  Check len cdi_ref
            self.assertEqual(len(o_cdi_ref[o_cdi_ref != ""]), 1)

            for i in range(o_elev.shape[0]):
                for j in range(o_elev.shape[1]):
                    if np.ma.is_masked(o_elev[i, j]):
                        self.assertTrue(np.ma.is_masked(o_cdi_index[i, j]))
                    else:
                        self.assertEqual(o_cdi_index[i, j], 0)

        os.remove(o_path)

    def test_sanity_check_cdi_compress(self):
        # Parameters
        i_paths = [self.path]
        cdi = True
        params = {"i_paths": i_paths, "cdi": cdi, 'overwrite': True}

        # Process
        sanityCheck = SanityCheckProcess(**params)
        sanityCheck()

        # Verify
        o_path = self.path[:-3] + "-cleaned" + DtmConstants.EXTENSION_NC

        with nc.Dataset(o_path) as o_ds, nc.Dataset(self.path) as i_ds:
            o_elev = o_ds[DtmConstants.ELEVATION_NAME][:]
            o_cdi_ref = o_ds[DtmConstants.CDI][:]
            o_cdi_index = o_ds[DtmConstants.CDI_INDEX][:]
            i_cdi_index = i_ds[DtmConstants.CDI_INDEX][:]

            # Check cdi_ref size
            self.assertEqual(len(o_cdi_ref[o_cdi_ref != ""]), 1)

            for i in range(o_elev.shape[0]):
                for j in range(o_elev.shape[1]):
                    if np.ma.is_masked(i_cdi_index[i, j]) and np.ma.is_masked(o_elev[i, j]):
                        self.assertTrue(np.ma.is_masked(o_cdi_index[i, j]))
                    else:
                        self.assertEqual(0, o_cdi_index[i, j])

        os.remove(o_path)

    def test_sanity_check_cdi_undefined(self):
        # Parameters
        i_paths = [self.path3]
        cdi = True
        allow_undefined_cdi = True
        params = {"i_paths": i_paths, "cdi": cdi, 'overwrite': True}

        # Process
        sanityCheck = SanityCheckProcess(**params)
        sanityCheck()

        # Verify
        o_path = self.path3[:-3] + "-cleaned" + DtmConstants.EXTENSION_NC

        with nc.Dataset(o_path) as o_ds, nc.Dataset(self.path3) as i_ds:
            o_elev = o_ds[DtmConstants.ELEVATION_NAME][:]
            o_cdi_ref = o_ds[DtmConstants.CDI][:]
            o_cdi_index = o_ds[DtmConstants.CDI_INDEX][:]
            i_cdi_index = i_ds[DtmConstants.CDI_INDEX][:]
            self.assertFalse(cdi_util.check_undefined_cdi(i_ds))
            self.assertFalse(cdi_util.check_undefined_cdi(o_ds))
            # Check cdi_ref size
            self.assertEqual(len(o_cdi_ref[o_cdi_ref != ""]), 1)
            for i in range(o_elev.shape[0]):
                for j in range(o_elev.shape[1]):
                    if np.ma.is_masked(i_cdi_index[i, j]) and np.ma.is_masked(o_elev[i, j]):
                        self.assertTrue(np.ma.is_masked(o_cdi_index[i, j]))
                    elif j == 0: #first col is undef
                        self.assertTrue(np.ma.is_masked(o_cdi_index[i, j]))
                    else:
                        self.assertEqual(0, o_cdi_index[i, j])
        os.remove(o_path)
    def test_sanity_check_all(self):
        # Parameters
        i_paths = [self.path]
        interp = True
        cdi = True
        params = {"i_paths": i_paths, "interp": interp, "cdi": cdi, 'overwrite': True}

        # Process
        sanityCheck = SanityCheckProcess(**params)
        sanityCheck()

        # Verify
        o_path = self.path[:-3] + "-cleaned" + DtmConstants.EXTENSION_NC

        with nc.Dataset(o_path) as o_ds, nc.Dataset(self.path) as i_ds:
            o_elev = o_ds[DtmConstants.ELEVATION_NAME][:]
            o_interp = o_ds[DtmConstants.INTERPOLATION_FLAG][:]
            i_interp = i_ds[DtmConstants.INTERPOLATION_FLAG][:]

            o_cdi_index = o_ds[DtmConstants.CDI_INDEX][:]
            i_cdi_index = i_ds[DtmConstants.CDI_INDEX][:]

            for i in range(o_elev.shape[0]):
                for j in range(o_elev.shape[1]):
                    # Check interpolation
                    if np.ma.is_masked(o_elev[i, j]):
                        self.assertTrue(np.ma.is_masked(o_interp[i, j]))
                    elif np.ma.is_masked(i_interp[i, j]):
                        self.assertEqual(0, o_interp[i, j])
                    else:
                        self.assertEqual(i_interp[i, j], o_interp[i, j])

                    # Check cdi_compress
                    if np.ma.is_masked(i_cdi_index[i, j]) and np.ma.is_masked(o_elev[i, j]):
                        self.assertTrue(np.ma.is_masked(o_cdi_index[i, j]))
                    else:
                        self.assertEqual(0, o_cdi_index[i, j])

        os.remove(o_path)

    @classmethod
    def tearDownClass(cls):
        print(f"End of {cls.__name__}.")
        os.remove(cls.path)
        os.remove(cls.path2)
        os.remove(cls.path3)


if __name__ == "__main__":
    unittest.main()
