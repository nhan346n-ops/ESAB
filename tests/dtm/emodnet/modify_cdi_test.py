#! /usr/bin/env python3
# coding: utf-8

import os
import unittest

import netCDF4 as nc
import numpy as np

import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.directory_utils as dir_util
from pyat.dtm.cdi.modify_cdi_process import ModifyCdiProcess
from tests.generator.dtm_generator import DtmGenerator, geoBox1, geoBox1_bis, geoBox1_bis_bis


class TestModifyCdi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")
        cls.directory = dir_util.get_test_directory()
        generator = DtmGenerator(cls.directory)
        cls.path = generator.create_long_lat(
            geoBox=geoBox1, zones=np.array([[1, 3, 1, 3], [10, 15, 10, 12]]), values=[10, 20], opt="zone"
        )
        cls.path2 = generator.create_long_lat(
            geoBox=geoBox1_bis, zones=np.array([[1, 3, 1, 3], [10, 15, 10, 12]]), values=[10, 20], opt="zone"
        )
        cls.path3 = generator.create_long_lat(
            geoBox=geoBox1_bis_bis,
            zones=np.array([[1, 3, 1, 3], [10, 15, 10, 12]]),
            values=[10, 20],
            opt="zone",
        )

    def test_modify_cdi_1(self):
        # Parameters
        path = self.path
        i_paths = [path]
        cdis = [{"old": "10", "new": "test1"}, {"old": "20", "new": "test2"}]
        params = {"i_paths": i_paths, "cdis": cdis}
        # create fake CDI references
        with nc.Dataset(self.path, "a") as i_dataset:
            cdi_index = i_dataset[DtmConstants.CDI_INDEX]
            cdi_index[0, 0] = 0
            cdi_index[1, 1] = 1
        # Process
        modifyCdi = ModifyCdiProcess(**params)
        modifyCdi()

        # Verify
        with nc.Dataset(self.path) as o_dataset:
            o_cdi_ref = o_dataset[DtmConstants.CDI]

            self.assertEqual(o_cdi_ref[0], "test1")
            self.assertEqual(o_cdi_ref[1], "test2")

            # Verify history
            self.assertTrue(", process with Python ModifyCdiProcess" in o_dataset.history)

    def test_modify_cdi_2(self):
        # Parameters
        path = self.path
        i_paths = [path]
        cdis = [{"old": "20", "new": "test2"}]
        params = {"i_paths": i_paths, "cdis": cdis}
        # create fake CDI references
        with nc.Dataset(self.path, "a") as i_dataset:
            cdi_index = i_dataset[DtmConstants.CDI_INDEX]
            cdi_index[0, 0] = 0
            cdi_index[1, 1] = 1
        # Process
        modifyCdi = ModifyCdiProcess(**params)
        modifyCdi()

        # Verify
        with nc.Dataset(self.path) as o_dataset:
            o_cdi_ref = o_dataset[DtmConstants.CDI]

            self.assertEqual(o_cdi_ref[1], "test2")

    # def test_modify_cdi_3(self):
    #     # Parameters
    #     path = self.path
    #     path2 = self.path2
    #     path3 = self.path3
    #     i_paths = [path, path2, path3]
    #     cdis = [{"old": "20", "new": "test2"}, {"old": "10", "new": "test0"}, {"old": "test1", "new": "test0"}]
    #     params = {"i_paths": i_paths, "cdis": cdis}
    #     # create fake CDI references
    #     with nc.Dataset(self.path, "a") as i_dataset:
    #         cdi_index = i_dataset[DtmConstants.CDI_INDEX]
    #         cdi_index[0, 0] = 0
    #         cdi_index[1, 1] = 1
    #     # Process
    #     ModifyCdi = ModifyCdiProcess(**params)
    #     ModifyCdi.run()
    #
    #     # Verify
    #     with nc.Dataset(self.path) as ds1, nc.Dataset(self.path2) as ds2, nc.Dataset(self.path3) as ds3:
    #         o_cdi_ref1 = ds1[DtmConstants.CDI][:]
    #         o_cdi_ref2 = ds2[DtmConstants.CDI][:]
    #         o_cdi_ref3 = ds3[DtmConstants.CDI][:]
    #
    #         self.assertEqual(o_cdi_ref1[1], "test2")
    #         self.assertEqual(o_cdi_ref2[0], "test0")
    #         self.assertEqual(o_cdi_ref3[1], "test2")

    @classmethod
    def tearDownClass(cls):
        print(f"End of {cls.__name__}.")
        os.remove(cls.path)
        os.remove(cls.path2)
        os.remove(cls.path3)


if __name__ == "__main__":
    unittest.main()
