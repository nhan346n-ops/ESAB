#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile as tmp
import unittest

import netCDF4 as nc
import numpy as np

import pyat.dtm.dtm_standard_constants as DtmConstants
from pyat.dtm.cdi.split_by_cdi_process import SplitByCdiProcess
from tests.generator.dtm_generator import DtmGenerator


class TestSplitByCdi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")
        cls.directory = tmp.mkdtemp()
        generator = DtmGenerator(cls.directory)
        cls.path = generator.create_reset_cell_file()

    def test_split_by_cdi(self):
        # Parameters
        i_paths = [self.path]
        params = {"i_paths": i_paths, "overwrite": True}

        # Process
        splitByCdi = SplitByCdiProcess(**params)
        splitByCdi()

        # Verify
        num_output_files = 0
        with nc.Dataset(self.path) as i_ds:
            for o_name in os.listdir(self.directory):
                if o_name.startswith(os.path.basename(self.path[:-3]) + "-cdi"):
                    with nc.Dataset(os.path.join(self.directory, o_name)) as o_ds:
                        o_elev = o_ds[DtmConstants.ELEVATION_NAME][:]
                        i_elev = i_ds[DtmConstants.ELEVATION_NAME][:]
                        o_cdi_index = o_ds[DtmConstants.CDI_INDEX][:]
                        i_cdi_index = i_ds[DtmConstants.CDI_INDEX][:]

                        o_cdi = o_ds[DtmConstants.CDI][:]
                        i_cdi = i_ds[DtmConstants.CDI][:]

                        self.assertEqual(len(o_cdi[o_cdi != ""]), 1)
                        cdi = o_name.split("_")[-1][:-3]
                        ind = int(np.where(i_cdi == cdi)[0])

                        for i in range(o_elev.shape[0]):
                            for j in range(o_elev.shape[1]):
                                if i_cdi_index[i, j] == ind:
                                    self.assertEqual(0, o_cdi_index[i, j])
                                    self.assertEqual(o_elev[i, j], i_elev[i, j])
                                else:
                                    self.assertTrue(np.ma.is_masked(o_cdi_index[i, j]))
                                    self.assertTrue(np.ma.is_masked(o_elev[i, j]))
                    num_output_files += 1
                    print(f"Split by cdi {cdi} for file {o_name}: OK")

            for file in os.listdir(self.directory):
                if file.startswith(os.path.basename(self.path[:-3]) + "-cdi"):
                    os.remove(os.path.join(self.directory, file))

            # check process actually worked
            self.assertGreater(num_output_files, 0)

    @classmethod
    def tearDownClass(cls):
        print(f"End of {cls.__name__}.")
        os.remove(cls.path)


if __name__ == "__main__":
    unittest.main()
