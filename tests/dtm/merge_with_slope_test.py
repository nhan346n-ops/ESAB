import os
import tempfile as tmp
import unittest

import netCDF4 as nc
import numpy as np

import pyat.dtm.dtm_standard_constants as cst
import tests.directory_utils as directory
from pyat.dtm.merge.merge_with_slope import SlopeMerge
from pyat.emo.emo_exporter import ToDtmExporter


class TestMergeWithSlope(unittest.TestCase):
    def emo_to_dtm(self, in_emo: str, out_dtm: str):
        # Process
        exporter = ToDtmExporter([in_emo], [out_dtm], overwrite=True)
        exporter()
        print(f"Generated test file {out_dtm}")

    # Cette méthode sera appelée avant chaque test.
    def setUp(self):
        print(f"Start of {self._testMethodName}.")
        input_pathA = directory.get_test_directory() + "/raw/slope_testA.emo"
        input_pathB = directory.get_test_directory() + "/raw/slope_testB.emo"
        self.reference_file = tmp.mktemp(suffix="_merge_slope_testA.dtm.nc")
        self.second_file = tmp.mktemp(suffix="_merge_slope_testB.dtm.nc")
        self.emo_to_dtm(input_pathA, self.reference_file)
        self.emo_to_dtm(input_pathB, self.second_file)
        self.output_file = tmp.mktemp(suffix="_merge_slope_test.dtm.nc")

    # Cette méthode sera appelée après chaque test.
    def tearDown(self):
        os.remove(self.reference_file)
        os.remove(self.second_file)
        os.remove(self.output_file)

    def test_merge_with_slope(self):
        # run a merge with slope tesst
        # Parameters
        i_paths = [self.reference_file]
        second_file = self.second_file
        coord = {"north": 0.0046875, "south": -0.0005208, "west": -0.0005208, "east": 0.0046875}
        params = {
            "o_path": self.output_file,
            "min_slope": 6,
            "max_slope": 8,
            "overwrite": True,
        }

        # coord parameter is mandatory
        try:
            process = SlopeMerge(i_paths, second_file, coord={}, **params)
            process()
            self.fail("The mandatory presence of the parameter 'coord' is not checked")
        except ValueError as e:
            # normal case
            self.assertTrue("Invalid parameter" in str(e))

        # Process
        process = SlopeMerge(i_paths, second_file, coord, **params)
        process()

        # now parse output file and check values
        with nc.Dataset(self.output_file, mode="r") as nc_data:
            elevation = nc_data[cst.ELEVATION_NAME][:]
            cdi_index = nc_data[cst.CDI_INDEX][:]
            self.assertEqual(nc_data[cst.CDI][0], "SDN:CDI:LOCAL:A")
            self.assertEqual(nc_data[cst.CDI][1], "SDN:CDI:LOCAL:B")
            # the upper part is supposed to come from the second file
            self.assertTrue(elevation[0, 2] == -2)
            self.assertTrue(np.all(cdi_index[0] == 1))
            # check for a special mixed point
            local_slope = 7.395142  # we know the slope
            f = (local_slope - 6) / 2
            local_depth = -(10 * f + (1 - f) * 5)
            if f > 0.5:
                self.assertTrue(np.all(cdi_index[1] == 1))
            else:
                self.assertTrue(np.all(cdi_index[1] == 0))
            self.assertAlmostEqual(elevation[1, 0], local_depth, delta=10e-5)
            # the lower part come from the first
            self.assertTrue(np.all(elevation[2] == -20))
            self.assertTrue(np.all(elevation[3] == -10))
            self.assertTrue(np.all(cdi_index[2] == 0))
            self.assertTrue(np.all(cdi_index[3] == 0))
            self.assertTrue(np.all(cdi_index[4] == 0))


if __name__ == "__main__":
    unittest.main()
