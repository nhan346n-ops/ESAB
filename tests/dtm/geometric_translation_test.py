import os
import tempfile as tmp
import unittest

import netCDF4 as nc
import numpy as np

import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.directory_utils as directory
from pyat.dtm.experimental.geometric_translation import GeometricTranslationProcess
from tests.tools.netcdf import comparator


class TestGeometricTranslation(unittest.TestCase):

    # Cette méthode sera appelée avant chaque test.
    def setUp(self):
        print(f"Start of {self._testMethodName}.")
        self.output_file = tmp.mktemp(suffix="_merge_slope_test.dtm.nc")

    # Cette méthode sera appelée après chaque test.
    def tearDown(self):
        os.remove(self.output_file)

    # run a merge with slope tesst
    def test_mixed(self):
        self.execute(2, 10)

    def test_row(self):
        self.execute(0, 10)

    def test_decimal(self):
        self.execute(1.5, 0.5)

    def test_col(self):
        self.execute(-10, 0)

    def execute(self, row, col):

        # Parameters
        reference_file = directory.get_test_directory() + "/raw/fill_holes.dtm.nc"
        i_paths = [reference_file]
        params = {
            "i_paths": i_paths,
            "o_paths": [self.output_file],
            "rows": row,
            "columns": col,
            "overwrite": True,
        }

        # Process
        process = GeometricTranslationProcess(**params)
        process()

        # now parse output file and check values
        with nc.Dataset(self.output_file, mode="r") as out_data, nc.Dataset(reference_file, mode="r") as in_data:
            for name in in_data.variables:
                if name not in [DtmConstants.CRS_NAME, DtmConstants.LAT_NAME, DtmConstants.LON_NAME, DtmConstants.CDI]:
                    comparator.compare_variables_data(in_data, name, out_data, name)

            comparator.compare_cdi_variables(in_data, DtmConstants.CDI, out_data, DtmConstants.CDI)

            # compare LAT and check that they are separated from 2 deg
            self.assertTrue(np.all(out_data[DtmConstants.LAT_NAME][:] == in_data[DtmConstants.LAT_NAME][:] + row))
            # compare LON and check for a 10 deg offset
            self.assertTrue(np.all(out_data[DtmConstants.LON_NAME][:] == in_data[DtmConstants.LON_NAME][:] + col))


if __name__ == "__main__":
    unittest.main()
