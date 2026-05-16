#! /usr/bin/env python3
# coding: utf-8

import csv
import tempfile as tmp
import unittest

import netCDF4 as nc

import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.directory_utils as dir_util
from pyat.dtm.export.dtm_to_emo import Dtm2Emo


class TestDtm2Emo(unittest.TestCase):
    def test_dtm2emo_unique(self):
        input_path = dir_util.get_test_directory() + "/raw/fill_holes.dtm.nc"

        # Parameters
        output_path = tmp.mktemp(suffix=".emo")

        params = {"i_paths": [input_path], "o_paths": [output_path]}

        # Process
        process = Dtm2Emo(**params)
        process()

        # Verify
        with open(output_path, encoding="utf8") as csv_file:
            with nc.Dataset(input_path, mode="r") as nc_data:

                csv_reader = csv.reader(csv_file, delimiter=";")
                line = 0
                for row in csv_reader:
                    assert abs(float(row[0]) - (nc_data[DtmConstants.LON_NAME][line])) < 10e-7
                    assert abs(float(row[1]) - (nc_data[DtmConstants.LAT_NAME][line])) < 10e-7
                    # sign  convention are reversed so min max too, and depth values
                    assert row[3] == f"{-nc_data[DtmConstants.ELEVATION_MIN][0, 0]:.2f}"
                    assert row[2] == f"{-nc_data[DtmConstants.ELEVATION_MAX][0, 0]:.2f}"
                    assert row[4] == f"{-nc_data[DtmConstants.ELEVATION_NAME][0, 0]:.2f}"
                    assert row[5] == f"{nc_data[DtmConstants.STDEV][0, 0]:.2f}"
                    assert row[6] == str(nc_data[DtmConstants.VALUE_COUNT][0, 0])
                    # 0 values for interpolation flag are replaced by empty string ""
                    assert row[7] == ""
                    assert row[8] == f"{-nc_data[DtmConstants.ELEVATION_SMOOTHED_NAME][0, 0]:.2f}"
                    assert row[10] == "486_1"
                    assert row[11] == ""
                    line += 1
                    break  # only check for first line


if __name__ == "__main__":
    unittest.main()
