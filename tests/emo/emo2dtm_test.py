#! /usr/bin/env python3
# coding: utf-8

import math
import os
import tempfile as tmp
import unittest

import netCDF4 as nc
import numpy as np
import pandas

import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.emo.emo_constants as EmoConstants
import tests.directory_utils as dir
from pyat.emo.emo_driver import EmoFile
from pyat.emo.emo_exporter import ToDtmExporter
from tests.generator.emo_generator import EmoGenerator

dtmEmoMapping = {
    DtmConstants.DIM_LON: EmoConstants.COL_LONGITUDE,
    DtmConstants.DIM_LAT: EmoConstants.COL_LATITUDE,
    DtmConstants.CDI: EmoConstants.COL_DTM_SOURCE,
    DtmConstants.ELEVATION_NAME: EmoConstants.COL_MEAN_DEPTH,
    DtmConstants.ELEVATION_MIN: EmoConstants.COL_MAX_DEPTH,
    DtmConstants.ELEVATION_MAX: EmoConstants.COL_MIN_DEPTH,
    DtmConstants.VALUE_COUNT: EmoConstants.COL_NB_OF_SOUNDS,
    DtmConstants.STDEV: EmoConstants.COL_STDEV,
    DtmConstants.CDI_INDEX: EmoConstants.COL_CDIID,
    DtmConstants.ELEVATION_SMOOTHED_NAME: EmoConstants.COL_SMOOTHED_DEPTH,
    DtmConstants.INTERPOLATION_FLAG: EmoConstants.COL_INTERPOLATED_CELL,
}


class TestEmo2Dtm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")
        generator = EmoGenerator(dir.get_output_directory())
        cls.n = 200
        cls.n_files = 5
        cls.paths = []
        for i in range(cls.n_files):
            cls.paths.append(generator.create_1("test_" + str(i), n=cls.n))

    def test_emo_export(self):
        # Parameters
        path_emo = self.paths[4]
        i_paths = [path_emo]
        o_paths = [tmp.mktemp(suffix=".dtm.nc")]
        params = {"i_paths": i_paths, "o_paths": o_paths}

        # Process
        exporter = ToDtmExporter(**params)
        exporter()

        # Verify
        emo_data = pandas.read_csv(
            path_emo, names=EmoFile.ColumnNames, delimiter=";", header=None, usecols=EmoFile.ColumnNames
        )
        with nc.Dataset(o_paths[0], mode="r") as nc_data:
            for name, value in nc_data.variables.items():
                if not name in ["lon", "lat", "crs", "cdi_reference"]:
                    for row in range(0, value.shape[0], round(self.n / 20)):
                        for col in range(0, value.shape[1], round(self.n / 20)):
                            if value.shape[0] == 1:
                                element = col
                            else:
                                element = row * value.shape[1] + col

                            if name in [
                                DtmConstants.ELEVATION_MIN,
                                DtmConstants.ELEVATION_MAX,
                                DtmConstants.ELEVATION_NAME,
                                DtmConstants.ELEVATION_SMOOTHED_NAME,
                            ]:
                                factor = -1
                            else:
                                factor = 1

                            if not np.isnan(emo_data[dtmEmoMapping[name]].values[element]):
                                # Min and Max should be reversed du to the -1 sign
                                self.assertLessEqual(
                                    nc_data[name][row, col] - emo_data[dtmEmoMapping[name]].values[element] * factor,
                                    1e-3,
                                )
        os.remove(o_paths[0])

    def test_emo_export_multiple(self):
        # Parameters
        i_paths = []
        o_paths = []
        for i in range(3):
            path_emo = self.paths[i]
            i_paths.append(path_emo)
            o_paths.append(tmp.mktemp(suffix=".dtm.nc"))
        params = {"i_paths": i_paths, "o_paths": o_paths}

        # Process
        exporter = ToDtmExporter(**params)
        exporter()

        # Verify
        for path, path_nc in zip(i_paths, o_paths):
            emo_data = pandas.read_csv(
                path, names=EmoFile.ColumnNames, delimiter=";", header=None, usecols=EmoFile.ColumnNames
            )
            with nc.Dataset(path_nc, mode="r") as nc_data:
                for name, value in nc_data.variables.items():
                    if not name in ["lon", "lat", "crs", "cdi_reference"]:
                        for row in range(0, value.shape[0], round(self.n / 20)):
                            for col in range(0, value.shape[1], round(self.n / 20)):
                                if value.shape[0] == 1:
                                    element = col
                                else:
                                    element = row * value.shape[1] + col

                                if name in [
                                    DtmConstants.ELEVATION_MIN,
                                    DtmConstants.ELEVATION_MAX,
                                    DtmConstants.ELEVATION_NAME,
                                    DtmConstants.ELEVATION_SMOOTHED_NAME,
                                ]:
                                    factor = -1
                                else:
                                    factor = 1

                                if not math.isnan(emo_data[dtmEmoMapping[name]].values[element]):
                                    self.assertLessEqual(
                                        nc_data[name][row, col]
                                        - emo_data[dtmEmoMapping[name]].values[element] * factor,
                                        1e-3,
                                    )
            os.remove(path_nc)

    @classmethod
    def tearDownClass(cls):
        for path in cls.paths:
            os.remove(path)
        print(f"End of {cls.__name__}.")


if __name__ == "__main__":
    unittest.main()
