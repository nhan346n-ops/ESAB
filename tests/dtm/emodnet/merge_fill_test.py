#! /usr/bin/env python3
# coding: utf-8

import math
import os
import unittest

import netCDF4 as nc
import numpy as np

import pyat.common.geo_file as gf
import pyat.dtm.dtm_standard_constants as DtmConstants
import tests.directory_utils as dir_util
from pyat.dtm.merge.merge_fill import MergeFillProcess
from tests.generator.dtm_generator import (
    DtmGenerator,
    geoBox1,
    geoBox2,
    geoBox3,
    geoBox4,
    geoBox5,
    geoBox6,
    geoBox7,
)
from tests.generator.kml_generator import create_kml


class TestMergeFill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")

    def setUp(self):
        self.directory = dir_util.get_test_directory()
        self.generator = DtmGenerator(self.directory)
        # Output File Path
        self.o_path = os.path.join(self.directory, "merged_fill" + DtmConstants.EXTENSION_NC)

    def test_merge_fill_1(self):
        # Parameters
        self.paths = [self.generator.create_1(value=10), self.generator.create_1(value=20)]
        params = {"i_paths": self.paths, "overwrite": True, "allow_undefined_cdi": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify merge
        with nc.Dataset(self.o_path, "r") as o_file:
            layer_elevation = o_file[DtmConstants.ELEVATION_NAME]
            for r in range(layer_elevation.shape[0]):
                for c in range(layer_elevation.shape[1]):
                    self.assertEqual(10.0, layer_elevation[r, c])
            # Verify history
            print(">>>>>>>>>>>>>>>", o_file.history)
            self.assertTrue(o_file.history.startswith("Process with Python merged_fill from "))

    def test_merge_fill_1_mask(self):
        # Parameters
        north = 47.005
        south = 47.001
        west = -4
        east = -3.99
        coord = [[west, south], [east, south], [east, north], [west, north]]
        kml_path = create_kml(self.directory, {"z": coord})
        kmls = [kml_path]
        self.paths = [self.generator.create_1(value=10), self.generator.create_1(value=20)]
        params = {"i_paths": self.paths, "mask": kmls, "allow_undefined_cdi": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify merge. Expects reference DTM as result in merged file
        with nc.Dataset(self.o_path, "r") as o_file:
            layer_elevation = o_file[DtmConstants.ELEVATION_NAME][:]
            for r in range(layer_elevation.shape[0]):
                for c in range(layer_elevation.shape[1]):
                    self.assertEqual(10.0, layer_elevation[r, c])

        os.remove(kml_path)

    def test_merge_fill_2(self):
        # Parameters
        self.paths = [self.generator.create_1(value=10), self.generator.create_1(value=20)]
        params = {"i_paths": self.paths[::-1], "overwrite": True, "allow_undefined_cdi": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        with nc.Dataset(self.o_path, "r") as o_file:
            # Verify merge
            layer_elevation = o_file[DtmConstants.ELEVATION_NAME]
            for r in range(layer_elevation.shape[0]):
                for c in range(layer_elevation.shape[0]):
                    self.assertEqual(20.0, layer_elevation[r, c])

    def test_merge_fill_3(self):
        # Parameters
        self.paths = [
            self.generator.create_1(value=10, missing_value=(3, 10)),
            self.generator.create_1(value=20),
        ]
        params = {"i_paths": self.paths, "overwrite": True, "allow_undefined_cdi": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify merge
        with nc.Dataset(self.o_path, "r") as o_file:
            layer_elevation = o_file[DtmConstants.ELEVATION_NAME]
            for r in range(layer_elevation.shape[0]):
                for c in range(layer_elevation.shape[0]):
                    if r == 3 and c == 10:
                        self.assertEqual(20.0, layer_elevation[r, c])
                    else:
                        self.assertEqual(10.0, layer_elevation[r, c])

    def test_merge_fill_wgs_84(self):
        # Parameters
        self.paths = [
            self.generator.create_pattern(
                value=20, spatial_reference=gf.SR_WGS_84, pair_impair=1, line_col=1, number=2, allValue=False
            ),
            self.generator.create_pattern(value=10, spatial_reference=gf.SR_WGS_84),
        ]
        params = {"i_paths": self.paths, "overwrite": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify merge
        with nc.Dataset(self.o_path, "r") as o_file:
            self.assertEqual(o_file[DtmConstants.CRS_NAME].__dict__["grid_mapping_name"], "latitude_longitude")
            self.__check_merged_fill()

    def test_merge_fill_mercator(self):
        # Parameters
        self.paths = [
            self.generator.create_pattern(
                value=20, spatial_reference=gf.SR_PSEUDO_MERCATOR, pair_impair=1, line_col=1, number=2, allValue=False
            ),
            self.generator.create_pattern(value=10, spatial_reference=gf.SR_PSEUDO_MERCATOR),
        ]
        params = {"i_paths": self.paths, "overwrite": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify merge
        with nc.Dataset(self.o_path, "r") as o_file:
            self.assertEqual(o_file[DtmConstants.CRS_NAME].__dict__["grid_mapping_name"], "mercator")
            self.__check_merged_fill()

    def __check_merged_fill(self):
        with nc.Dataset(self.o_path, "r") as o_file:
            layer_elevation = o_file[DtmConstants.ELEVATION_NAME]
            layer_elevation_min = o_file[DtmConstants.ELEVATION_MIN]
            layer_elevation_max = o_file[DtmConstants.ELEVATION_MAX]
            layer_cdi_indexLayer = o_file[DtmConstants.CDI_INDEX]
            layer_stdev = o_file[DtmConstants.STDEV]
            layer_value_count = o_file[DtmConstants.VALUE_COUNT]
            layer_cdiLayer = o_file[DtmConstants.CDI]

            cdi_10 = np.where(layer_cdiLayer[:] == "10")
            cdi_20 = np.where(layer_cdiLayer[:] == "20")

            self.assertTrue(cdi_10)
            self.assertTrue(cdi_20)
            self.assertNotEqual(cdi_10, cdi_20)

            for r in range(layer_elevation.shape[0]):
                for c in range(layer_elevation.shape[1]):
                    if not c % 2:
                        self.assertEqual(10, layer_value_count[r, c])
                        self.assertEqual(10, layer_elevation[r, c])
                        self.assertEqual(11, layer_elevation_max[r, c])
                        self.assertEqual(9, layer_elevation_min[r, c])
                        self.assertEqual(1, layer_stdev[r, c])
                        self.assertEqual(cdi_10[0], layer_cdi_indexLayer[r, c])
                    else:
                        self.assertEqual(20.0, layer_value_count[r, c])
                        self.assertEqual(20, layer_elevation[r, c])
                        self.assertEqual(21, layer_elevation_max[r, c])
                        self.assertEqual(19, layer_elevation_min[r, c])
                        self.assertEqual(2, layer_stdev[r, c])
                        self.assertEqual(cdi_20[0], layer_cdi_indexLayer[r, c])

    def test_merge_fill_copy_dims(self):
        # Parameters
        self.paths = [self.generator.create_1(value=10), self.generator.create_1(value=20)]
        params = {"i_paths": self.paths, "overwrite": True, "allow_undefined_cdi": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify dimensions
        with nc.Dataset(self.paths[0], "r") as ref_file, nc.Dataset(self.o_path, "r") as o_file:
            for name, dimension in o_file.dimensions.items():
                if name == DtmConstants.DIM_CDI:
                    self.assertEqual(dimension.size, 2, msg=f"Error while checking dimension {name}")
                else:
                    self.assertEqual(
                        dimension.size, ref_file.dimensions[name].size, msg=f"Error while checking dimension {name}"
                    )

    def test_merge_fill_copy_global_atts(self):
        # Parameters
        self.paths = [self.generator.create_1(value=10), self.generator.create_1(value=20)]
        params = {"i_paths": self.paths, "allow_undefined_cdi": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify global attributes
        with nc.Dataset(self.paths[0], "r") as ref_file, nc.Dataset(self.o_path, "r") as o_file:
            for name in list(o_file.__dict__):
                if not name in ["history", "dtm_convention_version"]:
                    self.assertEqual(ref_file.getncattr(name), o_file.getncattr(name))

    def test_merge_fill_copy_vars_atts(self):
        # Parameters
        self.paths = [self.generator.create_1(value=10), self.generator.create_1(value=20)]
        params = {"i_paths": self.paths, "allow_undefined_cdi": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify variables attributes.
        with nc.Dataset(self.paths[0], "r") as ref_file, nc.Dataset(self.o_path, "r") as o_file:
            for name, attrs in o_file.variables.items():
                for attr, value in attrs.__dict__.items():
                    if not attr in ["_FillValue", "valid_range", "flag_values"]:
                        self.assertEqual(value, ref_file[name].getncattr(attr))

    def test_merge_fill_copy_vars_atts_2(self):
        # Parameters
        self.paths = [
            self.generator.create_pattern(value=20, pair_impair=1, line_col=1, number=2, allValue=False),
            self.generator.create_pattern(value=10),
        ]
        params = {"i_paths": self.paths}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify variables attributes
        with nc.Dataset(self.paths[0], "r") as ref_file, nc.Dataset(self.o_path, "r") as o_file:
            for name, attrs in o_file.variables.items():
                for attr, value in attrs.__dict__.items():
                    if not attr in ["_FillValue", "valid_range", "flag_values"]:
                        self.assertEqual(value, ref_file[name].getncattr(attr))

    def test_merge_fill_4(self):
        # Parameters
        self.paths = [
            self.generator.create_long_lat(geoBox=geoBox1, zones=np.array([[0, 1, 0, 1]]), values=[100], opt="zone"),
            self.generator.create_long_lat(geoBox=geoBox1, zones=np.array([[2, 3, 2, 3]]), values=[30], opt="zone"),
            self.generator.create_long_lat(geoBox=geoBox1, zones=np.array([[4, 5, 4, 5]]), values=[2], opt="zone"),
        ]
        params = {"i_paths": self.paths, "overwrite": True, "allow_undefined_cdi": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify merge
        with nc.Dataset(self.o_path, "r") as o_file:
            elevation = o_file[DtmConstants.ELEVATION_NAME]
            self.assertEqual(100, elevation[0, 0])
            self.assertEqual(30, elevation[2, 2])
            self.assertEqual(2, elevation[4, 4])

    def test_merge_fill_geobox_2(self):
        # Parameters
        self.paths = [
            self.generator.create_long_lat(
                geoBox=geoBox1, zones=np.array([[3, 5, 1, 3], [8, 15, 0, 7]]), values=[20, 10], opt="zone"
            ),
            self.generator.create_long_lat(
                geoBox=geoBox1, zones=np.array([[1, 3, 1, 3], [10, 15, 10, 12]]), values=[10, 20], opt="zone"
            ),
            self.generator.create_long_lat(geoBox=geoBox2, zones=np.array([[8, 11, 8, 11]]), values=[30], opt="zone"),
        ]
        params = {"i_paths": self.paths, "allow_undefined_cdi": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify merge
        with nc.Dataset(self.o_path, "r") as o_file:
            layer_elevation = o_file[DtmConstants.ELEVATION_NAME][:]

            for r in range(layer_elevation.shape[0]):
                for c in range(layer_elevation.shape[1]):
                    # Check zone 3-5 / 1-3
                    if r in (3, 4, 5) and c in (1, 2, 3):
                        self.assertEqual(20, layer_elevation[r, c])

                    # Check zone 8-15 / 0-7
                    elif 8 <= r <= 15 and 0 <= c <= 7:
                        self.assertEqual(10, layer_elevation[r, c])

                    # Check zone 1-2 / 1-3
                    elif r in (1, 2) and c in (1, 2, 3):
                        self.assertEqual(10, layer_elevation[r, c])

                    # Check zone 10-15 / 10-12
                    elif 10 <= r <= 15 and 10 <= c <= 12:
                        self.assertEqual(20, layer_elevation[r, c])

                    # Check zone 4-7 / 4-7
                    elif 4 <= r <= 7 and 4 <= c <= 7:
                        self.assertEqual(30, layer_elevation[r, c])

    def test_merge_fill_geobox_3(self):
        # Parameters
        self.paths = [
            self.generator.create_long_lat(
                geoBox=geoBox1, zones=np.array([[3, 5, 1, 3], [8, 15, 0, 7]]), values=[20, 10], opt="zone"
            ),
            self.generator.create_long_lat(geoBox=geoBox3, zones=np.array([[0, 9, 0, 9]]), values=[30], opt="steps"),
        ]
        params = {"i_paths": self.paths, "allow_undefined_cdi": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify merge
        with nc.Dataset(self.o_path, "r") as o_file:
            layer_elevation = o_file[DtmConstants.ELEVATION_NAME][:]

            for r in range(layer_elevation.shape[0]):
                for c in range(layer_elevation.shape[1]):
                    # Check zone 3-5 / 1-3
                    if r in (3, 4, 5) and c in (1, 2, 3):
                        self.assertEqual(20, layer_elevation[r, c])

                    # Check zone 8-15 / 0-7
                    elif 8 <= r <= 15 and 0 <= c <= 7:
                        self.assertEqual(10, layer_elevation[r, c])

                    # Check zone 4-7 / 4-7
                    elif 4 <= r <= 7 and 4 <= c <= 7:
                        self.assertEqual(30, layer_elevation[r, c])

                    # Check steps 4-9 / 8-13
                    elif 4 <= r <= 9 and 8 <= c <= 13:
                        if c < 13 - (r - 4):
                            self.assertEqual(30, layer_elevation[r, c])

    def test_merge_fill_geobox_4(self):
        # Parameters
        self.paths = [
            self.generator.create_long_lat(
                geoBox=geoBox4, zones=np.array([[90, 700, 90, 700]]), values=[10], opt="zone"
            ),
            self.generator.create_long_lat(
                geoBox=geoBox5, zones=np.array([[90, 959, 90, 959]]), values=[20], opt="steps"
            ),
        ]
        params = {"i_paths": self.paths, "overwrite": True, "allow_undefined_cdi": True}

        # Process
        geobox = {
            "north": 47.00052083333333,
            "south": 46.00052083333333,
            "west": -4.999479166666666,
            "east": -3.999479166666666,
        }
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify merge
        with nc.Dataset(self.o_path, "r") as o_file:
            layer_elevation = o_file[DtmConstants.ELEVATION_NAME]

            half_ind = int(layer_elevation.shape[0] / 2)

            for r in range(0, layer_elevation.shape[0], 10):
                for c in range(0, layer_elevation.shape[1], 10):
                    # Check zone 90-700 / 90-700
                    if 700 >= r >= 90 and 700 >= c >= 90:
                        self.assertEqual(10, layer_elevation[r, c])

                    # Check step 0-89 / 480-569
                    elif 89 >= r >= 0 and half_ind + 89 >= c >= half_ind:
                        if c < half_ind - 89 - (r - 0):
                            self.assertEqual(20, layer_elevation[r, c])

                    # Check step 480-569 / 0-89
                    elif half_ind + 89 >= r >= half_ind and 89 >= c >= 0:
                        if c < 89 - (r - half_ind):
                            self.assertEqual(20, layer_elevation[r, c])

                    # Check zone 0-89 / 0-480
                    elif 89 >= r >= 0 and half_ind >= c >= 0:
                        self.assertEqual(20, layer_elevation[r, c])

                    # Check zone 0-480 / 0-89
                    elif half_ind >= r >= 0 and 89 >= c >= 0:
                        self.assertEqual(20, layer_elevation[r, c])

    def test_merge_fill_geobox_5(self):
        # Parameters
        self.paths = [
            self.generator.create_long_lat(
                geoBox=geoBox4, zones=np.array([[90, 700, 90, 700]]), values=[10], opt="zone"
            ),
            self.generator.create_long_lat(
                geoBox=geoBox6, zones=np.array([[90, 959, 90, 959]]), values=[40], opt="steps"
            ),
        ]
        params = {"i_paths": self.paths, "allow_undefined_cdi": True}

        # Process
        geobox = {
            "north": 47.00052083333333,
            "south": 46.00052083333333,
            "west": -4.999479166666666,
            "east": -3.999479166666666,
        }
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify merge
        with nc.Dataset(self.o_path, "r") as o_file:
            layer_elevation = o_file[DtmConstants.ELEVATION_NAME]

            for r in range(0, layer_elevation.shape[0], 10):
                for c in range(0, layer_elevation.shape[1], 10):
                    # Check zone 90-700 / 90-700
                    if 700 >= r >= 90 and 700 >= c >= 90:
                        self.assertEqual(10, layer_elevation[r, c])

                    # Check steps 0-89 / 518-607
                    elif 89 >= r >= 0 and 607 >= c >= 518:
                        if c < 607 - (r - 0):
                            self.assertEqual(40, layer_elevation[r, c])

                    # Check steps 518-607 / 0-89
                    elif 607 >= r >= 518 and 89 >= c >= 0:
                        if c < 89 - (r - 518):
                            self.assertEqual(40, layer_elevation[r, c])

                    # Check zone 0-89 / 0-518
                    elif 89 >= r >= 0 and 518 >= c >= 0:
                        self.assertEqual(40, layer_elevation[r, c])

                    # Check zone 0-518 / 0-89
                    elif 518 >= r >= 0 and 89 >= c >= 0:
                        self.assertEqual(40, layer_elevation[r, c])

    def test_merge_fill_geobox_6(self):
        # Parameters
        self.paths = [
            self.generator.create_long_lat(
                geoBox=geoBox4, zones=np.array([[90, 700, 90, 700]]), values=[10], opt="zone"
            ),
            self.generator.create_long_lat(
                geoBox=geoBox7, zones=np.array([[0, 479, 240, 719]]), values=[20], opt="steps"
            ),
        ]
        params = {"i_paths": self.paths, "allow_undefined_cdi": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify merge
        with nc.Dataset(self.o_path, "r") as o_file:
            layer_elevation = o_file[DtmConstants.ELEVATION_NAME]

            half_ind = int(layer_elevation.shape[0] / 2)
            offset = int(math.ceil(((geoBox7[0] - geoBox4[0]) / (geoBox4[1] - geoBox4[0])) * layer_elevation.shape[1]))

            for r in range(0, layer_elevation.shape[0], 10):
                for c in range(0, layer_elevation.shape[1], 10):
                    # Check zone 90-700 / 90-700
                    if 700 >= r >= 90 and 700 >= c >= 90:
                        self.assertEqual(10, layer_elevation[r, c])

                    # Check steps 668-738 / 19-89
                    elif 960 - offset - 1 >= r >= 960 - offset - 1 - 89 + 240 - offset and 89 >= c >= 240 - offset:
                        if c < 89 - (r - (960 - offset - 1 - 89 + 240 - offset)):
                            self.assertEqual(20, layer_elevation[r, c])

                    # Check zone 259-668 / 19-89
                    elif 960 - offset - 1 - 89 + 240 - offset >= r >= half_ind - offset and 89 >= c >= 240 - offset:
                        self.assertEqual(20, layer_elevation[r, c])

    def test_merge_fill_cdi(self):
        # Parameters
        self.paths = [
            self.generator.create_pattern_for_cdi_test(value=10, mode=1),
            self.generator.create_pattern_for_cdi_test(value=9, mode=0),
        ]
        params = {"i_paths": self.paths, "overwrite": True}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify merge
        with nc.Dataset(self.o_path, "r") as o_file:
            o_cdi = o_file[DtmConstants.CDI_INDEX][:]

            for row in range(o_cdi.shape[0]):
                for col in range(o_cdi.shape[1]):
                    if row % 2 == 0:
                        self.assertEqual(0, o_cdi[row, col])
                    else:
                        self.assertEqual(1, o_cdi[row, col])

    def test_merge_fill_cdi_2(self):
        # Parameters
        self.paths = [
            self.generator.create_pattern_for_cdi_test(value=8, mode=1, value_count=20, cdi="test"),
            self.generator.create_pattern_for_cdi_test(value=7, mode=0, value_count=10, cdi="test2"),
        ]
        params = {"i_paths": self.paths}

        # Process
        geobox = {"north": 47.0171875, "south": 47.00052083333333, "west": -3.9994791666666667, "east": -3.9828125}
        process = MergeFillProcess(coord=geobox, **params)
        process()

        # Verify merge
        with nc.Dataset(self.o_path, "r") as o_file:
            o_cdi = o_file[DtmConstants.CDI_INDEX][:]

            for row in range(o_cdi.shape[0]):
                for col in range(o_cdi.shape[1]):
                    if row % 2 == 0:
                        self.assertEqual(0, o_cdi[row, col])
                    else:
                        self.assertEqual(1, o_cdi[row, col])

    def tearDown(self):
        if os.path.exists(self.o_path):
            os.remove(self.o_path)

        for path in self.paths:
            if os.path.exists(path):
                os.remove(path)

    @classmethod
    def tearDownClass(cls):
        print(f"End of {cls.__name__}.")


if __name__ == "__main__":
    unittest.main()
