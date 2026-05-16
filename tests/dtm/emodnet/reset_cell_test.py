#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile
import unittest

import netCDF4 as nc
import numpy as np

import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.transform.reset_cell as const
import tests.directory_utils as dir_util
from pyat.dtm.transform.reset_cell import ResetCellProcess
from tests.generator.dtm_generator import DtmGenerator
from tests.generator.kml_generator import create_kml


class TestResetCell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")
        cls.directory = dir_util.get_test_directory()
        generator = DtmGenerator(cls.directory)
        cls.path = generator.create_reset_cell_file()
        cls.path_2 = generator.create_1(10)

        cls.VAL_EQ = 1010
        cls.VAL_MIN = 510
        cls.VAL_MAX = 1510
        cls.VAL_MISSING = np.nan

    def test_reset_cell_no_filter(self):
        # Parameters
        i_paths = [self.path]
        params = {"i_paths": i_paths}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # Verify
        o_path = self.path[:-3] + "-zeroed" + DtmConstants.EXTENSION_NC
        try:
            with nc.Dataset(o_path) as o_dataset:
                for layer in DtmConstants.LAYERS:
                    data = o_dataset[layer][:]
                    for row in range(data.shape[0]):
                        for col in range(data.shape[1]):
                            self.assertTrue(np.ma.is_masked(data[row, col]))
        finally:
            os.remove(o_path)

    def test_reset_cell_geo_zone(self):
        # Parameters
        i_paths = [self.path]
        suffix = "-zeroed_single_geo_zone"
        o_path = self.path_2[:-3] + suffix + DtmConstants.EXTENSION_NC
        lat_max = 46.5
        lat_min = 46
        lon_max = -4.5
        lon_min = -5
        coord = [[lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max]]
        kml_path = create_kml(self.directory, {"z": coord})
        params = {"i_paths": i_paths, "mask": [kml_path], "o_paths": [o_path]}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # Verify
        try:
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                for layer in DtmConstants.LAYERS:
                    data = o_dataset[layer][:]
                    i_data = i_dataset[layer][:]
                    lat = o_dataset[DtmConstants.DIM_LAT][:]
                    lon = o_dataset[DtmConstants.DIM_LON][:]

                    for row in range(data.shape[0]):
                        for col in range(data.shape[1]):
                            cell_lat = lat[row]
                            cell_lon = lon[col]
                            if lon_min < cell_lon < lon_max and lat_min < cell_lat < lat_max:
                                self.assertTrue(
                                    np.ma.is_masked(data[row, col]),
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                            elif np.ma.is_masked(data[row, col]):
                                self.assertTrue(
                                    np.ma.is_masked(i_data[row, col]),
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                            else:
                                self.assertEqual(
                                    data[row, col],
                                    i_data[row, col],
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
        finally:
            os.remove(o_path)
            os.remove(kml_path)

    def test_reset_cell_geo_zone_kml_multiple(self):
        # Parameters
        i_paths = [self.path_2]
        o_path = self.path_2[:-3] + "-zero-multiple-kml" + DtmConstants.EXTENSION_NC
        coord_1 = [[-4.001, 47.017], [-3.992, 47.017], [-3.992, 47.010], [-4.001, 47.010]]
        coord_2 = [[-3.989, 47.006], [-3.982, 47.006], [-3.982, 46.999], [-3.989, 46.999]]
        kml_1 = create_kml(self.directory, {"A": coord_1})
        kml_2 = create_kml(self.directory, {"B": coord_2})
        kmls = [kml_1, kml_2]
        params = {"i_paths": i_paths, "o_paths": [o_path], "mask": kmls}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()
        try:
            # Verify
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path_2) as i_dataset:
                for layer in DtmConstants.LAYERS:
                    data = o_dataset[layer][:]
                    i_data = i_dataset[layer][:]
                    lat = o_dataset[DtmConstants.DIM_LAT][:]
                    lon = o_dataset[DtmConstants.DIM_LON][:]

                    for row in range(data.shape[0]):
                        for col in range(data.shape[1]):
                            cell_lat = lat[row]
                            cell_lon = lon[col]
                            if -4.001 <= cell_lon <= -3.992 and 47.010 <= cell_lat <= 47.017:
                                self.assertTrue(
                                    np.ma.is_masked(data[row, col]),
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                            elif -3.989 <= cell_lon <= -3.982 and 46.999 <= cell_lat <= 47.006:
                                self.assertTrue(
                                    np.ma.is_masked(data[row, col]),
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                            elif np.ma.is_masked(data[row, col]):
                                self.assertTrue(
                                    np.ma.is_masked(i_data[row, col]),
                                    msg=f"Error while checking cell[{row},{col}] coord(lon,lat)=({cell_lon},{cell_lat})",
                                )
                            else:
                                self.assertEqual(
                                    data[row, col],
                                    i_data[row, col],
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
        finally:
            os.remove(o_path)
            os.remove(kml_1)
            os.remove(kml_2)

    def test_reset_cell_geo_kml_zone_multiple(self):
        # Parameters
        i_paths = [self.path_2]
        o_path = self.path_2[:-3] + "-zero-multiple-zone" + DtmConstants.EXTENSION_NC
        coord_1 = [[-4.001, 47.017], [-3.992, 47.017], [-3.992, 47.010], [-4.001, 47.010]]
        coord_2 = [[-3.989, 47.006], [-3.982, 47.006], [-3.982, 46.999], [-3.989, 46.999]]
        coords = {"zone1": coord_1, "zone2": coord_2}
        kml_1 = create_kml(dir=self.directory, coords=coords)
        kml = [kml_1]
        params = {"i_paths": i_paths, "o_paths": [o_path], "mask": kml}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # Verify
        try:
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path_2) as i_dataset:
                for layer in DtmConstants.LAYERS:
                    data = o_dataset[layer][:]
                    i_data = i_dataset[layer][:]
                    lat = o_dataset[DtmConstants.DIM_LAT][:]
                    lon = o_dataset[DtmConstants.DIM_LON][:]

                    for row in range(data.shape[0]):
                        for col in range(data.shape[1]):
                            cell_lat = lat[row]
                            cell_lon = lon[col]
                            # check zone1
                            if -4.001 <= cell_lon <= -3.992 and 47.010 <= cell_lat <= 47.017:
                                self.assertTrue(
                                    np.ma.is_masked(data[row, col]),
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                            elif -3.989 <= cell_lon <= -3.982 and 46.999 <= cell_lat <= 47.006:
                                self.assertTrue(
                                    np.ma.is_masked(data[row, col]),
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
                            elif not np.ma.is_masked(data[row, col]):
                                self.assertEqual(
                                    data[row, col],
                                    i_data[row, col],
                                    msg=f"Error while checking cell[{row},{col}]",
                                )
        finally:
            os.remove(o_path)
            os.remove(kml_1)

    def test_reset_cell_equal(self):
        # Parameters
        i_paths = [self.path]
        filters = [{"filter_layer": DtmConstants.ELEVATION_NAME, "oper": "equal", "a": self.VAL_EQ}]
        params = {"i_paths": i_paths, "filters": filters}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # Verify
        o_path = self.path[:-3] + "-zeroed" + DtmConstants.EXTENSION_NC
        try:
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                i_data = i_dataset[DtmConstants.ELEVATION_NAME][:]
                self._check_equal_filter(o_dataset, i_dataset, i_data, DtmConstants.ELEVATION_NAME)
        finally:
            os.remove(o_path)

    def test_reset_all_layer_for_elevation_equals_to_1010(self):
        """
        Reset all layers cells where elevation == 1010
        """
        # Parameters
        i_paths = [self.path]
        filters = [
            {
                "reset_layer": const.ALL_LAYERS,
                "filter_layer": DtmConstants.ELEVATION_NAME,
                "oper": "equal",
                "a": self.VAL_EQ,
            }
        ]
        params = {"i_paths": i_paths, "filters": filters}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # Verify some layers
        o_path = self.path[:-3] + "-zeroed" + DtmConstants.EXTENSION_NC
        try:
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                i_elev = i_dataset[DtmConstants.ELEVATION_NAME][:]
                for layer in [
                    DtmConstants.ELEVATION_NAME,
                    DtmConstants.ELEVATION_MIN,
                    DtmConstants.STDEV,
                    DtmConstants.VALUE_COUNT,
                ]:
                    self._check_equal_filter(o_dataset, i_dataset, i_elev, layer)
        finally:
            os.remove(o_path)

    def test_reset_value_count_and_elevation_min_for_elevation_equals_to_1010(self):
        """
        Reset only layer "value_count" and "elevation_min" where elevation == 1010
        """
        # Parameters
        i_paths = [self.path]
        filters = [
            {
                "reset_layer": DtmConstants.VALUE_COUNT,
                "filter_layer": DtmConstants.ELEVATION_NAME,
                "oper": "equal",
                "a": self.VAL_EQ,
            },
            {
                "reset_layer": DtmConstants.ELEVATION_MIN,
                "filter_layer": DtmConstants.ELEVATION_NAME,
                "oper": "more_than",
                "a": self.VAL_EQ,
            },
        ]
        params = {"i_paths": i_paths, "operator": const.OPERATOR_AND, "filters": filters}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # Verify some layers
        o_path = self.path[:-3] + "-zeroed" + DtmConstants.EXTENSION_NC
        try:
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                # Elevations have not changed
                i_elev = i_dataset[DtmConstants.ELEVATION_NAME][:].data
                o_elev = o_dataset[DtmConstants.ELEVATION_NAME][:].data
                self.assertTrue(np.array_equal(i_elev, o_elev))

                # Value count has some reset cells
                self._check_equal_filter(o_dataset, i_dataset, i_elev, DtmConstants.VALUE_COUNT)

                # Value elevation_min has some reset cells
                self._check_more_than_filter(o_dataset, i_dataset, i_elev, DtmConstants.ELEVATION_MIN)

        finally:
            os.remove(o_path)

    def test_reset_all_layer_for_elevation_missing(self):
        """
        Reset all layers cells where elevation is missing
        """
        # Parameters
        i_paths = [self.path]
        filters = [{"filter_layer": DtmConstants.ELEVATION_NAME, "oper": "missing"}]
        params = {"i_paths": i_paths, "filters": filters}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # Verify some layers
        o_path = self.path[:-3] + "-zeroed" + DtmConstants.EXTENSION_NC
        try:
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                i_elev = i_dataset[DtmConstants.ELEVATION_NAME][:]
                for layer in [
                    DtmConstants.ELEVATION_NAME,
                    DtmConstants.ELEVATION_MIN,
                    DtmConstants.STDEV,
                    DtmConstants.VALUE_COUNT,
                    DtmConstants.CDI_INDEX,
                ]:
                    self._check_missing_filter(o_dataset, i_dataset, i_elev, layer)
        finally:
            os.remove(o_path)

    def test_or_operator(self):
        """
        Reset layers where elevation <= VAL_MIN or elevation >= VAL_MAX (like a not between)
        """
        # Parameters
        i_paths = [self.path]
        filters = [
            {
                "reset_layer": const.ALL_LAYERS,
                "filter_layer": DtmConstants.ELEVATION_NAME,
                "oper": "less_than",
                "a": self.VAL_MIN,
            },
            {
                "reset_layer": const.ALL_LAYERS,
                "filter_layer": DtmConstants.ELEVATION_NAME,
                "oper": "more_than",
                "a": self.VAL_MAX,
            },
        ]
        params = {"i_paths": i_paths, "operator": const.OPERATOR_OR, "filters": filters}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # Verify some layers
        o_path = self.path[:-3] + "-zeroed" + DtmConstants.EXTENSION_NC
        try:
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                # Elevations have not changed
                i_elev = i_dataset[DtmConstants.ELEVATION_NAME][:]
                o_elev = o_dataset[DtmConstants.ELEVATION_NAME][:]

                # Apply the condition to the input file by masking expected values
                np.ma.masked_where((i_elev <= self.VAL_MIN) | (i_elev >= self.VAL_MAX), i_elev, copy=False)
                # Elevation mask of the NC variable must be equal
                self.assertTrue(np.array_equal(i_elev.mask, o_elev.mask))

        finally:
            os.remove(o_path)

    def test_and_operator(self):
        """
        Reset layers where elevation >= VAL_MIN AND elevation <= VAL_MAX (like a between)
        """
        # Parameters
        i_paths = [self.path]
        filters = [
            {
                "reset_layer": const.ALL_LAYERS,
                "filter_layer": DtmConstants.ELEVATION_NAME,
                "oper": "more_than",
                "a": self.VAL_MIN,
            },
            {
                "reset_layer": const.ALL_LAYERS,
                "filter_layer": DtmConstants.ELEVATION_NAME,
                "oper": "less_than",
                "a": self.VAL_MAX,
            },
        ]
        params = {"i_paths": i_paths, "operator": const.OPERATOR_AND, "filters": filters}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # Verify some layers
        o_path = self.path[:-3] + "-zeroed" + DtmConstants.EXTENSION_NC
        try:
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                # Elevations have not changed
                i_elev = i_dataset[DtmConstants.ELEVATION_NAME][:]
                o_elev = o_dataset[DtmConstants.ELEVATION_NAME][:]

                # Apply the condition to the input file by masking expected values
                np.ma.masked_where((i_elev >= self.VAL_MIN) & (i_elev <= self.VAL_MAX), i_elev, copy=False)
                # Elevation mask of the NC variable must be equal
                self.assertTrue(np.array_equal(i_elev.mask, o_elev.mask))

        finally:
            os.remove(o_path)

    def test_reset_cell_less_than(self):
        # Parameters
        i_paths = [self.path]
        filters = [{"filter_layer": DtmConstants.ELEVATION_NAME, "oper": "less_than", "a": self.VAL_MIN, "b": 0}]
        params = {"i_paths": i_paths, "filters": filters}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # Verify
        o_path = self.path[:-3] + "-zeroed" + DtmConstants.EXTENSION_NC
        try:
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                data = o_dataset[DtmConstants.ELEVATION_NAME][:]
                i_data = i_dataset[DtmConstants.ELEVATION_NAME][:]
                for row in range(data.shape[0]):
                    for col in range(data.shape[1]):
                        if i_data[row, col] <= self.VAL_MIN:
                            self.assertTrue(np.ma.is_masked(data[row, col]))
                        else:
                            self.assertEqual(data[row, col], i_data[row, col])

        finally:
            os.remove(o_path)

    def test_reset_cell_more_than(self):
        # Parameters
        i_paths = [self.path]
        filters = [{"filter_layer": DtmConstants.ELEVATION_NAME, "oper": "more_than", "a": self.VAL_MAX, "b": 0}]
        params = {"i_paths": i_paths, "filters": filters}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # Verify
        o_path = self.path[:-3] + "-zeroed" + DtmConstants.EXTENSION_NC
        try:
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                data = o_dataset[DtmConstants.ELEVATION_NAME][:]
                i_data = i_dataset[DtmConstants.ELEVATION_NAME][:]
                for row in range(data.shape[0]):
                    for col in range(data.shape[1]):
                        if i_data[row, col] >= self.VAL_MAX:
                            self.assertTrue(np.ma.is_masked(data[row, col]))
                        else:
                            self.assertEqual(data[row, col], i_data[row, col])
        finally:
            os.remove(o_path)

    def test_reset_cell_between(self):
        # Parameters
        i_paths = [self.path]
        filters = [
            {"filter_layer": DtmConstants.ELEVATION_NAME, "oper": "between", "a": self.VAL_MIN, "b": self.VAL_MAX}
        ]
        params = {"i_paths": i_paths, "filters": filters}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # Verify
        o_path = self.path[:-3] + "-zeroed" + DtmConstants.EXTENSION_NC
        try:

            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                data = o_dataset[DtmConstants.ELEVATION_NAME][:]
                i_data = i_dataset[DtmConstants.ELEVATION_NAME][:]
                for row in range(data.shape[0]):
                    for col in range(data.shape[1]):
                        if i_data[row, col] >= self.VAL_MIN and i_data[row, col] <= self.VAL_MAX:
                            self.assertTrue(np.ma.is_masked(data[row, col]))
                        else:
                            self.assertEqual(data[row, col], i_data[row, col])
        finally:
            os.remove(o_path)

    def test_reset_cell_between_double_filter(self):
        # Parameters
        i_paths = [self.path]
        filters = [
            {"filter_layer": DtmConstants.ELEVATION_NAME, "oper": "less_than", "a": self.VAL_MAX, "b": 0},
            {"filter_layer": DtmConstants.ELEVATION_NAME, "oper": "more_than", "a": self.VAL_MIN, "b": 0},
        ]
        params = {"i_paths": i_paths, "filters": filters}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # Verify
        o_path = self.path[:-3] + "-zeroed" + DtmConstants.EXTENSION_NC
        try:
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                data = o_dataset[DtmConstants.ELEVATION_NAME][:]
                i_data = i_dataset[DtmConstants.ELEVATION_NAME][:]
                for row in range(data.shape[0]):
                    for col in range(data.shape[1]):
                        if i_data[row, col] >= self.VAL_MIN and i_data[row, col] <= self.VAL_MAX:
                            self.assertTrue(np.ma.is_masked(data[row, col]))
                        else:
                            self.assertEqual(data[row, col], i_data[row, col])
        finally:
            os.remove(o_path)

    def test_reset_cell_cdi_and_between(self):
        # Parameters
        i_paths = [self.path]
        cdi = "500"
        filters = [
            {"filter_layer": DtmConstants.ELEVATION_NAME, "oper": "between", "a": self.VAL_MIN, "b": self.VAL_MAX},
            {"filter_layer": const.CDI_LAYER, "oper": "equal", "cdi": cdi},
        ]
        params = {"i_paths": i_paths, "filters": filters}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # Verify
        o_path = self.path[:-3] + "-zeroed" + DtmConstants.EXTENSION_NC
        try:

            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                o_elevations = o_dataset[DtmConstants.ELEVATION_NAME][:]
                i_elevations = i_dataset[DtmConstants.ELEVATION_NAME][:]

                o_cdi_reference = o_dataset[DtmConstants.CDI][:]
                i_cdi_reference = i_dataset[DtmConstants.CDI][:]

                ind_cdi = int(np.where(i_cdi_reference == cdi)[0])
                i_len_cdi_reference = len(i_cdi_reference[i_cdi_reference != ""])
                o_len_cdi_reference = len(o_cdi_reference[o_cdi_reference != ""])

                self.assertTrue(cdi in i_cdi_reference)
                self.assertTrue(cdi in o_cdi_reference)
                self.assertEqual(o_len_cdi_reference, i_len_cdi_reference)
                self.assertTrue(o_cdi_reference[ind_cdi] == cdi)

                o_cdi_indexes = o_dataset[DtmConstants.CDI_INDEX][:]
                i_cdi_indexes = i_dataset[DtmConstants.CDI_INDEX][:]

                for row in range(o_cdi_indexes.shape[0]):
                    for col in range(o_cdi_indexes.shape[1]):
                        if (
                            self.VAL_MIN <= i_elevations[row, col] <= self.VAL_MAX
                            and i_cdi_indexes[row, col] == ind_cdi
                        ):
                            self.assertTrue(np.ma.is_masked(o_elevations[row, col]))
                            self.assertTrue(np.ma.is_masked(o_cdi_indexes[row, col]))
                        else:
                            self.assertEqual(o_elevations[row, col], i_elevations[row, col])

                        # Verify no shift
                        if i_cdi_indexes[row, col] > ind_cdi:
                            self.assertTrue(o_cdi_indexes[row, col] == i_cdi_indexes[row, col])

        finally:
            os.remove(o_path)

    def test_reset_cell_cdi(self):
        # Parameters
        i_paths = [self.path]
        cdi = "500"
        filters = [{"filter_layer": const.CDI_LAYER, "oper": "equal", "cdi": cdi}]
        params = {"i_paths": i_paths, "filters": filters}

        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # Verify
        o_path = self.path[:-3] + "-zeroed" + DtmConstants.EXTENSION_NC
        try:
            with nc.Dataset(o_path) as o_dataset, nc.Dataset(self.path) as i_dataset:
                o_cdi = o_dataset[DtmConstants.CDI][:]
                i_cdi = i_dataset[DtmConstants.CDI][:]

                ind_cdi = int(np.where(i_cdi == cdi)[0])
                old_max = len(i_cdi[i_cdi != ""])

                self.assertTrue(cdi in i_cdi)
                self.assertFalse(cdi in o_cdi)
                self.assertEqual(len(o_cdi[o_cdi != ""]), old_max - 1)
                self.assertTrue(o_cdi[ind_cdi] == "600")

                data = o_dataset[DtmConstants.CDI_INDEX][:]
                i_data = i_dataset[DtmConstants.CDI_INDEX][:]
                for row in range(data.shape[0]):
                    for col in range(data.shape[1]):
                        if i_data[row, col] > ind_cdi:
                            self.assertTrue(data[row, col] == i_data[row, col] - 1)
                        elif i_data[row, col] == ind_cdi:
                            self.assertTrue(np.ma.is_masked(data[row, col]))
                        else:
                            self.assertTrue(data[row, col] == i_data[row, col])
        finally:
            os.remove(o_path)

    def reset_cell_multiple_cdi(self):

        dir_util.get_test_directory()

        i_paths = [dir_util.get_test_directory() + "/raw/reset_cell_multi_cdi.nc"]
        o_path = tempfile.mktemp(".dtm.nc")
        filters = [{"filter_layer": DtmConstants.ELEVATION_NAME, "oper": "equal", "a": -2.45, "b": 0}]

        # params = {"i_paths": i_paths, "cdi": "SDN:CDI:LOCAL:486_1", "o_paths": [o_path], "filters": filters}
        params = {"i_paths": i_paths, "o_paths": [o_path], "filters": filters}
        # Process
        resetCell = ResetCellProcess(**params)
        resetCell()

        # now check that central cell was set to one and more important that the CDI_index 1 was removed

        with nc.Dataset(o_path) as output:
            cdi_names = output[DtmConstants.CDI][:]
            cdi_names = cdi_util.trim_string_array(cdi_names)
            assert len(cdi_names) == 1
            assert cdi_names[0] == "SDN:CDI:LOCAL:486_1"
            assert np.ma.is_masked(output[DtmConstants.ELEVATION_NAME][1, 1])

    def _check_equal_filter(
        self, o_dataset: nc.Dataset, i_dataset: nc.Dataset, filter_layer: np.ndarray, layer_name: str
    ):
        o_val = o_dataset[layer_name][:]
        i_val = i_dataset[layer_name][:]
        # Apply the condition to the input file by masking expected values
        np.ma.masked_where(filter_layer == self.VAL_EQ, i_val, copy=False)
        # Mask of the output NC variable must be equal
        self.assertTrue(np.array_equal(i_val.mask, o_val.mask))

    def _check_more_than_filter(
        self, o_dataset: nc.Dataset, i_dataset: nc.Dataset, filter_layer: np.ndarray, layer_name: str
    ):
        o_val = o_dataset[layer_name][:]
        i_val = i_dataset[layer_name][:]
        # Apply the condition to the input file by masking expected values
        np.ma.masked_where(filter_layer >= self.VAL_EQ, i_val, copy=False)
        # Mask of the output NC variable must be equal
        self.assertTrue(np.array_equal(i_val.mask, o_val.mask))

    def _check_missing_filter(
        self, o_dataset: nc.Dataset, i_dataset: nc.Dataset, filter_layer: np.ndarray, layer_name: str
    ):
        o_val = o_dataset[layer_name][:]
        i_val = i_dataset[layer_name][:]
        # Apply the condition to the input file by masking expected values
        np.ma.masked_where(filter_layer == self.VAL_MISSING, i_val, copy=False)
        # Mask of the output NC variable must be equal
        self.assertTrue(np.array_equal(i_val.mask, o_val.mask))

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.path)
        os.remove(cls.path_2)
        print(f"End of {cls.__name__}.")


if __name__ == "__main__":
    unittest.main()
