#! /usr/bin/env python3
# coding: utf-8

import logging
import math
import os
import sys
import tempfile as tmp
from typing import Dict, Optional, Tuple

import numpy as np

import pyat.common.geo_file as gf
import pyat.dtm.dtm_standard_constants as DtmConstant
from pyat.dtm import dtm_driver


def make_dtm_with_data(
    cell1: Tuple[float, float], cell2: Tuple[float, float], data: Dict[str, np.ndarray], temp_dir: Optional[str] = None
):
    """
    Produce a the DTM and return its path.
    cell1(lon, lat) are the coord of the center of the first cell
    cell2(lon, lat) are the coord of the center of the last cell
    """
    path_dtm = tmp.mktemp(suffix=".dtm.nc", dir=temp_dir)
    with dtm_driver.open_dtm(path_dtm, "w") as o_driver:
        dtm_file = o_driver.dtm_file
        elevations = data[DtmConstant.ELEVATION_NAME]
        dtm_file.row_count, dtm_file.col_count = elevations.shape[0], elevations.shape[1]
        dtm_file.spatial_resolution_x = abs(cell1[0] - cell2[0]) / (dtm_file.col_count - 1)
        dtm_file.spatial_resolution_y = abs(cell1[1] - cell2[1]) / (dtm_file.row_count - 1)
        dtm_file.north = max(cell1[1], cell2[1]) + dtm_file.spatial_resolution_y / 2.0
        dtm_file.south = min(cell1[1], cell2[1]) - dtm_file.spatial_resolution_y / 2.0
        dtm_file.east = max(cell1[0], cell2[0]) + dtm_file.spatial_resolution_x / 2.0
        dtm_file.west = min(cell1[0], cell2[0]) - dtm_file.spatial_resolution_x / 2.0

        o_driver.initialize_file()
        cdi_index = data.get(DtmConstant.CDI_INDEX, None)
        if cdi_index is not None:
            all_cdi_indexes = list(np.unique(cdi_index))
            if dtm_driver.get_missing_value(DtmConstant.CDI_INDEX) in all_cdi_indexes:
                all_cdi_indexes.remove(dtm_driver.get_missing_value(DtmConstant.CDI_INDEX))
            cdis = ["CDI_" + str(index) for index in all_cdi_indexes]
            o_driver.create_cdi_reference_variable(cdis)
        for layer, layer_data in data.items():
            o_driver.add_layer(layer, layer_data)
    return path_dtm


def make_dtm_from_SW(
    origin_SW: Tuple[float, float],
    spatial_resolution: float,
    data: Dict[str, np.ndarray],
    temp_dir: Optional[str] = None,
):
    """
    Produce a the DTM and return its path.
    origin_SW(lon, lat) are the coord of the South West origin
    """
    elevations = data[DtmConstant.ELEVATION_NAME]
    row_count, col_count = elevations.shape[1], elevations.shape[0]
    return make_dtm_with_data(
        origin_SW,
        (origin_SW[0] + (col_count - 1) * spatial_resolution, origin_SW[1] + (row_count - 1) * spatial_resolution),
        data,
        temp_dir,
    )


def make_dtm_from_NW(
    origin_NW: Tuple[float, float],
    spatial_resolution: float,
    data: Dict[str, np.ndarray],
    temp_dir: Optional[str] = None,
):
    """
    Produce a the DTM and return its path.
    origin_NW(lon, lat) are the coord of the North West origin
    """
    elevations = data[DtmConstant.ELEVATION_NAME]
    row_count, col_count = elevations.shape[1], elevations.shape[0]
    return make_dtm_with_data(
        (origin_NW[0] - (col_count - 1) * spatial_resolution, origin_NW[1] - (row_count - 1) * spatial_resolution),
        origin_NW,
        data,
        temp_dir,
    )


def make_dtm_with_elevations(
    cell1: Tuple[float, float],
    cell2: Tuple[float, float],
    elevations: np.ndarray,
    temp_dir: Optional[str] = None,
):
    """
    Produce a the DTM and return its path.
    cell1(lon, lat) are the coord of the center of the first cell
    cell2(lon, lat) are the coord of the center of the last cell
    """
    return make_dtm_with_data(cell1, cell2, {DtmConstant.ELEVATION_NAME: elevations}, temp_dir)


# N, S, E, W
minute = 1 / 60.0
second = minute**2
RESOLUTION = 3.75 * second

geoBox1 = np.array([47 + minute, 47, -4 + minute, -4])
geoBox1_bis = np.array([47 + 2 * minute, 47, -4 + minute, -4])
geoBox1_bis_bis = np.array([47 + 3 * minute, 47, -4 + minute, -4])
geoBox2 = np.array([47 + minute / 2.0, 47 - minute / 4.0, -4 + minute / 2.0, -4 - minute / 4.0])
geoBox3 = np.array([47 + minute, 47 + minute / 4.0, -4 + minute, -4 + minute / 4.0])
geoBox4 = np.array([47, 46, -4, -5])
geoBox5 = np.array([47 - 15 * minute, 46 - 15 * minute, -4 - 15 * minute, -5 - 15 * minute])
geoBox6 = np.array([47 - 15 * minute + 0.02, 46 - 15 * minute + 0.02, -4 - 15 * minute + 0.02, -5 - 15 * minute + 0.02])
geoBox7 = np.array([47 - 15 * minute + 0.02, 46 + 15 * minute + 0.02, -4 - 15 * minute + 0.02, -5 - 15 * minute + 0.02])

ROW = "row"
COL = "col"


def fill_zone(data, zone, value):
    for line in range(zone[0], zone[1] + 1):
        for col in range(zone[2], zone[3] + 1):
            data[line, col] = value

    return data


def fill_step(data, zone, value):
    for line in range(zone[0], zone[1] + 1):
        for col in range(zone[2], zone[3] - (line - zone[0]) + 1):
            data[line, col] = value

    return data


def fill_gradient(data, zone, value, step):
    for line in range(zone[0], zone[1] + 1):
        for col in range(zone[2], zone[3] - (line - zone[0]) + 1):
            data[line, col] = value + (line - zone[0]) * step

    return data


class DtmGenerator:
    """Class generator of NetCdf4 file for the processes tests."""

    def __init__(self, directory):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.directory = directory

        self.shape = None
        self.path = None

    def initialize_file(self, **kwargs):
        geobox = kwargs["geobox"]
        spatial_reference = gf.SR_WGS_84
        if "spatial_reference" in kwargs:
            spatial_reference = kwargs["spatial_reference"]

        size = {}
        # extend in advance to
        size[ROW] = math.ceil((geobox[0] - (geobox[1] + 0.5 * RESOLUTION)) / RESOLUTION)
        size[COL] = math.ceil((geobox[2] - (geobox[3] + 0.5 * RESOLUTION)) / RESOLUTION)

        if not "path" in kwargs:
            name = "generated"

            if "longlat" in kwargs:
                name += "_longlat"

            name += "_{}x{}".format(size[ROW], size[COL])

            if "pattern" in kwargs:
                name += "_pattern"

            if "value" in kwargs:
                name += "_{}".format(kwargs["value"])

            if "value_2" in kwargs:
                name += "_{}".format(kwargs["value_2"])

            if "opt" in kwargs:
                name += "_{}".format(kwargs["opt"])

            if "mode" in kwargs:
                name += "_{}".format(kwargs["mode"])

            if "missing_value" in kwargs:
                if kwargs["missing_value"]:
                    name += "_missing_value_{}".format(kwargs["missing_value"])

            if "zones" in kwargs:
                for ind, zone in enumerate(kwargs["zones"]):
                    name += "_{}_{}".format(zone, kwargs["values"][ind])

            name += DtmConstant.EXTENSION_NC
        else:
            name = kwargs["path"]

        self.logger.info("Create file test: {}.".format(name))

        self.path = os.path.join(self.directory, name)
        result = dtm_driver.DtmDriver(self.path)
        result.create_file(
            col_count=size[COL],
            origin_x=geobox[3] + 0.5 * RESOLUTION,
            spatial_resolution_x=RESOLUTION,
            row_count=size[ROW],
            origin_y=geobox[1] + 0.5 * RESOLUTION,
            spatial_resolution_y=RESOLUTION,
            spatial_reference=spatial_reference,
            overwrite=True,
        )

        self.shape = (size[ROW], size[COL])

        return result

    def create_1(self, value=None, missing_value=None):
        driver = self.initialize_file(geobox=geoBox1, value=value, missing_value=missing_value)

        for layer in dtm_driver.LAYER_NAMES:
            if layer != DtmConstant.CDI:
                data = np.ma.MaskedArray(np.zeros(self.shape), mask=True)
                if layer == DtmConstant.ELEVATION_NAME:
                    if not value:
                        data[:] = np.random.rand(self.shape[0], self.shape[0])
                    else:
                        data[:] = value

                    if missing_value:
                        data.mask[missing_value[0], missing_value[1]] = True

                driver.add_layer(layer, data)

        driver.create_cdi_reference_variable([str(value)])
        driver.close()

        return self.path

    def create_1_without_value_count(self, value=None, missing_value=None):
        driver = self.initialize_file(
            geobox=geoBox1, value=value, missing_value=missing_value, opt="without_value_count"
        )

        for layer in dtm_driver.LAYER_NAMES:
            if layer == DtmConstant.VALUE_COUNT:
                continue
            if layer != DtmConstant.CDI:
                data = np.ma.MaskedArray(np.zeros(self.shape), mask=True)
                if layer == DtmConstant.ELEVATION_NAME:
                    if not value:
                        data[:] = np.random.rand(self.shape[0], self.shape[0])
                    else:
                        data[:] = value

                    if missing_value:
                        data.mask[missing_value[0], missing_value[1]] = True

                driver.add_layer(layer, data)

        driver.create_cdi_reference_variable([str(value)])
        driver.close()

        return self.path

    def create_pattern(self, value, pair_impair=0, line_col=1, number=2, allValue=True, **kwargs):
        driver = self.initialize_file(value=value, geobox=geoBox1, pattern=True, **kwargs)

        layers = list(dtm_driver.LAYER_NAMES)
        if "except_layer" in kwargs:
            for i in kwargs["except_layer"]:
                layers.remove(dtm_driver.LAYER_NAMES[i])

        for layer in layers:
            if layer != DtmConstant.CDI:
                data = np.ma.MaskedArray(np.zeros(self.shape), mask=True)
                for line in range(data.shape[0]):
                    for col in range(data.shape[1]):
                        coor = [line, col]
                        impair = coor[line_col] % number != 0
                        pair = coor[line_col] % number == 0
                        cond = [pair, impair]
                        if cond[pair_impair] or allValue:
                            if layer == DtmConstant.ELEVATION_NAME:
                                data[line, col] = value

                            if layer == DtmConstant.ELEVATION_MIN:
                                data[line, col] = value - 1

                            if layer == DtmConstant.ELEVATION_MAX:
                                data[line, col] = value + 1

                            if layer == DtmConstant.STDEV:
                                data[line, col] = abs(value / 10.0)

                            if layer == DtmConstant.VALUE_COUNT:
                                data[line, col] = value

                            if layer == DtmConstant.CDI_INDEX:
                                data[line, col] = 0

                driver.add_layer(layer, data)
            else:
                driver.create_cdi_reference_variable([str(value)])

        driver.close()

        return self.path

    def create_pattern_for_cdi_test(self, value, mode, **kwargs):
        driver = self.initialize_file(value=value, geobox=geoBox1, mode=mode, pattern=True)

        if "value_count" in kwargs:
            vl = kwargs["value_count"]
        else:
            vl = value

        if "cdi" in kwargs:
            cdi = kwargs["cdi"]
        else:
            cdi = str(value)

        for layer in dtm_driver.LAYER_NAMES:
            if layer != DtmConstant.CDI:
                data = np.ma.MaskedArray(np.zeros(self.shape), mask=True)
                for line in range(data.shape[0]):
                    for col in range(data.shape[1]):
                        if mode == 1:
                            if line % 2 == 0:
                                if layer == DtmConstant.ELEVATION_NAME:
                                    data[line, col] = value

                                if layer == DtmConstant.VALUE_COUNT:
                                    data[line, col] = vl

                                if layer == DtmConstant.CDI_INDEX:
                                    data[line, col] = 0
                        elif mode == 2:
                            if col % 2 == 0:
                                if layer == DtmConstant.ELEVATION_NAME:
                                    data[line, col] = value

                                if layer == DtmConstant.VALUE_COUNT:
                                    data[line, col] = vl

                                if layer == DtmConstant.CDI_INDEX:
                                    data[line, col] = 0
                        elif mode == 3:
                            if line % 2 == 0 and col % 2 == 0:
                                if layer == DtmConstant.ELEVATION_NAME:
                                    data[line, col] = value

                                if layer == DtmConstant.VALUE_COUNT:
                                    data[line, col] = vl

                                if layer == DtmConstant.CDI_INDEX:
                                    data[line, col] = 0
                        else:
                            if layer == DtmConstant.ELEVATION_NAME:
                                data[line, col] = value

                            if layer == DtmConstant.VALUE_COUNT:
                                data[line, col] = vl

                            if layer == DtmConstant.CDI_INDEX:
                                data[line, col] = 0

                driver.add_layer(layer, data)

        driver.create_cdi_reference_variable([cdi])
        driver.close()

        return self.path

    def create_pattern_sanity_check(self, value, **kwargs):
        if "value_count" in kwargs:
            vl = kwargs["value_count"]
        else:
            vl = value

        if "cdi" in kwargs:
            cdi = kwargs["cdi"]
        else:
            cdi = str(value)

        if "mode" in kwargs:
            mode = kwargs["mode"]
        else:
            mode = 1

        driver = self.initialize_file(value=value, geobox=geoBox1, mode=mode, pattern=True)

        for layer in dtm_driver.LAYER_NAMES:
            if layer != DtmConstant.CDI:
                data = np.ma.MaskedArray(np.zeros(self.shape), mask=True)
                for line in range(data.shape[0]):
                    for col in range(data.shape[1]):
                        if line % 2 == 0:
                            if layer == DtmConstant.ELEVATION_NAME:
                                data[line, col] = value

                            elif layer == DtmConstant.VALUE_COUNT:
                                data[line, col] = vl

                            elif layer == DtmConstant.CDI_INDEX:
                                if mode == 1:
                                    if col % 2 == 0:
                                        data[line, col] = 0
                                    else:
                                        data[line, col] = 1
                                elif mode == 2:
                                    data[line, col] = 0
                                elif mode == 3:
                                    if col == 0:
                                        data[line, col] = -1  # missing_value
                                    else:
                                        data[line, col] = 0
                driver.add_layer(layer, data)
        if mode == 1:
            cdis = [cdi, cdi, cdi]
        else:
            cdis = [cdi]
        driver.create_cdi_reference_variable(cdis)
        driver.close()

        return self.path

    def create_pattern_interpolation(self, value, value_2):
        driver = self.initialize_file(value=value, value_2=value_2, geobox=geoBox1, pattern=True)

        for layer in dtm_driver.LAYER_NAMES:
            if layer != DtmConstant.CDI:
                data = np.ma.MaskedArray(np.zeros(self.shape), mask=True)
                for line in range(data.shape[0]):

                    for col in range(data.shape[1]):
                        if layer == DtmConstant.ELEVATION_NAME:
                            if line % 4 == 0:
                                data[line, col] = value
                            elif line % 3 == 0:
                                data[line, col] = value_2

                        if layer == DtmConstant.ELEVATION_MIN:
                            data[line, col] = value - 1

                        if layer == DtmConstant.ELEVATION_MAX:
                            data[line, col] = value + 1

                        if layer == DtmConstant.STDEV:
                            data[line, col] = abs(value / 10.0)

                        if layer == DtmConstant.CDI_INDEX:
                            data[line, col] = 0

                driver.add_layer(layer, data)

        driver.create_cdi_reference_variable([str(value)])
        driver.close()
        return self.path

    def create_pattern_smoothing(self, value, value_2):
        driver = self.initialize_file(value=value, value_2=value_2, geobox=geoBox1, pattern=True)

        for layer in dtm_driver.LAYER_NAMES:
            if layer != DtmConstant.CDI:
                data = np.ma.MaskedArray(np.zeros(self.shape), mask=True)
                for line in range(data.shape[0]):
                    for col in range(data.shape[1]):
                        if col % 2 != 0:
                            if layer == DtmConstant.ELEVATION_NAME:
                                data[line, col] = value
                        else:
                            if layer == DtmConstant.ELEVATION_NAME:
                                data[line, col] = value_2

                driver.add_layer(layer, data)

        driver.create_cdi_reference_variable([str(value)])

        driver.close()
        return self.path

    def create_long_lat(self, geoBox, zones, values, opt):
        driver = self.initialize_file(values=values, zones=zones, geobox=geoBox, longlat=True, opt=opt)

        for layer in dtm_driver.LAYER_NAMES:
            if layer != DtmConstant.CDI:
                data = np.ma.MaskedArray(np.zeros(self.shape), mask=True)
                if layer == DtmConstant.ELEVATION_NAME:
                    for ind, zone in enumerate(zones):
                        if opt == "steps":
                            data = fill_step(data[:], zone, values[ind])
                        elif opt == "zone":
                            data = fill_zone(data[:], zone, values[ind])
                        elif opt == "gradient":
                            data = fill_gradient(data[:], zone, 130, 0.01)

                driver.add_layer(layer, data)

        driver.create_cdi_reference_variable([str(i) for i in values])
        driver.close()

        return self.path

    def create_reduction_file(self):
        driver = self.initialize_file(geobox=geoBox1, opt="reduction_file")

        cdi_value = []

        for layer in dtm_driver.LAYER_NAMES:
            if layer != DtmConstant.CDI:
                data = np.ma.MaskedArray(np.zeros(self.shape), mask=True)
                for line in range(data.shape[0]):
                    for col in range(data.shape[1]):
                        value = line * 100 + col + 1

                        if layer == DtmConstant.ELEVATION_NAME:
                            data[line, col] = value

                        elif layer == DtmConstant.ELEVATION_MIN:
                            data[line, col] = value - 1

                        elif layer == DtmConstant.ELEVATION_MAX:
                            data[line, col] = value + 1

                        elif layer == DtmConstant.STDEV:
                            data[line, col] = value

                        elif layer == DtmConstant.VALUE_COUNT:
                            data[line, col] = value

                        elif layer == DtmConstant.CDI_INDEX:
                            data[line, col] = len(cdi_value)
                            cdi_value.append(value)

                driver.add_layer(layer, data)

        driver.create_cdi_reference_variable([str(i) for i in cdi_value])
        driver.close()

        return self.path

    def create_reset_cell_file(self):
        driver = self.initialize_file(geobox=geoBox1, opt="reset_cells_file")

        cdis = []

        for layer in dtm_driver.LAYER_NAMES:
            if layer != DtmConstant.CDI:
                data = np.ma.MaskedArray(np.zeros(self.shape), mask=True)
                for row in range(data.shape[0]):
                    for col in range(data.shape[1]):
                        value = row * 100 + 10 * col

                        if layer == DtmConstant.ELEVATION_NAME:
                            data[row, col] = value
                            # invalidate first value for missing value test
                            if row == 0 and col == 0:
                                value = np.nan

                        if layer == DtmConstant.ELEVATION_MIN:
                            data[row, col] = value - 1

                        if layer == DtmConstant.ELEVATION_MAX:
                            data[row, col] = value + 1

                        if layer == DtmConstant.STDEV:
                            data[row, col] = value

                        if layer == DtmConstant.VALUE_COUNT:
                            data[row, col] = value

                        if layer == DtmConstant.CDI_INDEX:
                            cdi = str(int(row * 100 + 100))
                            if not cdi in cdis:
                                cdis.append(cdi)
                            data[row, col] = len(cdis) - 1

                        if layer == DtmConstant.INTERPOLATION_FLAG:
                            data[row, col] = 0

                driver.add_layer(layer, data)

        driver.create_cdi_reference_variable(cdis=cdis)
        driver.close()

        return self.path

    def create_set_interpolation_file(self):
        driver = self.initialize_file(geobox=geoBox1, opt="set_interpolation_file")

        cdis = []

        for layer in dtm_driver.LAYER_NAMES:
            if layer != DtmConstant.CDI:
                data = np.ma.MaskedArray(np.zeros(self.shape), mask=True)
                for row in range(data.shape[0]):
                    for col in range(data.shape[1]):
                        value = row * 100 + 10 * col

                        if layer == DtmConstant.ELEVATION_NAME:
                            data[row, col] = value

                        if layer == DtmConstant.INTERPOLATION_FLAG:
                            if row != 0 and col != 0:
                                data[row, col] = 0

                driver.add_layer(layer, data)

        driver.create_cdi_reference_variable(cdis=cdis)
        driver.close()

        return self.path

    def create(self):
        path = []
        path.append(self.create_1(value=10))
        path.append(self.create_1(value=20))
        path.append(self.create_1(value=10, missing_value=(3, 10)))
        path.append(self.create_pattern(value=10))
        path.append(self.create_pattern(value=20, pair_impair=1, line_col=1, number=2, allValue=False))
        path.append(self.create_pattern(value=30, pair_impair=0, line_col=1, number=2, allValue=False))
        path.append(
            self.create_long_lat(
                geoBox=geoBox1, zones=np.array([[1, 3, 1, 3], [10, 15, 10, 12]]), values=[10, 20], opt="zone"
            )
        )
        path.append(
            self.create_long_lat(
                geoBox=geoBox1, zones=np.array([[3, 5, 1, 3], [8, 15, 0, 7]]), values=[20, 10], opt="zone"
            )
        )
        path.append(self.create_long_lat(geoBox=geoBox2, zones=np.array([[8, 11, 8, 11]]), values=[30], opt="zone"))
        path.append(self.create_long_lat(geoBox=geoBox3, zones=np.array([[0, 9, 0, 9]]), values=[30], opt="steps"))
        path.append(self.create_long_lat(geoBox=geoBox4, zones=np.array([[90, 700, 90, 700]]), values=[10], opt="zone"))
        path.append(
            self.create_long_lat(geoBox=geoBox5, zones=np.array([[90, 959, 90, 959]]), values=[20], opt="steps")
        )
        path.append(
            self.create_long_lat(geoBox=geoBox6, zones=np.array([[90, 959, 90, 959]]), values=[40], opt="steps")
        )
        path.append(
            self.create_long_lat(geoBox=geoBox7, zones=np.array([[0, 479, 240, 719]]), values=[20], opt="steps")
        )
        path.append(
            self.create_long_lat(geoBox=geoBox4, zones=np.array([[0, 959, 0, 959]]), values=[20], opt="gradient")
        )

        # Big Files
        path.append(self.create_long_lat(geoBox=geoBox1, zones=np.array([[0, 1, 0, 1]]), values=[100], opt="zone"))
        path.append(self.create_long_lat(geoBox=geoBox1, zones=np.array([[2, 3, 2, 3]]), values=[30], opt="zone"))
        path.append(self.create_long_lat(geoBox=geoBox1, zones=np.array([[4, 5, 4, 5]]), values=[2], opt="zone"))
        return path


if __name__ == "__main__":
    arg = sys.argv[1:]
    generator = DtmGenerator(arg[0])
    p = generator.create()
    p = generator.create_pattern(value=10, pair_impair=0, line_col=1, number=2, allValue=False, except_layer=[-1])
    p = generator.create_pattern(
        value=20, pair_impair=0, line_col=1, number=2, allValue=False, except_layer=[1, 2, 3, 4, 5, 6, 7, 8]
    )
    p = generator.create_pattern_interpolation(value=10, value_2=20)
    p = generator.create_long_lat(geoBox=geoBox4, zones=np.array([[90, 700, 90, 700]]), values=[10], opt="zone")
    p = generator.create_long_lat(
        geoBox=geoBox1, zones=np.array([[1, 3, 1, 3], [10, 15, 10, 12]]), values=[10, 20], opt="zone"
    )
    p = generator.create_long_lat(
        geoBox=geoBox1, zones=np.array([[1, 3, 1, 3], [10, 15, 10, 12]]), values=[10, 20], opt="zone"
    )
    p = generator.create_long_lat(
        geoBox=geoBox1, zones=np.array([[1, 3, 1, 3], [10, 15, 10, 12]]), values=[10, 20], opt="zone"
    )
    p = generator.create_reduction_file()
    p = generator.create_long_lat(
        geoBox=geoBox1, zones=np.array([[3, 5, 1, 3], [8, 15, 0, 7]]), values=[20, 10], opt="zone"
    )
    p = generator.create_long_lat(
        geoBox=geoBox1, zones=np.array([[1, 3, 1, 3], [10, 15, 10, 12]]), values=[10, 20], opt="zone"
    )
    p = generator.create_pattern_smoothing(value=20, value_2=30)
    p = generator.create_pattern(value=20, pair_impair=1, line_col=1, number=2, allValue=False)
    p = generator.create_reset_cell_file()
    p = generator.create_pattern_sanity_check(value=20)
    p = generator.create_pattern_sanity_check(value=20, mode=2)
    p = generator.create_long_lat(geoBox=geoBox4, zones=np.array([[90, 700, 90, 700]]), values=[10], opt="zone")
    p = generator.create_pattern_smoothing(value=20, value_2=30)
