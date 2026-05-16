#! /usr/bin/env python3
# coding: utf-8

import tempfile

import netCDF4 as nc
import numpy as np
from osgeo import osr

import pyat.common.geo_file as gf
import pyat.dtm.dtm_legacy_constants as dtm_const


class Netcdf3DtmGenerator:
    """Class generator of NetCdf3 file for the processes tests."""

    def initialize_file(self, spatial_reference=gf.SR_WGS_84):
        result = tempfile.mktemp(suffix=".dtm")
        with nc.Dataset(result, "w", format="NETCDF3_CLASSIC") as dataset:
            dataset.mbProj4String = spatial_reference.ExportToProj4()
            dataset.mbEllipsoidName = "WGS-84"
            dataset.mbVersion = 200
            dataset.Sounder_type = 0
            dataset.Number_columns = dataset.Number_lines = 3
            dataset.South_latitude = dataset.West_longitude = 0.5
            dataset.North_latitude = dataset.East_longitude = 3.5
            dataset.createDimension(dtm_const.DIM_LINE, 3)
            dataset.createDimension(dtm_const.DIM_COLUMNS, 3)

            lines = dataset.createVariable(dtm_const.VARIABLE_LINE, "f8", (dtm_const.DIM_LINE,))
            columns = dataset.createVariable(dtm_const.VARIABLE_COLUMN, "f8", (dtm_const.DIM_COLUMNS,))

            transform = osr.CoordinateTransformation(gf.SR_WGS_84, spatial_reference)
            coordinates = transform.TransformPoints([[1.0, 1.0, 0.0], [3.0, 3.0, 0.0]])
            lines[:] = [coordinates[0][1], (coordinates[0][1] + coordinates[1][1]) / 2, coordinates[1][1]]
            columns[:] = [coordinates[0][0], (coordinates[0][0] + coordinates[1][0]) / 2, coordinates[1][0]]

            dataset.Element_y_size = lines[1] - lines[0]
            dataset.Element_x_size = columns[1] - columns[0]

            dataset.Xmin_metric = columns[0] - dataset.Element_x_size / 2
            dataset.Xmax_metric = columns[-1] + dataset.Element_x_size / 2
            dataset.Ymin_metric = lines[0] - dataset.Element_y_size / 2
            dataset.Ymax_metric = lines[-1] + dataset.Element_y_size / 2

            vsoundings = self.createVariable(dataset, dtm_const.VARIABLE_VSOUNDINGS, "i4", 0.0, 1.0, 2147483647, 1, 2)
            vsoundings[0, 1] = vsoundings[2, 1] = 2147483647

            self.createVariable(dataset, dtm_const.VARIABLE_DEPTH, "i2", -4.3, 0.001, 32767, -3.3, -1.6)
            self.createVariable(dataset, dtm_const.VARIABLE_MAX_ACROSS_DISTANCE, "i2", -1, 0.001, 32767, 0, 0.5)
            self.createVariable(dataset, dtm_const.VARIABLE_MIN_ACROSS_DISTANCE, "i2", -1, 0.001, 32767, 0, 0.5)
            self.createVariable(dataset, dtm_const.VARIABLE_MAX_SOUNDING, "i2", -4.3, 0.001, 32767, -3.3, -1.1)
            self.createVariable(dataset, dtm_const.VARIABLE_MIN_SOUNDING, "i2", -4.3, 0.001, 32767, -3.3, -2.1)
            self.createVariable(dataset, dtm_const.VARIABLE_STDEV, "i2", -1, 0.001, 32767, 0.0, 0.5)
            self.createVariable(dataset, dtm_const.VARIABLE_INTERPOLATION_FLAG, "b", 0, 1, 127, 0.0, 1.0)

            # CDIs
            cdis = self.createVariable(dataset, dtm_const.VARIABLE_CDI, "i2", -1, 0.001, 32767, 0, 1)
            cdis[:] = np.full((3, 3), 0.0)
            cdis[1, 1] = 1.0

            _ = dataset.createDimension("nchars", 19)
            _ = dataset.createDimension(dtm_const.DIM_CDI_INDEX_NBR, None)
            cdis_index = dataset.createVariable(
                dtm_const.VARIABLE_CDI_INDEX, "S1", (dtm_const.DIM_CDI_INDEX_NBR, "nchars")
            )
            cdis_index[:] = nc.stringtochar(
                np.array(["SDN:CDI:LOCAL:486_1", "SDN:CDI:LOCAL:486_2"], dtype="S19"), encoding="ascii"
            )

            # history variables
            dataset.createDimension(dtm_const.DIM_MB_HIST_REC_NBR, 1)
            dataset.createDimension(dtm_const.DIM_MB_NAME_LEN, 20)
            dataset.createDimension(dtm_const.DIM_MB_COMMENT_LEN, 256)
            hist_julian_date = dataset.createVariable(
                dtm_const.ATT_MB_HISTORY_DATE, "i4", (dtm_const.DIM_MB_HIST_REC_NBR,)
            )
            hist_time_in_ms = dataset.createVariable(
                dtm_const.ATT_MB_HISTORY_TIME, "i4", (dtm_const.DIM_MB_HIST_REC_NBR,)
            )
            hist_autor = dataset.createVariable(
                dtm_const.ATT_MB_HISTORY_AUTOR, "S1", (dtm_const.DIM_MB_HIST_REC_NBR, dtm_const.DIM_MB_NAME_LEN)
            )
            hist_module = dataset.createVariable(
                dtm_const.ATT_MB_HISTORY_MODULE, "S1", (dtm_const.DIM_MB_HIST_REC_NBR, dtm_const.DIM_MB_NAME_LEN)
            )
            hist_comment = dataset.createVariable(
                dtm_const.ATT_MB_HISTORY_COMMENT, "S1", (dtm_const.DIM_MB_HIST_REC_NBR, dtm_const.DIM_MB_COMMENT_LEN)
            )
            dataset.mbNbrHistoryRec = 1
            hist_julian_date[:] = [2457349]  # 2015-11-22
            hist_time_in_ms[:] = [45000000]  # 12:30:00
            hist_autor[:] = nc.stringtochar(np.array(["unit_test_generator"], dtype="S20"), encoding="ascii")
            hist_module[:] = nc.stringtochar(np.array(["Netcdf3DtmGenerator"], dtype="S20"), encoding="ascii")
            hist_comment[:] = nc.stringtochar(np.array(["Generated for unit test"], dtype="S256"), encoding="ascii")

        return result

    def createVariable(self, dataset, name, dtype, add_offset, scale_factor, missing_value, min_value, max_value):
        variable = dataset.createVariable(name, dtype, (dtm_const.DIM_LINE, dtm_const.DIM_COLUMNS), fill_value=False)
        variable.valid_minimum = (min_value - add_offset) / scale_factor
        variable.valid_maximum = (max_value - add_offset) / scale_factor
        variable.add_offset = add_offset
        variable.scale_factor = scale_factor
        variable.missing_value = missing_value
        variable.set_auto_scale(True)
        variable[:] = np.arange(min_value, max_value, (max_value - min_value) / 9).reshape(3, 3)
        return variable


if __name__ == "__main__":
    generator = Netcdf3DtmGenerator()
    print(generator.initialize_file())
    print(generator.initialize_file(gf.SR_MERCATOR))
