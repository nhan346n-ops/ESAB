#! /usr/bin/env python3
# coding: utf-8

import datetime
import os
import shutil
import tempfile

import numpy as np
from osgeo import gdal, gdalconst
from pygws.service.progress_monitor import DefaultMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log


class GDALGapFillingProcess:
    """Gap filling class with bilinear interpolation."""

    def __init__(
        self,
        i_paths: list,
        o_paths: list = None,
        suffix="_fillnodata",
        overwrite=False,
        mask_size: str = "3",
        smooth_iterations: str = "0",
        monitor=DefaultMonitor,
    ):
        """Constructor.

        Arguments:
            i_paths {list} -- Input file list (.nc).
            o_paths {list} -- Optional output file list (.nc). (default: {None})
            suffix {str} -- Suffix of generated output path. Used when o_paths is empty. (default: {_fillnodata})
            overwrite {bool} -- true to overwrite output file if exists. (default: {False})
            monitor -- Progress monitor. (default is a silent monitor: {DefaultMonitor})

        Raises:
            ValueError: wrong value of mask_size or smooth_iterations.
        """

        self.i_paths = i_paths
        self.o_paths = o_paths
        self.suffix = suffix
        self.overwrite = overwrite
        self.mask_size = arg_util.parse_int("mask_size", mask_size, 3)
        self.smooth_iterations = arg_util.parse_int("smooth_iterations", smooth_iterations)
        self.monitor = monitor

        self.logger = log.logging.getLogger(self.__class__.__name__)

        self.logger.debug(f"Set mask_size to {self.mask_size}.")
        self.logger.debug(f"Set smooth_iterations to {self.smooth_iterations}.")

        # Check size of the mask
        if self.mask_size <= 0:
            raise ValueError("The size of the mask must be positif.")

        if self.smooth_iterations < 0:
            raise ValueError("smooth_iterations must be positif.")

    def run(self) -> None:
        """Super method run with gap filling process data."""
        begin = datetime.datetime.now()
        for ind, i_path in enumerate(self.i_paths):
            try:
                o_path = self.o_paths[ind]
                # we copy the source file to the destination path
                if not os.path.exists(o_path) or self.overwrite:
                    shutil.copy(i_path, o_path)
                else:
                    raise FileExistsError(
                        "File already exists and overwrite not allowed (allow overwrite with option : '-o --overwrite)"
                    )
                filler = fill_nodata(o_path, max_distance=self.mask_size, smooth_iterations=0)
                filler.run()
            except Exception as e:
                self.logger.error(f"{type(e).__name__}: {e}")


class fill_nodata:
    # Execute a gdal_fillnodata command on a given file. The file is modified
    def __init__(self, filename, max_distance=3, smooth_iterations=0):
        self.file = filename
        self.max_distance = max_distance
        self.smooth_iterations = smooth_iterations

    def CopyBand(self, srcband, dstband):
        for line in range(srcband.YSize):
            line_data = srcband.ReadRaster(0, line, srcband.XSize, 1)
            dstband.WriteRaster(0, line, srcband.XSize, 1, line_data, buf_type=srcband.DataType)

    def run(self):
        print(
            "gdal_fill_nodata for file ",
            self.file,
            " max_distance = ",
            self.max_distance,
            ",smooth_iteration =",
            self.smooth_iterations,
        )
        input_dataset = gdal.Open(f"NETCDF:{self.file}:{DtmConstants.ELEVATION_NAME}")

        # gdal netcdf driver does not allow for update in dataset, so we create a temporary tiff to
        # store data and use netcdf driver to update the input file
        tmp_tiff_file = tempfile.mktemp(suffix=".tif")

        drv = gdal.GetDriverByName("GTiff")
        dst_ds = drv.Create(
            tmp_tiff_file,
            input_dataset.RasterXSize,
            input_dataset.RasterYSize,
            1,
            gdalconst.GDT_Float32,
        )
        wkt = input_dataset.GetProjection()
        if wkt != "":
            dst_ds.SetProjection(wkt)
        dst_ds.SetGeoTransform(input_dataset.GetGeoTransform())

        temp_band = dst_ds.GetRasterBand(1)
        srcband = input_dataset.GetRasterBand(1)
        self.CopyBand(srcband, temp_band)

        ndv = srcband.GetNoDataValue()
        if ndv is not None:
            temp_band.SetNoDataValue(ndv)

        # call gdal fill nodata
        gdal.FillNodata(
            temp_band,
            srcband.GetMaskBand(),
            self.max_distance,
            self.smooth_iterations,
            [],
            callback=None,
        )
        # close input dataset
        input_dataset = None

        # update netcdf file with the results
        with dtm_driver.open_dtm(self.file, mode="r+") as o_dtm_driver:
            # first create if not existing the interpolation layer
            o_dtm_driver.create_interpolation_layer()

            elevation = o_dtm_driver[DtmConstants.ELEVATION_NAME]
            elevation_old_values = elevation[:]
            # TODO read line by line to prevent consuming too much memory
            filled_data = temp_band.ReadAsArray()
            # line are reverserd between geotiff and netcdf
            elevation[:] = filled_data[::-1, :]

            # compute modified values flag, value is marked as modified if not nan and nan in origin data
            flag = ~np.isnan(elevation) & np.isnan(elevation_old_values[:])
            interpolation = o_dtm_driver[DtmConstants.INTERPOLATION_FLAG]
            interpolation_values = interpolation[:]
            interpolation_values[flag] = 1
            interpolation[:] = interpolation_values

        # close all
        temp_band = None
        dst_ds = None
        maskband = None
