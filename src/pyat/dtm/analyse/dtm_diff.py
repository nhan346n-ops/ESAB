#! /usr/bin/env python3
# coding: utf-8
import os
import tempfile as tmp
from pathlib import Path
from typing import List, Optional

import numpy as np
from osgeo import gdal, gdalconst
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_legacy_constants as Legacy
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.dtm.mask import compute_geo_mask_from_dataset


class DiffMnt:
    def __init__(
        self,
        reference_file: str,
        second_file: str,
        output_dir: str,
        mask: Optional[List[str]] = None,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Initialize a new DiffMnt process
        :param target_file:
        :param reference_file:
        :return:
        """
        self.logger = log.logging.getLogger(DiffMnt.__name__)

        self.reference_file = reference_file
        self.target_file = second_file

        file_prefix = f"{Path(self.reference_file).stem}-{Path(self.target_file).stem}_"
        self.output_file = tmp.mktemp(suffix=".tiff", prefix=file_prefix, dir=output_dir)

        # check if we process old or new dtm format
        extension = Path(self.reference_file).suffix
        self.legacy_format = extension in (".dtm", ".mnt")

        # Specify the layer name to read
        if self.legacy_format:
            self.layer_name = Legacy.VARIABLE_DEPTH
        else:
            self.layer_name = DtmConstants.ELEVATION_NAME

        self.mask_files = arg_util.parse_list_of_files("mask", mask)

        self.monitor = monitor

    def __call__(self):
        self.monitor.set_work_remaining(5)

        # Open netcdf file.nc with gdal
        reference_dataset = gdal.Open(f"NETCDF:{self.reference_file}:{self.layer_name}")

        input_projection = reference_dataset.GetProjection()
        ulx, xres, _, uly, _, yres = reference_dataset.GetGeoTransform()
        lrx = ulx + (reference_dataset.RasterXSize * xres)
        lry = uly + (reference_dataset.RasterYSize * yres)

        secondary_dataset = gdal.Open(f"NETCDF:{self.target_file}:{self.layer_name}")

        # on reprojette dans le même type de grille les données qui nous intéressent
        reprojected_file = tmp.mktemp(suffix=".tiff")
        self.logger.info(
            f"Reprojection {self.target_file} to geotiff {reprojected_file}  with resolution {xres}x{yres} and size {reference_dataset.RasterXSize}x{reference_dataset.RasterYSize}"
        )
        gdal.Warp(
            reprojected_file,
            secondary_dataset,
            dstSRS=input_projection,
            outputBounds=(ulx, lry, lrx, uly),
            xRes=np.abs(xres),
            yRes=np.abs(yres),
        )

        self.monitor.worked(1)

        # close old dataset
        del secondary_dataset

        # puis on ouvre le dataset de travail (en geotiff)
        secondary_dataset = gdal.Open(reprojected_file, gdalconst.GA_ReadOnly)

        # check size
        if reference_dataset.RasterXSize != secondary_dataset.RasterXSize:
            raise ValueError("Dataset does not have the same size")
        if reference_dataset.RasterYSize != secondary_dataset.RasterYSize:
            raise ValueError("Dataset does not have the same size")

        secondary_band = secondary_dataset.GetRasterBand(1)
        nodata = secondary_band.GetNoDataValue()

        secondary_raster = secondary_band.ReadAsArray()
        secondary_raster = np.ma.masked_equal(secondary_raster, nodata)
        if secondary_raster.dtype != np.dtype("f4"):
            secondary_raster = secondary_raster.astype("f4")
            secondary_raster = np.ma.filled(secondary_raster, np.nan)
            secondary_raster *= secondary_band.GetScale()
            secondary_raster += secondary_band.GetOffset()
        self.monitor.worked(1)

        # On masque les differentes valeurs invalides des datasets
        # do we need to take into account for scale_factor ?
        # raster_target = raster_target * scale_factor

        # Read full data from netcdf
        reference_band = reference_dataset.GetRasterBand(1)
        reference_target = reference_dataset.ReadAsArray(
            0, 0, reference_dataset.RasterXSize, reference_dataset.RasterYSize
        )
        nodata = reference_band.GetNoDataValue()
        reference_target = np.ma.masked_equal(reference_target, nodata)
        if reference_target.dtype != np.dtype("f4"):
            reference_target = reference_target.astype("f4")
            reference_target = np.ma.filled(reference_target, np.nan)
            reference_target *= reference_band.GetScale()
            reference_target += reference_band.GetOffset()

        # On calcule la difference
        self.logger.info(f"Computing difference : {self.reference_file} - {reprojected_file}")
        d = np.subtract(reference_target, secondary_raster)
        self.monitor.worked(1)

        # Are we using a KML to restrict the comparison ?
        if self.mask_files:
            mask_int = compute_geo_mask_from_dataset(reference_dataset, self.mask_files)
            # mask_int has a netcdf like origin (left bottom) vs origin of d array (left top)
            mask_int = mask_int[::-1, :]
            # Set to Nan all cells outside the kml zone
            d[mask_int == 0] = np.nan
        self.monitor.worked(1)

        # Create output
        driver = gdal.GetDriverByName("GTiff")

        outRaster = driver.Create(
            self.output_file, reference_dataset.RasterXSize, reference_dataset.RasterYSize, 1, gdal.GDT_Float32
        )
        outRaster.SetGeoTransform(reference_dataset.GetGeoTransform())
        outRaster.SetProjection(reference_dataset.GetProjection())
        outband: gdal.Band = outRaster.GetRasterBand(1)
        outband.SetNoDataValue(np.nan)
        outband.WriteArray(d)
        outband.FlushCache()

        absolute_diff = np.abs(d)

        # Close datasets
        del reference_dataset
        del secondary_dataset
        os.remove(reprojected_file)
        self.logger.info(f"Delete temporary file {reprojected_file}")
        with np.errstate(under="ignore"):
            # ignore numpy's underflow error that might appear
            self.logger.info(
                f"Difference layer statistics : std {np.nanstd(absolute_diff)} , max {np.nanmax(absolute_diff)}"
            )
        self.logger.info(f"Difference geotiff file created : {self.output_file} ")
        self.monitor.done()
