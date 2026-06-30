#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile as tmp
from typing import Optional, Tuple

import numpy as np
import osgeo.gdal as gdal
from osgeo import osr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.gdal_utils as gdal_util
import pyat.utils.pyat_logger as log
from pyat.dtm.dtm_driver import DtmDriver, DtmFile, get_missing_value
from pyat.utils import nc_encoding


class ReprojectProcess:
    """Reproject process class. The function is based on the warp function of gdal."""

    def __init__(
        self,
        i_paths: list,
        coord: dict,
        o_paths: list = None,
        suffix="-reprojected",
        overwrite: bool = False,
        target_spatial_reference: Optional[str] = None,
        target_resolution: Optional[float] = None,
        default_algorithm: str = "bilinear",
        cdi_algorithm: str = "mode",
        stdev_algorithm: str = "near",
        value_count: str = "near",
        interpolation_flag: str = "near",
        monitor=DefaultMonitor,
    ):
        """Constructor.

        Arguments:
            i_paths {list} -- Input file list (.nc).
            coord {dict} -- Coordinates of geographic bounds
            o_paths {list} -- Optional output file list (.nc). (default: {None})
            suffix {str} -- Suffix of generated output path. Used when o_paths is empty. (default: {-reprojected})
            overwrite {bool} -- true to overwrite output file if exists. (default: {False})
            target_spatial_reference {str} -- spatial reference of the output file. (default: same as input file)
            target_resolution {float} -- output file resolution. (default: same as input file)
            default_algorithm {str} -- default algorithm for all numeric layers. (default: {bilinear})
            cdi_algorithm {str} -- algorithm for cdi layer. (default: {mode})
            stdev_algorithm {str} -- algorithm for stdev layer. (default: {near})
            value_count {str} -- algorithm for value_count layer. (default: {near})
            interpolation_flag {str} -- algorithm for interpolation_flag layer. (default: {near})
            monitor {list} -- Progress monitor. (default is a silent monitor: {DefaultMonitor})
        """
        self.i_paths = i_paths
        self.custom_geobox = arg_util.parse_geobox("coord", coord)
        self.o_paths = o_paths
        self.suffix = suffix
        self.overwrite = overwrite

        self.target_spatial_reference = None
        if not target_spatial_reference is None:
            self.target_spatial_reference = osr.SpatialReference()
            self.target_spatial_reference.ImportFromProj4(target_spatial_reference)

        self.target_resolution = (
            arg_util.parse_float("target_resolution", target_resolution) if not target_resolution is None else None
        )
        self.default_algorithm = gdal_util.translate_algorithm("default algorithm", default_algorithm)
        self.cdi_algorithm = gdal_util.translate_algorithm("cdi_algorithm", cdi_algorithm)
        self.stdev_algorithm = gdal_util.translate_algorithm("stdev_algorithm", stdev_algorithm)
        self.value_count = gdal_util.translate_algorithm("value_count", value_count)
        self.interpolation_flag = gdal_util.translate_algorithm("interpolation_flag", interpolation_flag)
        self.monitor = monitor

        self.logger = log.logging.getLogger(self.__class__.__name__)

    def _parse_spatial_resolution(self, i_file: DtmFile) -> Optional[Tuple[float, float]]:
        result = None
        if not self.target_resolution is None:
            result = (self.target_resolution, self.target_resolution)
        elif i_file.spatial_reference.IsSame(self.target_spatial_reference):
            result = (i_file.spatial_resolution_x, i_file.spatial_resolution_y)
            self.logger.info(f"Using spatial resolution {result}")
        return result

    def process_data(self, i_driver: DtmDriver, o_driver: DtmDriver, monitor: ProgressMonitor) -> None:
        """Create variable, then process it.

        Arguments:
            ind {int} -- Number of the processed file.
        """
        i_file = i_driver.dataset
        if self.target_spatial_reference is None:
            self.target_spatial_reference = i_driver.dtm_file.spatial_reference

        spatial_res = self._parse_spatial_resolution(i_driver.dtm_file)
        # create a netcdf dataset interpolating elevation, this allow to copy dimensions informations
        filename = os.path.basename(nc_encoding.filepath(i_file))
        elevation_gdal_dataset, elevation_file_path = self.__create_ds(
            DtmConstants.ELEVATION_NAME,
            nc_encoding.filepath(i_file),
            filename + "_" + DtmConstants.ELEVATION_NAME,
            spatial_res,
        )
        # Creates dimensions and grid mapping in the output file
        o_driver.dtm_file.initialize_with_gdal_dataset(elevation_gdal_dataset)
        o_driver.initialize_file()

        # Used for the log
        count = 0
        n = len(i_file.variables)
        monitor.set_work_remaining(n)

        # create elevation layer
        data = np.array(elevation_gdal_dataset.ReadAsArray(), dtype=dtm_driver.get_type(DtmConstants.ELEVATION_NAME))
        data = data[::-1]
        o_driver.add_layer(layer_name=DtmConstants.ELEVATION_NAME, data=data)

        # close and delete temps file
        elevation_gdal_dataset = None
        self.__delete_ds(elevation_file_path)

        count += 1
        log.info_progress_layer(self.logger, "project", DtmConstants.ELEVATION_NAME, count, n)
        monitor.worked(1)
        ignored_variables = {
            DtmConstants.LAT_NAME,
            DtmConstants.LON_NAME,
            DtmConstants.ABSCISSA_NAME,
            DtmConstants.ORDINATE_NAME,
            DtmConstants.LON_NAME,
            DtmConstants.CRS_NAME,
            DtmConstants.ELEVATION_NAME,
        }
        # parse all variables in input file and create their projection
        filename = os.path.basename(nc_encoding.filepath(i_file))

        for name, variable in i_file.variables.items():
            if not name in ignored_variables:
                if name in DtmConstants.LAYERS:
                    count += 1
                    log.info_progress_layer(self.logger, "project", name, count, n)

                    # Create variable in the o_file.
                    dataset, temp_file = self.__create_ds(
                        name, nc_encoding.filepath(i_file), filename + "_" + DtmConstants.ELEVATION_NAME, spatial_res
                    )
                    # create elevation layer since we already computed its data
                    data = np.array(dataset.ReadAsArray(), dtype=dtm_driver.get_type(name))
                    data = data[::-1]
                    o_driver.add_layer(layer_name=name, data=data)

                    # close and delete temps file
                    dataset = None
                    self.__delete_ds(temp_file)

                elif name == DtmConstants.CDI:
                    # Copy cdi_ref
                    count += 1
                    log.info_progress_layer(self.logger, "layer", name, count, n)
                    o_driver.create_cdi_reference_variable(cdi_util.trim_string_array(variable[:]))

                monitor.worked(1)

        # Clean CDI
        if DtmConstants.CDI_INDEX in o_driver and DtmConstants.ELEVATION_NAME in o_driver:
            self.logger.info("Cleaning CDI...")
            elevation_mask = o_driver[DtmConstants.ELEVATION_NAME][:].mask
            cdi_index = o_driver[DtmConstants.CDI_INDEX][:]
            o_driver[DtmConstants.CDI_INDEX][:] = np.where(
                elevation_mask, get_missing_value(DtmConstants.CDI_INDEX), cdi_index
            )

    def __create_ds(
        self,
        layer: str,
        input_file,
        output_file_template,
        spatial_res: Optional[Tuple[float, float]],
    ) -> Tuple[gdal.Dataset, str]:
        """Get the projected sub dataset with gdal. Process with warp function.
        Returns:
            gdal.Dataset -- Dataset projected.
        """
        # Stop GDAL printing both warnings and errors to STDERR
        # gdal.PushErrorHandler("CPLQuietErrorHandler")
        # Make GDAL raise python exceptions for errors (warnings won't raise an exception)
        gdal.UseExceptions()

        # Create the path of the output sub dataset and open input sub dataset.
        o_path = tmp.mktemp(prefix=output_file_template, suffix=".tiff")
        src_path = f"NETCDF:{input_file}:{layer}"

        # Process
        dict_opt = {}

        missing_value = dtm_driver.get_missing_value(layer)
        dict_opt["srcNodata"] = missing_value
        dict_opt["dstNodata"] = missing_value
        gdal_type_map = {np.int32: gdal.GDT_Int32, np.int8: gdal.GDT_Byte, np.float32: gdal.GDT_Float32}

        output_type = gdal_type_map[dtm_driver.get_type(layer)]
        dict_opt["outputType"] = output_type

        algorithm = self.default_algorithm
        if layer == DtmConstants.CDI_INDEX:
            algorithm = self.cdi_algorithm
        elif layer == DtmConstants.VALUE_COUNT:
            algorithm = self.value_count
        elif layer == DtmConstants.STDEV:
            algorithm = self.stdev_algorithm
        elif layer == DtmConstants.INTERPOLATION_FLAG:
            algorithm = self.interpolation_flag
        # 0: gdal.GRIORA_NearestNeighbour
        # 1: gdal.GRIORA_Bilinear
        # 2: gdal.GRIORA_Cubic
        # 3: gdal.GRIORA_CubicSpline
        # 4: gdal.GRIORA_Lanczos
        # 5: gdal.GRIORA_Average
        # 6: gdal.GRIORA_Mode
        # 7: gdal.GRIORA_Gauss
        dict_opt["resampleAlg"] = algorithm

        # Geobox (minX, minY, maxX, maxY)
        dict_opt["outputBounds"] = [
            self.custom_geobox.left,
            self.custom_geobox.lower,
            self.custom_geobox.right,
            self.custom_geobox.upper,
        ]

        # target spatial reference
        if not self.target_spatial_reference is None:
            dict_opt["dstSRS"] = self.target_spatial_reference.ExportToProj4()

        # spatial resolution
        if not spatial_res is None:
            dict_opt["xRes"] = spatial_res[0]
            dict_opt["yRes"] = spatial_res[1]

        # If TypeError: in method 'wrapper_GDALWarpDestName', argument 4 of type 'GDALWarpAppOptions *'
        # Add GDAL_DATA=path to gdal.py in anaconda to variable environnement
        # i.e GDAL_DATA : C:\Users\gguardia\AppData\Local\Continuum\anaconda3\envs\pyATDev\Lib\site-packages\osgeo\gdal.py
        # Error of anaconda environnement
        ds = gdal.Warp(o_path, src_path, options=gdal.WarpOptions(**dict_opt))

        return ds, o_path

    def __delete_ds(self, output_nc_file: str):
        """close dataset, reference should be set to null after this function"""
        gdal.Unlink(output_nc_file)

    def __call__(self) -> None:
        process_util.process_each_input_dtm_to_output_dtm(
            self.__class__.__name__,
            self.i_paths,
            self.process_data,
            self.logger,
            self.o_paths,
            self.suffix,
            self.overwrite,
            self.monitor,
        )
