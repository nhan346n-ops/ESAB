#! /usr/bin/env python3
# coding: utf-8

import tempfile as tmp

import numpy as np
import scipy.stats.stats as stats
from osgeo import gdal, osr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.numba.merge_functions as nb
import pyat.utils.argument_utils as arg_util
from pyat.dtm.merge.abstract_merge import AbstractMergeProcess
from pyat.utils.exceptions.exception_list import BadParameter
from pyat.utils.gdal_utils import TemporaryDataset, gdal_to_netcdf


class SlopeMerge(AbstractMergeProcess):
    def _check_input_parameters(self):

        # AbstractMergeProcess expect a list as input files
        if isinstance(self.i_paths, str):
            self.i_paths = [self.i_paths]

        if len(self.i_paths) == 0:
            raise BadParameter("One reference file requested")
        if len(self.i_paths) > 1:
            raise BadParameter("Only one reference file is supported")

    def __init__(
        self,
        i_paths: list,
        second_file: str,
        coord: dict,
        o_path: str = None,
        overwrite=False,
        mask: str = None,
        min_slope: str = "1.0",
        max_slope: str = "5.5",
        allow_undefined_cdi: bool = False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Initialize a new DiffMnt process
        :param target_file:
        :param reference_file:
        :return:
        """
        super().__init__(
            process_name="slope_merge",
            i_paths=i_paths,
            o_path=o_path,
            overwrite=overwrite,
            coord=coord,
            mask=mask,
            allow_undefined_cdi=allow_undefined_cdi,
            monitor=monitor,
        )

        self.second_file_path = second_file
        self.min_slope = arg_util.parse_float("min_slope", min_slope, 1.0)
        self.max_slope = arg_util.parse_float("max_slope", max_slope, 5.5)

        self.temp_dir = None
        self.slope_array = None
        self.reference_file_path = self.i_paths[0]
        self.second_driver = dtm_driver.DtmDriver(self.second_file_path)

    def process_global_data(self, mask):
        # The two considered files
        self.reference_driver = self.i_drivers[0]
        self.second_driver.open()

        # now start to merge
        slope_array = self.compute_slope()
        slope_array[slope_array == -9999] = np.nan  # invalid values are identified as -9999 by the algorithm

        desc = stats.describe(slope_array, axis=None, nan_policy="omit")
        self.logger.info(
            f"Slope computed : min {desc.minmax[0]}, max {desc.minmax[1]}, mean {desc.mean}, stdev {np.sqrt(desc.variance)}"
        )

        # reproject slope on destination grid

        # create an array based on slope
        self.logger.info(f"Create decision tree array ")

        # reproject slope data
        # output file lat long variables
        o_y = self.o_driver.get_y_axis()[:].data
        o_x = self.o_driver.get_x_axis()[:].data

        i_y = self.reference_driver.get_y_axis()[:].data
        i_x = self.reference_driver.get_x_axis()[:].data

        self.slope_array = nb.merge_project(i_y, o_y, i_x, o_x, slope_array, np.nan, mask)

        # now source is filled with 0,1,2 : 0 copy source data, merge data, 2 copy second file data
        # generic algorithm is value = value_ref if slope<min_slope, value = value_second_file is slope > max_slope else it is a interpolation between both

    def _process_layer(self, layer_name: str, geo_mask: np.ndarray, smoothing_mask: np.ndarray = None) -> None:
        """For each file, project the layer in first. Then process it.

        Arguments:
            name {str} -- Name of the layer.
        """
        # do nothing if layer is not in reference file layer

        if layer_name not in self.reference_driver:
            return

        # create output layer
        self.o_driver.add_layer(layer_name)

        # Initialisation
        temp_buffer = self.o_driver[layer_name][:].data

        # output file lat long variables
        o_y = self.o_driver.get_y_axis()[:].data
        o_x = self.o_driver.get_x_axis()[:].data

        missing_value = dtm_driver.get_missing_value(layer_name)

        # reproject reference file (only a shift in coordinates).
        i_y = self.reference_driver.get_y_axis()[:].data
        i_x = self.reference_driver.get_x_axis()[:].data
        input_reference_data = self.reference_driver[layer_name][:].data
        input_reference_data = nb.merge_project(i_y, o_y, i_x, o_x, input_reference_data, missing_value, geo_mask)

        if layer_name not in self.second_driver:
            # we need to create a temporary buffer with default values for this variables
            # At least we expect to have one elevation layer
            i_elevation = self.second_driver[DtmConstants.ELEVATION_NAME]
            i_elevation_data = i_elevation[:].data
            # we use elevation data as a mask for valid/invalid values
            second_file_data = np.empty(shape=i_elevation_data.shape, dtype=dtm_driver.LAYER_TYPES[layer_name])
            second_file_data.fill(missing_value)
            self.o_driver.fill_default_layer_buffer(
                layer_name, second_file_data, i_elevation_data, i_elevation._FillValue
            )
        else:
            second_file_data = self.second_driver[layer_name][:].data

        # Project points.
        i_y = self.second_driver.get_y_axis()[:].data
        i_x = self.second_driver.get_x_axis()[:].data
        second_file_data = nb.merge_project(i_y, o_y, i_x, o_x, second_file_data, missing_value, geo_mask)

        # now we got two layer reprojected to the final grid

        if layer_name in [DtmConstants.ELEVATION_NAME, DtmConstants.ELEVATION_SMOOTHED_NAME]:
            temp_buffer = nb.merge_operation_with_slope(
                temp_buffer,
                input_reference_data,
                second_file_data,
                self.slope_array,
                min_slope_value=self.min_slope,
                max_slope_value=self.max_slope,
                invalid_value=missing_value,
                operation=nb.SlopeOperation.INTERPOLATION.value,
            )

        elif layer_name == DtmConstants.ELEVATION_MAX:
            temp_buffer = nb.merge_operation_with_slope(
                temp_buffer,
                input_reference_data,
                second_file_data,
                self.slope_array,
                min_slope_value=self.min_slope,
                max_slope_value=self.max_slope,
                invalid_value=missing_value,
                operation=nb.SlopeOperation.MAX.value,
            )

        elif layer_name == DtmConstants.ELEVATION_MIN:
            temp_buffer = nb.merge_operation_with_slope(
                temp_buffer,
                input_reference_data,
                second_file_data,
                self.slope_array,
                min_slope_value=self.min_slope,
                max_slope_value=self.max_slope,
                invalid_value=missing_value,
                operation=nb.SlopeOperation.MIN.value,
            )  #
        elif layer_name == DtmConstants.STDEV:
            # we chose to take max stdev, given the use case for this merge_with_slope it is highly probable that they will be equals in each files
            temp_buffer = nb.merge_operation_with_slope(
                temp_buffer,
                input_reference_data,
                second_file_data,
                self.slope_array,
                min_slope_value=self.min_slope,
                max_slope_value=self.max_slope,
                invalid_value=missing_value,
                operation=nb.SlopeOperation.MAX.value,
            )
        elif layer_name == DtmConstants.VALUE_COUNT:
            temp_buffer = nb.merge_operation_with_slope(
                temp_buffer,
                input_reference_data,
                second_file_data,
                self.slope_array,
                min_slope_value=self.min_slope,
                max_slope_value=self.max_slope,
                invalid_value=missing_value,
                operation=nb.SlopeOperation.SUM.value,
            )  #
        elif layer_name == DtmConstants.INTERPOLATION_FLAG:
            temp_buffer = nb.merge_operation_with_slope(
                temp_buffer,
                input_reference_data,
                second_file_data,
                self.slope_array,
                min_slope_value=self.min_slope,
                max_slope_value=self.max_slope,
                invalid_value=missing_value,
                operation=nb.SlopeOperation.MAX.value,
            )

        self.o_driver[layer_name][:] = temp_buffer

    def _process_cdis(self, mask: np.array) -> None:
        """Merge cdi. Project layer then process it."""
        cdi_layer = DtmConstants.CDI
        cdi_index = DtmConstants.CDI_INDEX
        self.o_driver.add_layer(DtmConstants.CDI_INDEX)
        self.o_driver.add_layer(DtmConstants.CDI)

        # Initialisation
        reference_driver = self.i_drivers[0]
        second_driver = self.second_driver

        # first concat CDI ids
        if cdi_layer in reference_driver:
            reference_cdi = reference_driver[cdi_layer][:]
        else:
            reference_cdi = []
        reference_cdi = cdi_util.trim_string_array(reference_cdi)
        if cdi_layer in second_driver:
            second_cdi = second_driver[cdi_layer][:]
        else:
            second_cdi = []
        second_cdi = cdi_util.trim_string_array(second_cdi)

        cdi_util.reset_all_cdi_id(self.o_driver.dataset)
        second_file_cdi_index_offset = 0
        for i, name in enumerate(reference_cdi):
            # VLEN can be only accessed one at a time
            self.o_driver[cdi_layer][i] = name
            second_file_cdi_index_offset = i + 1

        for i, name in enumerate(second_cdi):
            # VLEN can be only accessed one at a time
            self.o_driver[cdi_layer][i + second_file_cdi_index_offset] = name

        missing_value = dtm_driver.get_missing_value(cdi_index)
        # now set cdi index

        # reproject reference file (only a shift in coordinates).
        o_y = self.o_driver.get_y_axis()[:].data
        o_x = self.o_driver.get_x_axis()[:].data

        i_y = reference_driver.get_y_axis()[:].data
        i_x = reference_driver.get_x_axis()[:].data
        if cdi_index in reference_driver:
            reference_cdi_index = reference_driver[cdi_index][:].data
        else:
            reference_cdi_index = reference_driver.prepare_data(DtmConstants.CDI_INDEX)

        input_reference_data_reprojected = nb.merge_project(
            i_y, o_y, i_x, o_x, reference_cdi_index, missing_value, mask
        )
        i_y = second_driver.get_y_axis()[:].data
        i_x = second_driver.get_x_axis()[:].data
        if cdi_index in second_driver:
            second_cdi_index = second_driver[cdi_index][:].data
        else:
            second_cdi_index = second_driver.prepare_data(DtmConstants.CDI_INDEX)

        input_second_file_data_reprojected = nb.merge_project(i_y, o_y, i_x, o_x, second_cdi_index, missing_value, mask)
        input_second_file_data_reprojected[
            input_second_file_data_reprojected != missing_value
        ] += second_file_cdi_index_offset
        temp_buffer = np.full(
            shape=(len(self.o_driver.get_y_axis()), len(self.o_driver.get_x_axis())),
            fill_value=missing_value,
        )
        temp_values = nb.merge_operation_with_slope(
            temp_buffer,
            input_reference_data_reprojected,
            input_second_file_data_reprojected,
            self.slope_array,
            min_slope_value=self.min_slope,
            max_slope_value=self.max_slope,
            invalid_value=missing_value,
            operation=nb.SlopeOperation.DOMINANT.value,
        )

        self.o_driver[cdi_index][:] = temp_values
        cdi_util.clean_cdi(self.o_driver.dataset)

    def compute_slope(self):
        self.logger.info(f"Computing Slope {self.reference_file_path} dataset")

        gdal.UseExceptions()
        # Create the path of the output sub dataset and open input sub dataset.
        reference_dataset = gdal.Open(f"NETCDF:{self.reference_file_path}:{DtmConstants.ELEVATION_NAME}")
        self.logger.info(f"Opening {self.reference_file_path} dataset")
        wkt = reference_dataset.GetProjection()
        inSRS_converter = osr.SpatialReference()  # makes an empty spatial ref object
        inSRS_converter.ImportFromWkt(wkt)  # populates the spatial ref object with our WKT SRS
        ulx, xres, xskew, uly, yskew, yres = reference_dataset.GetGeoTransform()
        lrx = ulx + (reference_dataset.RasterXSize * xres)
        lry = uly + (reference_dataset.RasterYSize * yres)
        centerx = (ulx + lrx) / 2
        centery = (uly + lry) / 2

        # Project the reference file when spatial reference is geographic
        projected_dataset = None
        self.logger.info(f"Compute slope to in metric coordinates")
        slope_merc = tmp.mktemp(suffix="_slope_merc.tiff", dir=self.temp_dir)
        source_dataset = reference_dataset
        if inSRS_converter.IsGeographic():
            proj_string = f"+proj=merc +lat_ts={centery} +lon_0={centerx} +ellps=WGS84"
            mercator_file = tmp.mktemp(suffix="_elevation_merc.tiff", dir=self.temp_dir)
            self.logger.info(f"Warping {self.reference_file_path} to a mercator projection ({proj_string}")
            source_dataset = gdal.Warp(mercator_file, reference_dataset, dstSRS=proj_string)
            projected_dataset = TemporaryDataset(source_dataset, mercator_file)

        # reproject in UTM or mercator with gdal. then call slope processing
        slope_mercator_dataset = gdal.DEMProcessing(
            destName=slope_merc,
            srcDS=source_dataset,
            processing="slope",
            computeEdges=True,
        )  # compute slope

        # Closing file
        source_dataset = None

        # create auto erasable wrapper
        slope_mercator_dataset = TemporaryDataset(slope_mercator_dataset, slope_merc)
        # now go back to reference grid
        slope_reference = tmp.mktemp(suffix="_slope.tiff", dir=self.temp_dir)
        self.logger.info(f"Reproject dataset to reference projection")
        slope_dataset = gdal.Warp(
            slope_reference,
            slope_mercator_dataset.dataset,
            dstSRS=wkt,
            outputBounds=[ulx, lry, lrx, uly],
            xRes=xres,
            yRes=-yres,
        )
        # create auto erasable wrapper
        slope_dataset = TemporaryDataset(slope_dataset, slope_reference)
        slope_array = gdal_to_netcdf(slope_dataset.dataset)

        self.logger.debug(f"done computing slope in : {slope_reference}")
        return slope_array

    def _close(self):
        if self.second_driver is not None:
            self.second_driver.close()
