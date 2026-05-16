#! /usr/bin/env python3
# coding: utf-8

from os import PathLike
from typing import List, NamedTuple

import numba
import numpy as np
import osgeo.gdal as gdal
from numpy.typing import ArrayLike
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from scipy.ndimage import generic_filter

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.pyat_logger as log
from pyat.utils import gdal_utils

PROCESS_NAME = "quality_indicator"
__logger = log.logging.getLogger(PROCESS_NAME)


class Layers(NamedTuple):
    elevations: ArrayLike
    filtered_count: ArrayLike
    value_count: ArrayLike
    max_across_distance: ArrayLike
    max_across_angle: ArrayLike
    stdev: ArrayLike
    interpolation_flag: ArrayLike


class Flags(NamedTuple):
    interpolation: ArrayLike
    sufficient_nb_sound: ArrayLike
    compared_stdev: ArrayLike
    rate_of_invalidated_sounds: ArrayLike
    angle_of_incidence: ArrayLike


no_data_value = 255


class QualityIndicatorArgs(NamedTuple):
    """
    Class representing all arguments for configuring the process
    See pyat/app/emodnet/conf/compute_quality_indicator.json for more details
    """

    i_paths: List[PathLike]
    o_paths: List[PathLike] | None = None
    overwrite: bool = False
    across_angle_max_threshold: float = 65.0
    minimum_detection_count: int = 5
    invalid_rate_max_threshold: float = 1.5
    monitor: ProgressMonitor = DefaultMonitor


def computes(**kwargs) -> None:
    """
    Function accepting all arguments of the process as a dict. Possible arguments are listed in "QualityIndicatorArgs" class
    """
    computes_with_QualityIndicatorArgs(QualityIndicatorArgs(**kwargs))


def computes_with_QualityIndicatorArgs(args: QualityIndicatorArgs) -> None:
    """
    Main function

    Calculation of statistical data on the different layers of one or more DTMs.

    returns the resulting statistics are grouped in a AllMetrics instance
    """

    # pylint:disable=unused-argument
    def __process_one_dtm(i_path: PathLike, o_path: PathLike, sub_monitor):
        with dtm_driver.open_dtm(i_path) as i_driver:
            layers = load_layers(i_driver)
            flags = computes_all_flags(layers, args)
            indicators = np.zeros_like(layers.elevations.data)
            for one_layer_flags in flags:
                if one_layer_flags is not None:
                    indicators = indicators + one_layer_flags

            # Masking cells without elevation
            indicators = np.where(layers.elevations.mask, 255, indicators)

            write_tif(i_driver.dtm_file, indicators, o_path, args)

    # Launch
    process_util.process_each_input_file_to_output_file(
        PROCESS_NAME,
        i_paths=args.i_paths,
        process_data_func=__process_one_dtm,
        logger=__logger,
        o_paths=args.o_paths,
        suffix="_" + PROCESS_NAME,
        extension=".tif",
        overwrite=args.overwrite,
        monitor=args.monitor,
    )


def computes_all_flags(layers: Layers, args: QualityIndicatorArgs) -> Flags:
    angle_of_incidence = computes_max_across_angle(layers, args)
    sufficient_nb_sound = computes_sufficient_detection_count(layers, args)
    rate_of_invalidated_sounds = computes_rate_of_invalidated_sounds(layers, args)
    compared_stdev = computes_compared_stdev(layers)
    interpolation = computes_interpolation(layers)
    return Flags(interpolation, sufficient_nb_sound, compared_stdev, rate_of_invalidated_sounds, angle_of_incidence)


def load_layers(i_driver: dtm_driver.DtmDriver) -> Layers:
    """Load layer from DTM"""
    elevations = i_driver[DtmConstants.ELEVATION_NAME][:]
    filtered_count = None
    if DtmConstants.FILTERED_COUNT in i_driver:
        filtered_count = i_driver[DtmConstants.FILTERED_COUNT][:]
        filtered_count = np.ma.filled(filtered_count, 0)
    value_count = i_driver[DtmConstants.VALUE_COUNT][:] if DtmConstants.VALUE_COUNT in i_driver else None
    max_across_distance = (
        i_driver[DtmConstants.MAX_ACROSS_DISTANCE][:] if DtmConstants.MAX_ACROSS_DISTANCE in i_driver else None
    )
    max_across_angle = (
        i_driver[DtmConstants.MAX_ACCROSS_ANGLE][:] if DtmConstants.MAX_ACCROSS_ANGLE in i_driver else None
    )
    stdev = i_driver[DtmConstants.STDEV][:] if DtmConstants.STDEV in i_driver else None
    interpolation_flag = (
        i_driver[DtmConstants.INTERPOLATION_FLAG][:] if DtmConstants.INTERPOLATION_FLAG in i_driver else None
    )

    return Layers(
        elevations, filtered_count, value_count, max_across_distance, max_across_angle, stdev, interpolation_flag
    )


def computes_max_across_angle(layers: Layers, args: QualityIndicatorArgs) -> ArrayLike | None:
    """Computes flag Angle of incidence of sounds in cells"""
    if layers.max_across_angle is not None:
        __logger.info(f"Check max across angle (maximum: {args.across_angle_max_threshold} degrees)...")
        return np.where(np.abs(layers.max_across_angle) > args.across_angle_max_threshold, 1, 0)
    if layers.max_across_distance is not None:
        __logger.info(f"Check max across angle (maximum: {args.across_angle_max_threshold} degrees)...")
        max_across_angle = np.abs(np.arctan(layers.max_across_distance / layers.elevations) * 180.0 / np.pi)
        return np.where(max_across_angle > args.across_angle_max_threshold, 0b00000001, 0)
    else:
        __logger.info(f"Check density : not possible, max_across_angle or max_across_distance layers not available.")
        return None


def computes_sufficient_detection_count(layers: Layers, args: QualityIndicatorArgs) -> ArrayLike | None:
    """Computes flag Sufficient number of valid sound"""
    if layers.value_count is not None and layers.filtered_count is not None:
        __logger.info(f"Check density (minimun detection count : {args.minimum_detection_count})...")
        return np.where(layers.value_count < args.minimum_detection_count, 0b00001000, 0)
    else:
        __logger.info(f"Check density : not possible, value_count or filtered_count layers not available.")
        return None


def computes_interpolation(layers: Layers) -> ArrayLike | None:
    """Computes flag interpolation"""
    if layers.interpolation_flag is not None:
        __logger.info(f"Check interpolation...")
        return np.where(layers.interpolation_flag.data == 1, 0b00010000, 0)
    else:
        __logger.info(f"Check interpolation : not possible, interpolation_flag layer not available.")
        return None


def computes_rate_of_invalidated_sounds(layers: Layers, args: QualityIndicatorArgs) -> ArrayLike | None:
    """Computes flag Rate of invalidated sounds"""
    __logger.info("Check rate of invalid detection...")
    if layers.value_count is not None and layers.filtered_count is not None:
        mean_rate = np.nanmean(layers.filtered_count / (layers.filtered_count + layers.value_count))
        cell_rate = layers.filtered_count / (layers.filtered_count + layers.value_count)
        return np.where(cell_rate / mean_rate > args.invalid_rate_max_threshold, 0b00000010, 0)
    else:
        __logger.info(f"Check invalid detection : not possible, value_count or filtered_count layers not available.")
        return None


@numba.njit(fastmath=True)
def np_nanstd(values):
    """
    For performance reasons, np.nanstd is encapsulated in this JIT function.
    generic_filter runs (VERY !) much faster this way, rather than invoking np.nanstd directly
    """
    return np.nanstd(values)


@numba.njit(fastmath=True)
def np_nanmean(values):
    """
    For performance reasons, np.nanmean is encapsulated in this JIT function.
    generic_filter runs (VERY !) much faster this way, rather than invoking np.nanmean directly
    """
    return np.nanmean(values)


def computes_compared_stdev(layers: Layers) -> ArrayLike | None:
    """Computes flag Standard deviation compared to neighbouring stdev"""
    # Standard deviation compared to neighbouring stdev
    if layers.value_count is not None and layers.stdev is not None:
        __logger.info("Check standard detection...")
        neighbourhood = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        mean_neighbourhood_stdev = generic_filter(
            layers.stdev, function=np_nanmean, footprint=neighbourhood, mode="constant", cval=np.nan
        )
        elev_stdev = generic_filter(
            layers.elevations, function=np_nanstd, footprint=neighbourhood, mode="constant", cval=np.nan
        )
        return np.where((layers.stdev - mean_neighbourhood_stdev) > elev_stdev, 0b00000100, 0)
    else:
        __logger.info("Check standard detection : not possible, value_count or stdev layers not available.")
        return None


def write_tif(dtm_file: dtm_driver.DtmFile, indicators: ArrayLike, o_path: PathLike, args: QualityIndicatorArgs):
    """Writing the result in a tiff file"""
    dataset = gdal.GetDriverByName("GTiff").Create(
        o_path,
        xsize=dtm_file.col_count,
        ysize=dtm_file.row_count,
        bands=1,
        eType=gdal.GDT_Byte,
    )
    dataset.SetGeoTransform(
        (dtm_file.west, dtm_file.spatial_resolution_x, 0.0, dtm_file.north, 0.0, -dtm_file.spatial_resolution_y)
    )

    dataset.SetProjection(dtm_file.spatial_reference.ExportToProj4())
    band = dataset.GetRasterBand(1)
    bit_field = f"interpolated,less_than_{args.minimum_detection_count}_detections,anormal_stdev,invalid_rate_{args.invalid_rate_max_threshold}_over_mean,max_across_angle_over_{args.across_angle_max_threshold}"
    band.SetDescription(f"bitfield=None,None,None,{bit_field}")
    band.SetNoDataValue(no_data_value)
    color_palette = (
        (255, 255, 255),  # 0b00000000
        (33, 150, 243),  # 0b00000001, blue
        (76, 175, 80),  # 0b00000010, green
        (253, 23, 146),  # 0b00000100, fuchsia
        (244, 67, 54),  # 0b00001000, red
        (255, 235, 59),  # 0b00010000, yellow
        (255, 127, 80),  # 0b00100000, CORAL
        (220, 20, 60),  # 0b01000000, CRIMSON
        (255, 215, 0),  # 0b01000000, Gold
    )

    # create color table
    # set color for each value
    color_table = gdal.ColorTable()
    color_table.SetColorEntry(0, color_palette[0])
    for bit in range(0, 8):
        value = 1 << bit
        for i in range(0, 1 << bit):
            color_table.SetColorEntry(value + i, color_palette[bit + 1])

    # Transparent color for no_data_value
    color_table.SetColorEntry(255, (0, 0, 0, 0))

    # set color table and color interpretation
    band.SetRasterColorTable(color_table)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    band.WriteArray(gdal_utils.netcdf_to_gdal(indicators))
    dataset.FlushCache()
    dataset = None
