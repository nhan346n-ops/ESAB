#! /usr/bin/env python3
# coding: utf-8

from typing import List

import numpy as np
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.numba.gap_filling_functions as nb
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.dtm.dtm_driver import get_missing_value
from pyat.dtm.numba.default_layers_functions import create_layer
from pyat.dtm.mask import compute_geo_mask_from_dtm


def __process_layer(o_driver: dtm_driver.DtmDriver, mask_size: int, mask: np.array) -> None:
    """For each layer except elevation layer and backscatter, copy data.
    For the elevation and backscatter layer, gap filling.

    Arguments:
        name {str} -- Name of the layer.
        ind {int} -- Indice of the input file.
    """
    # Initialisation
    i_elev = o_driver[DtmConstants.ELEVATION_NAME][:].data
    o_elev = np.copy(i_elev)

    o_interp = o_driver[DtmConstants.INTERPOLATION_FLAG][:].data
    # Fake CDI array if no CDI is defined in the input file
    o_cdi = (
        o_driver[DtmConstants.CDI_INDEX][:].data
        if DtmConstants.CDI_INDEX in o_driver
        else o_driver.prepare_data(DtmConstants.CDI_INDEX)
    )
    o_val_count = (
        o_driver[DtmConstants.VALUE_COUNT][:].data
        if DtmConstants.VALUE_COUNT in o_driver
        else np.zeros_like(o_elev, dtype=int)
    )

    # In function of the size of the mask, create matrix distance.
    index = nb.find_distance(mask_size)
    # Then transform it into coordinates.
    coord = nb.find_coord(index)

    # Elevation interpolation
    o_elev, o_interp, o_cdi, o_val_count = nb.interpolation(
        o_elev, i_elev, o_interp, o_cdi, o_val_count, coord, mask_size, mask
    )

    o_driver[DtmConstants.ELEVATION_NAME][:] = o_elev
    o_driver[DtmConstants.INTERPOLATION_FLAG][:] = o_interp
    if DtmConstants.CDI_INDEX in o_driver:
        o_driver[DtmConstants.CDI_INDEX][:] = o_cdi
    if DtmConstants.VALUE_COUNT in o_driver:
        o_driver[DtmConstants.VALUE_COUNT][:] = o_val_count

    if DtmConstants.BACKSCATTER in o_driver:
        # Backscatter interpolation
        i_backscatter = o_driver[DtmConstants.BACKSCATTER][:].data
        o_backscatter = np.copy(i_backscatter)
        o_backscatter, _, _, _ = nb.interpolation(
            o_backscatter, i_backscatter, o_interp, o_cdi, o_val_count, coord, mask_size, mask
        )
        o_driver[DtmConstants.BACKSCATTER][:] = o_backscatter


def process(o_driver: dtm_driver.DtmDriver, mask_size: int, mask: np.array, logger, current_step, step_count) -> None:
    """
    Process the elevation layer.
    """

    # Create interpolation layer if not exist
    current_step += 1
    log.info_progress_layer(logger, "layer", DtmConstants.INTERPOLATION_FLAG, current_step, step_count)
    if not DtmConstants.INTERPOLATION_FLAG in o_driver:
        # Create variable
        m_val = get_missing_value(DtmConstants.INTERPOLATION_FLAG)
        o_driver.add_layer(DtmConstants.INTERPOLATION_FLAG)

        # Initialisation
        o_data = o_driver[DtmConstants.INTERPOLATION_FLAG][:].data
        i_data = o_driver[DtmConstants.ELEVATION_NAME][:].data
        o_driver[DtmConstants.INTERPOLATION_FLAG][:] = create_layer(o_data, i_data, m_val, mode=2)

    # Process elevation layer at the end.
    current_step += 1
    log.info_progress_layer(logger, "layer", DtmConstants.ELEVATION_NAME, current_step, step_count)
    __process_layer(o_driver, mask_size, mask)


class GapFillingProcess:
    """Gap filling class with bilinear interpolation."""

    def __init__(
        self,
        i_paths: List,
        o_paths: List = None,
        suffix="-gap_filling",
        overwrite: bool = False,
        mask_size: int = 3,
        mask: List[str] | None = None,
        reverse_mask: bool = False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """Constructor.

        Arguments:
            i_paths {list} -- Input file list (.nc).
            o_paths {list} -- Optional output file list (.nc). (default: {None})
            suffix {str} -- Suffix of generated output path. Used when o_paths is empty. (default: {-gap_filling})
            overwrite {bool} -- true to overwrite output file if exists. (default: {False})
            mask_size {int} -- Size of the mask. (default: {3})
            mask {list} -- Mask file list. (default: {None})
            monitor {list} -- Progress monitor. (default is a silent monitor: {DefaultMonitor})
        """
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.suffix = suffix
        self.overwrite = overwrite
        self.mask_size = arg_util.parse_int("mask_size", mask_size, default=3, min_value=3, max_value=31)
        self.mask_files = arg_util.parse_list_of_files("mask", mask)
        self.reverse_mask = reverse_mask
        self.monitor = monitor

        self.logger = log.logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Set mask_size to {self.mask_size}")

    def __process_data(
        self, i_driver: dtm_driver.DtmDriver, o_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor
    ) -> None:
        """For each variable in input file, create the variable, copy/process data
        in the output variable. Process the elevation layer at the end.

        Arguments:
            ind {int} -- Number of the processed file.
        """

        # Initialize output file
        process_util.initialize_output_file(i_driver, o_driver, process_name=self.__class__.__name__)

        # Used for the log
        count = 0
        n = len(i_driver.get_layers())
        if not DtmConstants.INTERPOLATION_FLAG in i_driver.get_layers().keys():
            n += 1

        monitor.set_work_remaining(n)
        mask = compute_geo_mask_from_dtm(i_driver.get_file_path(), self.mask_files, self.reverse_mask)

        for name, variable in i_driver.get_layers().items():
            if name in DtmConstants.LAYERS:
                o_layer = o_driver.add_layer(name)
                o_layer[:] = i_driver[name][:]
                count += 1
                log.info_progress_layer(self.logger, "layer", name, count, n)

            elif name in [DtmConstants.CDI]:
                # Copy cdi_ref
                count += 1
                log.info_progress_layer(self.logger, "layer", name, count, n)
                o_driver.create_cdi_reference_variable(cdi_util.trim_string_array(variable[:]))

            monitor.worked(1)

        process(o_driver, self.mask_size, mask, self.logger, count, n)

        monitor.worked(1)

    def __call__(self) -> None:
        process_util.process_each_input_dtm_to_output_dtm(
            self.__class__.__name__,
            self.i_paths,
            self.__process_data,
            self.logger,
            self.o_paths,
            self.suffix,
            self.overwrite,
            self.monitor,
        )
