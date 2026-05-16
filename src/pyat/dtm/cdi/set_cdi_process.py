#! /usr/bin/env python3
# coding: utf-8

import os
from typing import Dict, Optional

import numpy as np
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.dtm.mask import compute_geo_mask_from_dtm


class SetCdiProcess:
    """Set cdi process class. Put all cdis together in one. If no cdi, create one cdi
    based on elevation layer.
    """

    def __init__(
        self,
        i_paths: list,
        cdi: Optional[Dict[str, str]] = None,
        o_paths: list = None,
        suffix="-cdi",
        mask: str = None,
        cell_without_cdi: bool = False,
        overwrite: bool = False,
        monitor=DefaultMonitor,
    ):
        """By default, the name of the cdi is SDN:CDI:LOCAL:0. The name of the output path is
        i_path + "-cdi.

        Arguments:
            i_paths {list} -- Input file list (.nc).
            cdi {str} -- Name of the futur cdi.
            o_paths {list} -- Optional output file list (.nc). (default: {None})
            suffix {str} -- Suffix of generated output path. Used when o_paths is empty. (default: {-cdi})
            overwrite {bool} -- true to overwrite output file if exists. (default: {False})
            monitor {list} -- Progress monitor. (default is a silent monitor: {DefaultMonitor})
        """
        self.i_paths = i_paths
        self.cdi = cdi
        self.o_paths = o_paths
        self.suffix = suffix
        self.mask_files = arg_util.parse_list_of_files("mask", mask)
        self.cell_without_cdi = cell_without_cdi
        self.overwrite = overwrite
        self.monitor = monitor

        self.logger = log.logging.getLogger(self.__class__.__name__)

        if not self.cdi:
            raise ValueError(
                "Useless process without parameters.\nStop the program.\nPlease enter the following parameter:\n"
                "- Set cdi with the option -c CDI, --cdi CDI.\n"
            )

    def __create_mask(self, i_driver: dtm_driver.DtmDriver) -> np.ndarray:
        """Create global mask.
        If no filters, mask is an array full of true.
        Arguments:
            i_driver -- input DTM.
        Returns:
            [np.array] -- Mask array.
        """
        result = compute_geo_mask_from_dtm(i_driver.get_file_path(), self.mask_files) > 0
        result = np.logical_and(result, ~i_driver[DtmConstants.ELEVATION_NAME][:].mask)
        if self.cell_without_cdi and DtmConstants.CDI_INDEX in i_driver:
            result = np.logical_and(result, i_driver[DtmConstants.CDI_INDEX][:].mask)
        return result

    def __infer_cdi(self, dtm_file_path: str) -> Optional[str]:
        if self.cdi is None or len(self.cdi) == 0:
            return None
        dtm_file_name = os.path.basename(dtm_file_path)
        if dtm_file_name in self.cdi:
            self.logger.info(f"CDI of {dtm_file_name} is {self.cdi[dtm_file_name]}")
            return self.cdi[dtm_file_name]
        return None

    def __process_data(
        self, i_driver: dtm_driver.DtmDriver, o_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor
    ) -> None:
        """Create variable and process it. Copy dimensions, copy layers except the cdi_index.

        Arguments:
            ind {int} -- Number of the processed file.
        """

        cdi = self.__infer_cdi(i_driver.dtm_file.file_path)
        if not cdi:
            self.logger.warning(f"No CDI specified for {i_driver.dtm_file.file_path}. File skipped")
            return

        # Initialize output file
        process_util.initialize_output_file(i_driver, o_driver, process_name=self.__class__.__name__)

        # If there is no filter, the purpose is to set the same CDI to all cells
        new_cdi_index = 0
        all_cdi = []
        if (len(self.mask_files) > 0 or self.cell_without_cdi) and DtmConstants.CDI in i_driver:
            # Define a new index for the new CDI
            all_cdi = list(i_driver[DtmConstants.CDI][:])
            new_cdi_index = len(all_cdi)
        all_cdi.append(cdi)

        # Create the mask
        mask = self.__create_mask(i_driver)

        # Used for the log
        count = 0
        n = len(i_driver.get_layers())
        monitor.set_work_remaining(n)

        for name in i_driver.get_layers().keys():
            monitor.worked(1)
            if name in [DtmConstants.CDI_INDEX, DtmConstants.CDI]:
                # Layer ignore, processed last
                continue
            count += 1
            log.info_progress_layer(self.logger, "layer", name, count, n)
            if name in DtmConstants.LAYERS:
                # Create variable in the o_drivers[ind].
                o_driver.add_layer(name, i_driver[name][:])

        # Initialization of CDI_INDEX layer
        count += 1
        log.info_progress_layer(self.logger, "layer", DtmConstants.CDI_INDEX, count, n)
        o_layer_cdi_index = o_driver.add_layer(DtmConstants.CDI_INDEX)
        if DtmConstants.CDI_INDEX in i_driver:
            o_layer_cdi_index.setncatts(i_driver[DtmConstants.CDI_INDEX].__dict__)
            o_layer_cdi_index[:] = i_driver[DtmConstants.CDI_INDEX][:]
        # The .data propertiy allow us to access values regardless of the mask
        o_cdi_index = o_layer_cdi_index[:].data
        o_layer_cdi_index[:] = np.where(mask, new_cdi_index, o_cdi_index)

        # Copy cdi_ref
        count += 1
        log.info_progress_layer(self.logger, "layer", DtmConstants.CDI, count, n)
        o_driver.create_cdi_reference_variable(cdi_util.trim_string_array(all_cdi))
        cdi_util.clean_cdi(o_driver.dataset)

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
