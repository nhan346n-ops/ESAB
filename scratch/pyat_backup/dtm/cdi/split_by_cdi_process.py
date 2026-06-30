#! /usr/bin/env python3
# coding: utf-8

from typing import List

from pygws.service.progress_monitor import DefaultMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.pyat_logger as log
from pyat.dtm.cdi.cdi_layer_util import check_undefined_cdi
from pyat.dtm.transform.reset_cell import ResetCellProcess

# Constants for filter operations
NOT_EQUAL: str = "not_equal"
NOT_MISSING: str = "not_missing"

class SplitByCdiProcess:
    """Split by cdi process class. The process is going to split all the layers by cdi. It creates
    one file for one cdi. For the cdi_index layer, set it to missing_value or 0 in function of the
    selected cdi. For the cdi_ref, put the selected on the "0" index and remove the others.
    """

    def __init__(
        self,
        i_paths: list,
        suffix="-cdi",
        overwrite: bool = False,
        monitor=DefaultMonitor,
    ):
        """Can process multiple files.

        Arguments:
            i_paths {list} -- List of dtm file input paths.
            suffix {str} -- Suffix of generated output path. Used when o_paths is empty. (default: {-cdi})
            overwrite {bool} -- true to overwrite output file if exists. (default: {False})

        """
        self.i_paths = i_paths
        self.suffix = suffix
        self.overwrite = overwrite
        self.monitor = monitor

        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __process_data(self, i_driver: dtm_driver.DtmDriver, monitor) -> None:
        # For each cdi, use the reset cell process, and create a new file.
        if not DtmConstants.CDI in i_driver:
            raise ValueError("No CDI found")

        cdis: List[str] = list(filter(None, i_driver[DtmConstants.CDI][:]))
        cdi_index = i_driver[DtmConstants.CDI_INDEX][:]
        n = len(cdis)

        # Check if there is valid cells with missing CDI
        has_missing_cdi = not check_undefined_cdi(i_driver.dataset)
        if has_missing_cdi:
            n += 1

        sub_monitor = monitor.split(n)

        # Close file
        i_path = i_driver.dtm_file.file_path
        i_driver.close()

        for ind_cdi, cdi in enumerate(cdis):
            # Check if the cdi is in the cdi_index layer.
            if (cdi_index == ind_cdi).any():
                # Parameters
                if ":" in cdi:
                    cdi = cdi[cdi.rfind(":") + 1 :]
                suffix = f"{self.suffix}_{cdi}"
                filters = [{"filter_layer": DtmConstants.CDI_INDEX, "oper": NOT_EQUAL, "a": ind_cdi}]

                log.info_progress(self.logger, "Create file for cdi " + cdi, ind_cdi + 1, n)

                # Process
                resetCell = ResetCellProcess(
                    i_paths=[i_path],
                    filters=filters,
                    suffix=suffix,
                    monitor=sub_monitor,
                    overwrite=self.overwrite,
                )
                resetCell()

        # Generate missing CDI DTM if there is valid cells with missing CDI
        if has_missing_cdi:
            filters = [{"filter_layer": DtmConstants.CDI_INDEX, "oper": NOT_MISSING}]

            log.info_progress(self.logger, "Create file for missing cdi ", len(cdis), n)
            suffix = f"{self.suffix}_missing"

            # Process
            resetCell = ResetCellProcess(
                i_paths=[i_path],
                filters=filters,
                suffix=suffix,
                monitor=sub_monitor,
                overwrite=self.overwrite,
            )
            resetCell()

    def __call__(self) -> None:
        process_util.process_each_input_file_in_read_mode(
            self.i_paths,
            self.__class__.__name__,
            self.logger,
            self.monitor,
            self.__process_data,
        )
