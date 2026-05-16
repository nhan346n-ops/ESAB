#! /usr/bin/env python3
# coding: utf-8

from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.pyat_logger as log


class DefaultLayersProcess:
    """Default layers process class."""

    def __init__(
        self,
        i_paths: list,
        o_paths: list = None,
        suffix="-default_layers_added",
        overwrite=False,
        monitor=DefaultMonitor,
    ):
        """Constructor.

        Arguments:
            i_paths {list} -- Input file list (.nc).
            o_paths {list} -- Optional output file list (.nc). (default: {None})
            suffix {str} -- Suffix of generated output path. Used when o_paths is empty. (default: {-default_layers_added})
            overwrite {bool} -- true to overwrite output file if exists. (default: {False})
            monitor -- Progress monitor. (default is a silent monitor: {DefaultMonitor})
        """
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.suffix = suffix
        self.overwrite = overwrite
        self.monitor = monitor

        self.logger = log.logging.getLogger(self.__class__.__name__)

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
        n = len(DtmConstants.LAYERS_TYPE) + 3

        monitor.set_work_remaining(n)

        # Create layer from input
        for name, variable in i_driver.get_layers().items():
            if name in DtmConstants.LAYERS:
                # Create variable in the o_drivers[ind].
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

        # Create default layers
        for layer in DtmConstants.LAYERS_TYPE.keys():
            layers = o_driver.get_layers().keys()
            if not layer in layers:
                # Create default variable.
                o_driver.create_missing_layer(layer, i_driver[DtmConstants.ELEVATION_NAME])

                count += 1
                log.info_progress_layer(self.logger, "layer", layer, count, n)

                self.monitor.worked(1)

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
