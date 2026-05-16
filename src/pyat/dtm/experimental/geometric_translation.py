#! /usr/bin/env python3
# coding: utf-8

from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log


class GeometricTranslationProcess:
    """Geometric Translation class which process the layers elevation, elevation_min,
    elevation_max and stdev.
    """

    def __init__(
        self,
        i_paths: list,
        o_paths: list = None,
        suffix="-geometric_translation",
        overwrite: bool = False,
        rows: str = "0.0",
        columns: str = "0.0",
        monitor=DefaultMonitor,
    ):
        """Constructor.

        Arguments:
            i_paths {list} -- Input file list (.nc).
            o_paths {list} -- Optional output file list (.nc). (default: {None})
            suffix {str} -- Suffix of generated output path. Used when o_paths is empty. (default: {-default_layers_added})
            overwrite {bool} -- true to overwrite output file if exists. (default: {False})
            rows {str} -- Number of rows shift applied to the data (unit is a cell size). (default: {0.0})
            columns {str} -- Number of columns shift applied to the data (unit is a cell size). (default: {0.0})
            monitor {list} -- Progress monitor. (default is a silent monitor: {DefaultMonitor})
        """
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.suffix = suffix
        self.overwrite = overwrite
        self.row_shift = arg_util.parse_float("row_shift", rows)
        self.col_shift = arg_util.parse_float("col_shift", columns)
        self.monitor = monitor

        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __process_data(
        self, i_driver: dtm_driver.DtmDriver, o_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor
    ) -> None:
        """Create variable, and process it.

        Arguments:
            ind {int} -- Number of the processed file.
        """
        # Initialize output file
        process_util.initialize_output_file(i_driver, o_driver, process_name=self.__class__.__name__)

        # Used for the log
        count = 0
        n = len(i_driver.get_layers())

        monitor.set_work_remaining(n)

        for name, variable in i_driver.get_layers().items():
            if name in DtmConstants.LAYERS:
                count += 1
                log.info_progress_layer(self.logger, "layer", name, count, n)

                # Create variable in the o_drivers[ind].
                o_layer = o_driver.add_layer(name)
                o_layer[:] = i_driver[name][:]

            elif name in [DtmConstants.CDI]:
                # Copy cdi_ref
                count += 1
                log.info_progress_layer(self.logger, "layer", name, count, n)
                o_driver.create_cdi_reference_variable(cdi_util.trim_string_array(variable[:]))

            monitor.worked(1)

        # Now do the shift
        if self.col_shift != 0.0:
            data = o_driver.get_x_axis()[:]
            data += o_driver.dtm_file.spatial_resolution_x * self.col_shift
            o_driver.get_x_axis()[:] = data

        if self.row_shift != 0.0:
            data = o_driver.get_y_axis()[:]
            data += o_driver.dtm_file.spatial_resolution_y * self.row_shift
            o_driver.get_y_axis()[:] = data

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
