#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.dtm.mask import compute_geo_mask_from_dtm


class LinearTransformProcess:
    """Linear transformation class which process the layers elevation, elevation_min,
    elevation_max and stdev.
    """

    def __init__(
        self,
        i_paths: list,
        o_paths: list = None,
        suffix="-linear_transform",
        overwrite: bool = False,
        a: float = 1.0,
        b: float = 0.0,
        mask: str = None,
        monitor=DefaultMonitor,
    ):
        """Constructor.

        Arguments:
            i_paths {list} -- Input file list (.nc).
            o_paths {list} -- Optional output file list (.nc). (default: {None})
            suffix {str} -- Suffix of generated output path. Used when o_paths is empty. (default: {-linear_transform})
            overwrite {bool} -- true to overwrite output file if exists. (default: {False})
            a {float} -- Multiplicator. (default: {1.0})
            b {float} -- Adder. (default: {0.0})
            mask {list} -- Mask file list. (default: {None})
            monitor {list} -- Progress monitor. (default is a silent monitor: {DefaultMonitor})
        """
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.suffix = suffix
        self.overwrite = overwrite
        self.a = arg_util.parse_float("a", a, default=1.0)
        self.b = arg_util.parse_float("b", b, default=0.0)
        self.mask_files = arg_util.parse_list_of_files("mask", mask)
        self.monitor = monitor

        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __process_data(
        self, i_driver: dtm_driver.DtmDriver, o_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor
    ) -> None:
        """Create variable, and process it.

        Arguments:
            ind {int} -- Number of the processed file.
        """

        if self.a == 1 and self.b == 0:
            self.logger.warning(
                "Useless process without parameters (or with default values). Please enter the parameter a or b with the option -a A or --a A, or -b B or --b B."
            )

        # Initialize output file
        process_util.initialize_output_file(i_driver, o_driver, process_name=self.__class__.__name__)

        # Used for the log
        count = 0
        n = len(i_driver.get_layers())

        monitor.set_work_remaining(n)
        mask = compute_geo_mask_from_dtm(i_driver.get_file_path(), self.mask_files)

        for name, variable in i_driver.get_layers().items():
            if name in DtmConstants.LAYERS:
                count += 1
                log.info_progress_layer(self.logger, "layer", name, count, n)

                # Create variable in the o_drivers[ind].
                o_driver.add_layer(name)
                self.__process_layer(i_driver, name, o_driver, mask)

            elif name in [DtmConstants.CDI]:
                # Copy cdi_ref
                count += 1
                log.info_progress_layer(self.logger, "layer", name, count, n)
                o_driver.create_cdi_reference_variable(cdi_util.trim_string_array(variable[:]))

            monitor.worked(1)

    def __process_layer(self, i_driver: dtm_driver.DtmDriver, name: str, o_driver: dtm_driver.DtmDriver, mask) -> None:
        """Guide the layer to their corresponding numba function.

        Arguments:
            name {str} -- Name of the layer.
            ind {int} -- Indice of the input file.
        """
        # Initialisation
        o_data = o_driver[name][:].data
        i_data = i_driver[name][:].data

        if name in [DtmConstants.ELEVATION_NAME]:
            o_driver[name][:] = np.where(mask, i_data * self.a + self.b, i_data)

        elif name in [DtmConstants.ELEVATION_MIN, DtmConstants.ELEVATION_MAX]:
            if self.a <= 0:
                if name == DtmConstants.ELEVATION_MAX:
                    i_data = i_driver[DtmConstants.ELEVATION_MIN][:].data
                else:
                    i_data = i_driver[DtmConstants.ELEVATION_MAX][:].data
            o_driver[name][:] = np.where(mask, i_data * self.a + self.b, i_data)

        elif name == DtmConstants.STDEV:
            o_driver[name][:] = np.where(mask, i_data * abs(self.a), i_data)
        else:
            o_driver[name][:] = i_driver[name][:]

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
