#! /usr/bin/env python3
# coding: utf-8

import netCDF4 as nc
import numpy as np
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.numba.smoothing_functions as nb
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.netcdf_utils as nc_util
import pyat.utils.pyat_logger as log
from pyat.dtm.mask import compute_geo_mask_from_dtm
from pyat.utils import nc_encoding


class SmoothingProcess:
    """Smoothing process class. Create the layer elevation_smoothed based on the layer
    elevation.
    """

    def __init__(
        self,
        i_paths: list,
        o_paths: list = None,
        suffix="-smoothed",
        overwrite: bool = False,
        row_size: str = "3",
        col_size: str = "3",
        mask: str = None,
        monitor=DefaultMonitor,
    ):
        """By default, the name of the output file is i_path + "-smoothed". The size of the
        smoothed window is (3, 3). No zone selected.

        Arguments:
            i_paths {list} -- Input file list (.nc).
            o_paths {list} -- Optional output file list (.nc). (default: {None})
            suffix {str} -- Suffix of generated output path. Used when o_paths is empty. (default: {-smoothed})
            overwrite {bool} -- true to overwrite output file if exists. (default: {False})
            mask {list} -- Mask file list. (default: {None})
            monitor {list} -- Progress monitor. (default is a silent monitor: {DefaultMonitor})

        Raises:
            TypeError: row_size and col_size must be int or float.
            ValueError: row_size must be >= 3 and odd.
            ValueError: col_size must be >= 3 and odd.
        """
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.suffix = suffix
        self.overwrite = overwrite
        self.row_size = arg_util.parse_int("row_size", row_size, default=3, min_value=3)
        self.col_size = arg_util.parse_int("col_size", col_size, default=3, min_value=3)
        self.mask_files = arg_util.parse_list_of_files("mask", mask)
        self.monitor = monitor

        self.logger = log.logging.getLogger(self.__class__.__name__)

        # Check size
        if self.row_size < 3 or self.row_size % 2 == 0:
            raise ValueError(f"row_size must be >= 3 and odd. (not {self.row_size})")
        if self.col_size < 3 or self.col_size % 2 == 0:
            raise ValueError(f"col_size must be >= 3 and odd. (not {self.col_size})")

    def __process_data(
        self, i_driver: dtm_driver.DtmDriver, o_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor
    ) -> None:
        """Create variable and process it. Copy layers and create the smoothed layer at the end.

        Arguments:
            ind {int} -- Number of the processed file.
        """

        # Initialize output file
        process_util.initialize_output_file(i_driver, o_driver, process_name=self.__class__.__name__)

        i_file = i_driver.dataset
        o_file = o_driver.dataset

        # Used for the log
        count = 0
        n = len(i_file.variables)
        if not DtmConstants.ELEVATION_SMOOTHED_NAME in i_file.variables:
            n += 1
        monitor.set_work_remaining(n)

        # Create mask
        mask = compute_geo_mask_from_dtm(nc_encoding.filepath(i_file), self.mask_files)

        for name, variable in i_file.variables.items():
            if name in DtmConstants.LAYERS:
                count += 1
                # Create variable in the o_files[ind].
                o_file.createVariable(
                    name, variable.datatype, variable.dimensions, compression=nc_util.DEFAULT_COMPRESSION_LIB
                )
                log.info_progress_layer(self.logger, "layer", name, count, n)
                if name != DtmConstants.ELEVATION_SMOOTHED_NAME:
                    self.__process_layer(i_file, name, o_file)
                else:
                    self.__process_smoothing(i_file, name, o_file, mask)

            elif name == DtmConstants.CDI:
                # Copy cdi_ref
                count += 1
                log.info_progress_layer(self.logger, "layer", name, count, n)
                o_driver.create_cdi_reference_variable(cdi_util.trim_string_array(variable[:]))

            monitor.worked(1)

        if not DtmConstants.ELEVATION_SMOOTHED_NAME in o_file.variables.keys():
            # Create smoothed variable.
            count += 1
            name = DtmConstants.ELEVATION_SMOOTHED_NAME
            log.info_progress_layer(self.logger, "layer", name, count, n)
            variable = o_file[DtmConstants.ELEVATION_NAME]
            o_file.createVariable(
                name,
                variable.datatype,
                variable.dimensions,
                fill_value=variable._FillValue,
                compression=nc_util.DEFAULT_COMPRESSION_LIB,
            )
            self.__process_smoothing(i_file, name, o_file, mask)

            self.monitor.worked(1)

    def __process_layer(self, i_file: nc.Dataset, name: str, o_file: nc.Dataset) -> None:
        """Copy layer or set cdi for the cdi_index layer.

        Arguments:
            name {str} -- Name of the layer.
            ind {int} -- Indice of the input file.
        """
        # copy variable attributes all at once via dictionary
        o_file[name].setncatts(i_file[name].__dict__)
        o_file[name][:] = i_file[name][:]

    def __process_smoothing(self, i_file: nc.Dataset, name: str, o_file: nc.Dataset, mask: np.array) -> None:
        """Copy the attribute of stdev layer. Add long_name. Process the layer.

        Arguments:
            name {str} -- Name of the layer.
            ind {int} -- Indice of the input file.
        """
        # copy variable attributes all at once via dictionary
        o_file[name].setncatts(i_file[DtmConstants.STDEV].__dict__)
        o_file[name].long_name = "Smoothed elevation relative to sea level, computing with elevation variable"
        # Initialisation
        o_data = o_file[name][:].data
        i_data = i_file[DtmConstants.ELEVATION_NAME][:].data

        o_file[name][:] = nb.smoothing(o_data, i_data, mask, self.row_size, self.col_size)

    def run(self) -> None:
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
