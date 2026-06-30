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
from pyat.common.convolve import convolve
from pyat.dtm.utils.min_max_layer_util import update_min_max
from pyat.dtm.mask import compute_geo_mask_from_dtm

# kernels and their coefficients
flat = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 1]])

plus_kernel = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]])

x_kernel = np.array([[1, 0, 1], [0, 4, 0], [1, 0, 1]])

basic_gauss = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])

# if you need to create your custom kernel, you can change the coefficient below,
# even increase the size (as long as it is a odd size)

custom_kernel = np.array([[0, 5, 0], [5, 0, 5], [0, 5, 0]])

custom_kernel2 = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]])

# list of known kernel and their name, must match the list defined in kernel_smoothing.json
known_kernels = {
    "3x3 flat": flat,
    "3x3 cross": plus_kernel,
    "3x3 X": x_kernel,
    "5x5 gaussian": basic_gauss,
    "custom": custom_kernel,
    "custom2": custom_kernel2,
}


#


def convolve_array(input_array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    convolve the input array with the given kernel
    """
    # normalize kernel array
    kernel = kernel / kernel.sum()
    v = convolve.convolve(input_array, kernel, preserve_nan=True)
    return v


class SmoothingProcess:
    """Smoothing process class. Smooth the elevation layer and update min/max layer accordingly"""

    def __init__(
        self,
        i_paths: list,
        o_paths: list = None,
        suffix="-smoothed",
        overwrite: bool = False,
        kernel_choice: str = "3x3 flat",
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
            kernel_choice {bool} -- kernel choice. (default: {"3x3 flat"})
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
        self.mask_files = arg_util.parse_list_of_files("mask", mask)
        self.monitor = monitor

        self.logger = log.logging.getLogger(self.__class__.__name__)

        self.gaussian_sigma = None
        self.kernel = None

        if kernel_choice in known_kernels:
            self.kernel = known_kernels[kernel_choice]
            self.logger.info(f"Using kernel {kernel_choice}\n:{self.kernel}")
        else:
            raise ValueError(f"Invalid value '{kernel_choice}' for argument kernel_choice")

    def __process_data(
        self, i_driver: dtm_driver.DtmDriver, o_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor
    ) -> None:
        """Create variable and process it. Copy layers and create the smoothed layer at the end.

        Arguments:
            ind {int} -- Number of the processed file.
        """
        # Initialize output file
        process_util.initialize_output_file(i_driver, o_driver, process_name=self.__class__.__name__)

        # Used for the log
        count = 0
        n = len(i_driver.get_layers())
        monitor.set_work_remaining(n)

        # Create mask
        mask = compute_geo_mask_from_dtm(i_driver.get_file_path(), self.mask_files)

        # find and process elevation variable
        if DtmConstants.ELEVATION_NAME not in i_driver:
            raise ValueError(
                f"{i_driver.get_file_path()} : cannot find mandatory netcdf variable {DtmConstants.ELEVATION_NAME}"
            )

        elevation_variable = i_driver[DtmConstants.ELEVATION_NAME]
        self.update_elevation(i_driver, elevation_variable, o_driver, mask=mask)
        count += 1
        log.info_progress_layer(self.logger, "layer", DtmConstants.ELEVATION_NAME, count, n)

        for name, variable in i_driver.get_layers().items():
            if name in [DtmConstants.ELEVATION_NAME]:
                # already processed
                pass
            elif name in [DtmConstants.CDI]:
                # Copy cdi_ref
                count += 1
                log.info_progress_layer(self.logger, "layer", name, count, n)
                o_driver.create_cdi_reference_variable(cdi_util.trim_string_array(variable[:]))
            elif name in [DtmConstants.ELEVATION_MIN]:
                count += 1
                log.info_progress_layer(self.logger, "layer", name, count, n)
                o_driver.add_layer(name, update_min_max(elevation_variable, min_layer=variable))
            elif name in [DtmConstants.ELEVATION_MAX]:
                count += 1
                log.info_progress_layer(self.logger, "layer", name, count, n)
                o_driver.add_layer(name, update_min_max(elevation_variable, max_layer=variable))
            elif name in DtmConstants.LAYERS:
                count += 1
                # Create variable in the o_drivers[ind].
                o_driver.add_layer(name, variable[:])
                log.info_progress_layer(self.logger, "layer", name, count, n)
            else:
                count += 1

            monitor.worked(1)

    def update_elevation(
        self, i_driver: dtm_driver.DtmDriver, elevation_variable, o_driver: dtm_driver.DtmDriver, mask
    ):
        o_driver.add_layer(DtmConstants.ELEVATION_NAME, i_driver[DtmConstants.ELEVATION_NAME][:])

        mask = None
        if self.mask_files is not None:
            mask = compute_geo_mask_from_dtm(i_driver.get_file_path(), self.mask_files)
        # smooth depth layer
        if self.kernel is not None:
            ret = convolve_array(input_array=elevation_variable[:], kernel=self.kernel)
            ret = np.ma.masked_invalid(ret)
            if mask is not None:
                values = o_driver[DtmConstants.ELEVATION_NAME][:]
                np.putmask(values, mask == 1, ret)
                o_driver[DtmConstants.ELEVATION_NAME][:] = values
            else:
                o_driver[DtmConstants.ELEVATION_NAME][:] = ret

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
