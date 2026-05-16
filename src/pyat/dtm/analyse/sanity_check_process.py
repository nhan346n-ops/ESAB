#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.numba.sanity_check_functions as nb
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.pyat_logger as log
from pyat.dtm.numba.reset_cell_functions import reset_layer


class SanityCheckProcess:
    """Sanity Check process class. This process will perform a sanity check on the given files

    Check for interpolation flag layer. For each cell, set it to zero
    (not interpolated) if depth value exists and if its interpolation flag is set to invalid value.
    Leave it unchanged if it is already set to a valid value.

    Reset cells where elevation data is missing

    Compress CDI metadata or Force cdi: If one and only one CDI is declared in the nc file, and if this
    CDI is not used. i.e. the CDI_LAYER is fully filled with invalid values, then set this CDI to valid
    depth values in the file.
    Compress CDI metadata. Remove declared CDI entries that are not effectively
    used in the file.

    CDi interpolation : Recompute cdi of interpolated cell with the nearest valid CDI
    """

    def __init__(
        self,
        i_paths: list,
        o_paths: list = None,
        suffix="-cleaned",
        overwrite: bool = False,
        reset_missing_elev: bool = False,
        interp: bool = False,
        cdi: bool = False,
        cdi_interp: bool = False,
        monitor=DefaultMonitor,
    ):
        """

        Arguments:
            i_paths {list} -- Input file list (.nc).
            o_paths {list} -- Optional output file list (.nc). (default: {None})
            suffix {str} -- Suffix of generated output path. Used when o_paths is empty. (default: {-cleaned})
            overwrite {bool} -- true to overwrite output file if exists. (default: {False})
            reset_missing_elev {bool} -- true to reset cells where elevation value is missing. (default: {False})
            interp {bool} -- set to recompute interpolation flags. (default: {False})
            cdi {bool} -- set to compress CDIs. (default: {False})
            cdi_interp {bool} -- set to recompute interpolated CDIs. (default: {False})
            monitor {list} -- Progress monitor. (default is a silent monitor: {DefaultMonitor})

        """
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.suffix = suffix
        self.overwrite = overwrite
        self.interp = interp
        self.cdi = cdi
        self.cdi_interp = cdi_interp
        self.reset_missing_elev = reset_missing_elev

        self.monitor = monitor

        if not (self.cdi or self.interp or self.cdi_interp or self.reset_missing_elev):
            raise ValueError(
                "Useless process without parameters.\nStop the program.\nPlease enter at least one of the following parameters:\n"
                "- Reset cells where elevation value is missing with the option -r, --reset\n"
                "- Recompute interpolation flag with the option -in, --interp.\n"
                "- Compress cdi reference list with the option -c, --cdi.\n"
                "- Recompute cdi of interpolated cells with the option -ci, --cdi_interp."
            )

        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __process_data(
        self, i_driver: dtm_driver.DtmDriver, o_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor
    ) -> None:
        """Create variable and process it. Copy or clean layers.

        Arguments:
            ind {int} -- Number of the processed file.
        """
        # Initialize output file
        process_util.initialize_output_file(i_driver, o_driver, process_name=self.__class__.__name__)

        # Used for the log
        count = 0
        n = len(i_driver.get_layers())
        monitor.set_work_remaining(n + 1)

        for name in i_driver.get_layers().keys():
            if name in DtmConstants.LAYERS:
                count += 1
                # Create variable in the o_drivers[ind].
                o_driver.add_layer(name)
                # Copy variable attributes all at once via dictionary
                log.info_progress_layer(self.logger, "layer", name, count, n)
                self.__process_layer(i_driver, name, o_driver)

            elif name in [DtmConstants.CDI]:
                # Copy cdi_ref
                count += 1
                log.info_progress_layer(self.logger, "layer", name, count, n)
                self.__process_cdi_ref(i_driver, o_driver)

            monitor.worked(1)

        # cdi ref cleanup
        self.__clean_and_compress_cdi(o_driver)

        # cdi interpolation after layers creation and cdi compression
        self.__update_interpolated_cdi(o_driver)

        # Check presence of undefined CDI.
        self.__check_undefined_cdi(i_driver)

        monitor.worked(1)

    def __process_layer(self, i_driver: dtm_driver.DtmDriver, name: str, o_driver: dtm_driver.DtmDriver) -> None:
        """Copy layer or update the interpolation_flag layer.

        Arguments:
            name {str} -- Name of the layer.
            ind {int} -- Indice of the input file.
        """

        # data copy. If a valid_range is defined, we retrieve a masked_array
        o_driver[name][:] = i_driver[name][:]
        # Get input data with masked value filled
        m_val = o_driver[name]._FillValue
        i_data = np.ma.filled(o_driver[name][:], fill_value=m_val)

        if self.interp and name == DtmConstants.INTERPOLATION_FLAG:
            o_data = o_driver[name][:].data
            i_elev = i_driver[DtmConstants.ELEVATION_NAME][:].data
            self.logger.debug("Update interpolation flag.")
            o_driver[name][:] = nb.update_interp(o_data, i_data, i_elev, m_val)
        else:
            o_driver[name][:] = i_data

        if self.reset_missing_elev and name != DtmConstants.ELEVATION_NAME:
            o_data = o_driver[name][:].data
            i_elev_mask = i_driver[DtmConstants.ELEVATION_NAME][:].mask
            o_driver[name][:] = reset_layer(o_data, i_data, m_val, i_elev_mask)

    def __process_cdi_ref(self, i_driver: dtm_driver.DtmDriver, o_driver: dtm_driver.DtmDriver) -> None:
        """Set the long_name attributes and copy data for the cdi_ref layer."""
        # Initialisation
        i_data = i_driver[DtmConstants.CDI][:]
        o_driver.create_cdi_reference_variable(cdi_util.trim_string_array(i_data))

    def __clean_and_compress_cdi(self, o_driver: dtm_driver.DtmDriver) -> None:
        """Clean and compress cdi"""
        if self.cdi:
            cdi_util.clean_cdi(o_driver.dataset)

    def __update_interpolated_cdi(self, o_driver: dtm_driver.DtmDriver) -> None:
        """Update interpolated cdis with closest not interpolated cdi"""
        # Initialisation
        if self.cdi_interp:
            if DtmConstants.INTERPOLATION_FLAG in o_driver:
                interpolation_mask = o_driver[DtmConstants.INTERPOLATION_FLAG][:] == 1
                cdi_util.update_with_closest_cdi(o_driver, interpolation_mask)

    def __check_undefined_cdi(self, o_driver: dtm_driver.DtmDriver) -> None:
        """Check the presence of valid cells without valid CDI"""
        cdi_util.check_undefined_cdi(o_driver.dataset)

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
