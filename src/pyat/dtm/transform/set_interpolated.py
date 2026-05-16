#! /usr/bin/env python3
# coding: utf-8

import netCDF4 as nc
from numpy import int8
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.numba.reset_cell_functions as nb
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.netcdf_utils as nc_util
import pyat.utils.pyat_logger as log
from pyat.dtm.mask import compute_geo_mask_from_dtm


class SetInterpolatedProcess:
    """Set Interpolated process class. Can set cells as interpolated."""

    def __init__(
        self,
        i_paths: list,
        o_paths: list = None,
        suffix: str = "-interpolated",
        overwrite: bool = False,
        mask: str = None,
        monitor=DefaultMonitor,
    ):
        """By default, the name of the output file is i_path + "-interpolated". No filter of zone.

        Arguments:
            i_paths {list} -- Input file list (.nc).
            o_paths {list} -- Optional output file list (.nc). (default: {None})
            suffix {str} -- Suffix of generated output path. Used when o_paths is empty. (default: {-zeroed})
            overwrite {bool} -- true to overwrite output file if exists. (default: {False})
            mask {list} -- Mask file list. (default: {None})
            monitor {list} -- Progress monitor. (default is a silent monitor: {DefaultMonitor})

        Raises:
            TypeError: Not good format for lat / lon.
            ValueError: Raise an exception if the layer isn't in the list layers_filter.
            ValueError: Raise an exception if the operation filter isn't in the list name_oper.
        """
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.suffix = suffix
        self.overwrite = overwrite
        self.mask_files = arg_util.parse_list_of_files("mask", mask)
        self.monitor = monitor
        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __process_data(
        self, i_driver: dtm_driver.DtmDriver, o_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor
    ) -> None:
        """Create the layers and process it.

        Arguments:
            ind {int} -- Number of the processed file.
        """

        # Initialize output file
        process_util.initialize_output_file(i_driver, o_driver, process_name=self.__class__.__name__)

        o_file = o_driver.dataset

        # Used for the log
        count = 0
        n = len(i_driver.get_layers())
        monitor.set_work_remaining(n + 1)

        geo_mask = compute_geo_mask_from_dtm(i_driver.get_file_path(), self.mask_files)
        # geo_mask is set to one
        geo_mask = geo_mask > 0  # convert to boolean array

        # copy all dtm layers
        for name, variable in i_driver.get_layers().items():
            if name in DtmConstants.LAYERS:
                count += 1
                log.info_progress_layer(self.logger, "layer", name, count, n)

                # Create variable in the o_files[ind].
                o_file.createVariable(
                    name, variable.datatype, variable.dimensions, compression=nc_util.DEFAULT_COMPRESSION_LIB
                )
                self.__process_layer(i_driver, name, o_file, geo_mask)

            elif name == DtmConstants.CDI:
                # Copy cdi layer
                count += 1
                log.info_progress_layer(self.logger, "layer", name, count, n)
                o_driver.create_cdi_reference_variable(cdi_util.trim_string_array(variable[:]))
            monitor.worked(1)

        # create default interpolation layer if not exist
        if DtmConstants.INTERPOLATION_FLAG not in o_driver.get_layers().keys():
            self.__create_interpolation_layer(o_driver, geo_mask)

    def __process_layer(self, i_driver: dtm_driver.DtmDriver, name: str, o_file: nc.Dataset, mask) -> None:
        """Process layer.

        Arguments:
            name {str} -- Name of the layer.
        """
        # copy variable attributes all at once via dictionary
        o_file[name].setncatts(i_driver[name].__dict__)

        # Initialisation
        if name == DtmConstants.INTERPOLATION_FLAG:
            o_data = o_file[name][:].data
            i_data = i_driver[name][:].data
            m_val = i_driver[name]._FillValue
            # mark as interpolated if not missing
            o_file[name][:] = nb.set_layer(o_arr=o_data, i_arr=i_data, val=int8(1), m_val=m_val, mask=mask)
        else:
            # simple copy
            o_file[name][:] = i_driver[name][:]

    def __create_interpolation_layer(self, o_driver: dtm_driver.DtmDriver, mask) -> None:
        o_driver.create_interpolation_layer()
        # Initialisation
        o_data = o_driver[DtmConstants.INTERPOLATION_FLAG][:].data
        m_val = o_driver[DtmConstants.INTERPOLATION_FLAG]._FillValue
        # mark as interpolated if not missing
        o_driver[DtmConstants.INTERPOLATION_FLAG][:] = nb.set_layer(
            o_arr=o_data, i_arr=o_data, val=int8(1), m_val=m_val, mask=mask
        )

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
