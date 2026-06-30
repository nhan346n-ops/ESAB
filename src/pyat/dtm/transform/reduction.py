#! /usr/bin/env python3
# coding: utf-8

from collections import Counter

import numpy as np
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.numba.reduction_functions as nb
import pyat.dtm.utils.dtm_utils as dtmut
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.netcdf_utils as nc_util
import pyat.utils.pyat_logger as log


class ReductionProcess:
    """Class Reduce which is used for the reduction process of a dtm file (nc4)."""

    def __init__(
        self,
        i_paths: list,
        o_paths: list = None,
        suffix="-reduced_",
        overwrite: bool = False,
        factor: str = "4",
        layers: dict = None,
        monitor=DefaultMonitor,
    ):
        """If o_paths == None, the program create the directory = input_dir + "/reduced/.
        The name of the output file = input_name + "_Reduced_" + factor of reduction + ".nc".

        If params == None, then the factor of reduction = 4.
                                all layers are reduced.

        Arguments:
            i_paths {list} -- Input file list (.nc).
            o_paths {list} -- Optional output file list (.nc). (default: {None})
            suffix {str} -- Suffix of generated output path. Used when o_paths is empty. (default: {-reduced_})
            overwrite {bool} -- true to overwrite output file if exists. (default: {False})
            factor {int} -- Reduction factor. (default: {4})
            layers {list} -- Activate layers. (default: {None})
            monitor {list} -- Progress monitor. (default is a silent monitor: {DefaultMonitor})

        Raises:
            ValueError: Raise a TypeError if the reduction factor is not an integer.
        """
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.suffix = suffix + str(factor)
        self.overwrite = overwrite
        self.factor = arg_util.parse_int("factor", factor, default=4)
        self.monitor = monitor

        self.logger = log.logging.getLogger(self.__class__.__name__)

        self.reduced_layers = arg_util.parse_layers(layers)

        for layer in DtmConstants.LAYERS:
            self.logger.debug(f"Layer {layer} : {self.reduced_layers[layer]}.")

    def __process_data(
        self, i_driver: dtm_driver.DtmDriver, o_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor
    ) -> None:
        """The layer value_count is processed first. If the layer doesn't exist, create it. Set 1 in
        the cells corresponding of the cells defined in the elevation layer. Else, put 0 in it.

        Arguments:
            ind {int} -- Number of the processed file.
        """
        # Initialize output file as input file
        i_dtm_file = i_driver.dtm_file
        o_dtm_file = o_driver.dtm_file
        dtm_driver.copy_metadata(i_dtm_file, o_dtm_file)

        # adapt row and col count
        o_dtm_file.spatial_resolution_x *= self.factor
        o_dtm_file.spatial_resolution_y *= self.factor
        o_dtm_file.col_count = dtmut.estimate_col(
            left_or_west=i_dtm_file.west,
            right_or_east=i_dtm_file.east,
            spatial_resolution=o_dtm_file.spatial_resolution_x,
        )
        o_dtm_file.row_count = dtmut.estimate_row(i_dtm_file.south, i_dtm_file.north, o_dtm_file.spatial_resolution_y)
        o_driver.initialize_file()

        # History
        o_driver.dataset.history = str(i_driver.dataset.history)
        nc_util.set_history_attr(o_driver.dataset, self.__class__.__name__, i_driver.dtm_file.file_path)

        # Used for the log
        count = 0
        n = len(i_driver.get_layers()) - Counter(self.reduced_layers.values())[False]
        monitor.set_work_remaining(n)

        # If cell_value exist process it first, else create cell_value.
        i_vc = np.ones(i_driver[DtmConstants.ELEVATION_NAME].shape, dtype=np.int32) * np.invert(
            i_driver[DtmConstants.ELEVATION_NAME][:].mask
        )
        name = DtmConstants.VALUE_COUNT
        if name in i_driver.get_layers().keys():
            variable = i_driver[name]
            if self.reduced_layers[name]:
                count += 1
                log.info_progress_layer(self.logger, "layer", name, count, n)
                i_vc = variable[:].data
                self.__process_layer(i_driver, name, o_driver, i_vc)

                self.monitor.worked(1)

        for name, variable in i_driver.get_layers().items():
            if name in DtmConstants.LAYERS:
                if self.reduced_layers[name] and name != DtmConstants.VALUE_COUNT:
                    count += 1
                    log.info_progress_layer(self.logger, "layer", name, count, n)
                    self.__process_layer(i_driver, name, o_driver, i_vc)

            elif name in [DtmConstants.CDI]:
                # Copy cdi_ref, and crs
                count += 1
                log.info_progress_layer(self.logger, "dimension", name, count, n)
                o_driver.create_cdi_reference_variable(cdi_util.trim_string_array(variable[:]))

            monitor.worked(1)

    def __process_layer(
        self, i_driver: dtm_driver.DtmDriver, name: str, o_driver: dtm_driver.DtmDriver, i_vc: np.ndarray
    ) -> None:
        """Guide method ofr the layer.

        Arguments:
            name {str} -- Name of the layer.
            ind {int} -- Indice of the input file.
            i_vc {np.ndarray} -- Data layer value count.
        """

        # Initialisation
        o_driver.add_layer(name)
        o_data = o_driver[name][:].data
        i_data = i_driver[name][:].data
        m_val = i_driver[name]._FillValue

        # Reduce array
        if name == DtmConstants.ELEVATION_MIN:
            o_driver.dataset[name][:] = nb.reduce_min_max(o_data, i_data, m_val, self.factor, 0)
        elif name == DtmConstants.ELEVATION_MAX:
            o_driver.dataset[name][:] = nb.reduce_min_max(o_data, i_data, m_val, self.factor, 1)
        elif name in [DtmConstants.CDI_INDEX]:
            o_driver.dataset[name][:] = nb.reduce_cdi(o_data, i_data, i_vc, m_val, self.factor)
        elif name == DtmConstants.STDEV:
            o_driver.dataset[name][:] = nb.reduce_main(o_data, i_data, m_val, self.factor, 2, i_vc)
        elif name in [DtmConstants.VALUE_COUNT, DtmConstants.FILTERED_COUNT]:
            o_driver.dataset[name][:] = nb.reduce_value_count(o_data, i_data, m_val, self.factor)
        elif name in [DtmConstants.INTERPOLATION_FLAG]:
            o_driver.dataset[name][:] = nb.reduce_interpolation_flag(o_data, i_data, m_val, self.factor)
        else:
            o_driver.dataset[name][:] = nb.reduce_main(o_data, i_data, m_val, self.factor, 1, i_vc)

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
