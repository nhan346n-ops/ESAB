#! /usr/bin/env python3
# coding: utf-8

from typing import Dict, List

import netCDF4 as nc
import numpy as np
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

# Constants for filter operations
EQUAL: str = "equal"
NOT_EQUAL: str = "not_equal"
LESS_THAN: str = "less_than"
MORE_THAN: str = "more_than"
BETWEEN: str = "between"
MISSING: str = "missing"
NOT_MISSING: str = "not_missing"
OPERATION = [EQUAL, NOT_EQUAL, LESS_THAN, MORE_THAN, BETWEEN, MISSING, NOT_MISSING]

CDI_LAYER = "CDI"
ALL_LAYERS = "All"

OPERATOR_AND = "AND"
OPERATOR_OR = "OR"


class ResetCellProcess:
    """Reset Cell process class. Can reset cells in function of filters parameters."""

    def __init__(
        self,
        i_paths: List,
        o_paths: List | None = None,
        suffix: str = "-zeroed",
        overwrite: bool = False,
        operator: str = OPERATOR_AND,
        filters: List | None = None,
        mask: List[str] | None = None,
        reverse_mask: bool = False,
        monitor=DefaultMonitor,
    ):
        """By default, the name of the output file is i_path + "-zeroed". No filter of zone,
        no filter by cdi, no filter by layer.

        Arguments:
            i_paths {list} -- Input file list (.nc).
            o_paths {list} -- Optional output file list (.nc). (default: {None})
            suffix {str} -- Suffix of generated output path. Used when o_paths is empty. (default: {-zeroed})
            overwrite {bool} -- true to overwrite output file if exists. (default: {False})
            filters {list} -- List of filters. (default: {None})
            mask {list} -- Mask file list. (default: {None})
            reverse_mask {bool} -- true to reverse the mask, processing data outside of the provided polygons
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
        self.reverse_mask = reverse_mask
        self.monitor = monitor

        self.operator = np.logical_or if operator == OPERATOR_OR else np.logical_and
        self.filters = []
        if filters:
            for f in filters:
                oneFilter = {}
                if "reset_layer" in f:
                    if f["reset_layer"] in DtmConstants.LAYERS or f["reset_layer"] == ALL_LAYERS:
                        oneFilter["reset_layer"] = f["reset_layer"]
                    else:
                        raise ValueError(f'The name of the layer {f["reset_layer"]} isn\'t processed.')
                else:
                    oneFilter["reset_layer"] = ALL_LAYERS

                if f["filter_layer"] in DtmConstants.LAYERS or f["filter_layer"] == CDI_LAYER:
                    oneFilter["filter_layer"] = f["filter_layer"]
                else:
                    raise ValueError(f'The name of the layer {f["filter_layer"]} isn\'t processed.')

                if f["oper"] in OPERATION:
                    oneFilter["oper"] = f["oper"]
                else:
                    raise ValueError(f'The operation {f["oper"]} isn\'t processed.')

                if "a" in f:
                    oneFilter["a"] = float(f["a"])
                if "b" in f:
                    oneFilter["b"] = float(f["b"])
                if "cdi" in f:
                    oneFilter["cdi"] = f["cdi"]

                self.filters.append(oneFilter)

        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __create_filter_mask(self, i_driver: dtm_driver.DtmDriver, layer_name: str) -> np.ndarray | None:
        """
        Create mask with all filters
        """
        # SetUp
        shape = i_driver[DtmConstants.ELEVATION_NAME].shape

        # filter mask, set to True when a given value shall be erased
        filters_mask: np.ndarray | None = None
        for f in self.filters:
            # Have to reset this layer ?
            if f["reset_layer"] in [ALL_LAYERS, layer_name]:
                if filters_mask is None:
                    filters_mask = np.full(shape, self.operator == np.logical_and, dtype=bool)

                filter_layer = f["filter_layer"]
                if filter_layer == CDI_LAYER:
                    filters_mask = self.__complete_cdi_mask(i_driver, f, filters_mask)
                else:
                    filters_mask = self.__complete_layer_mask(i_driver, f, filters_mask)

        return filters_mask

    def __complete_cdi_mask(
        self, i_driver: dtm_driver.DtmDriver, cdi_filter: Dict, filters_mask: np.ndarray
    ) -> np.ndarray:
        """
        Complete the masking of filters_mask by applying a CDI filter
        """

        if DtmConstants.CDI not in i_driver or DtmConstants.CDI_INDEX not in i_driver:
            return filters_mask

        oper = cdi_filter["oper"]
        if oper == MISSING:
            data = i_driver[DtmConstants.CDI_INDEX][:].mask
            cdi_index = -1
        elif oper == NOT_MISSING:
            data = ~i_driver[DtmConstants.CDI_INDEX][:].mask
            cdi_index = -1
        else:
            cdi = cdi_filter["cdi"]
            index_array = np.where(i_driver[DtmConstants.CDI][:] == cdi)[0]
            if len(index_array) == 0:
                self.logger.warning(f"The cdi {cdi} isn't in the input file {i_driver.get_file_path()}.")
                # CDI in not in the given file, thus no CDI filter is applied
                return filters_mask

            data = i_driver[DtmConstants.CDI_INDEX][:].data
            cdi_index = index_array[0]

        return self.__apply_operator(
            oper=oper,
            current_mask=filters_mask,
            data_to_filter=data,
            value1=int(cdi_index),
        )

    def __complete_layer_mask(
        self, i_driver: dtm_driver.DtmDriver, numeric_filter: Dict, filters_mask: np.ndarray
    ) -> np.ndarray:
        """
        Complete the masking of filters_mask by applying the specified filter
        """
        filter_layer = numeric_filter["filter_layer"]
        oper = numeric_filter["oper"]
        a = numeric_filter["a"] if "a" in numeric_filter else 0.0
        b = numeric_filter["b"] if "b" in numeric_filter else 0.0
        if oper == MISSING:
            data = i_driver[filter_layer][:].mask
        elif oper == NOT_MISSING:
            data = ~i_driver[filter_layer][:].mask
        else:
            data = i_driver[filter_layer][:].data

        return self.__apply_operator(oper=oper, current_mask=filters_mask, data_to_filter=data, value1=a, value2=b)

    def __apply_operator(
        self, oper: str, current_mask: np.ndarray, data_to_filter: np.ndarray, value1, value2=0
    ) -> np.ndarray:
        """
        Apply the operator "oper" to mask some cells on array data_to_filter.
        Then apply the logical And/Or on this resulting array and the current_mask
        """
        if oper == EQUAL:
            data_to_filter = data_to_filter == value1
        elif oper == NOT_EQUAL:
            data_to_filter = data_to_filter != value1
        elif oper == LESS_THAN:
            data_to_filter = data_to_filter <= value1
        elif oper == MORE_THAN:
            data_to_filter = data_to_filter >= value1
        elif oper == BETWEEN:
            data_to_filter = (data_to_filter >= value1) & (data_to_filter <= value2)

        return self.operator(current_mask, data_to_filter)

    def __create_mask(self, geo_mask: np.ndarray | None, filters_mask: np.ndarray | None) -> np.ndarray | None:
        """
        Combine the geographic mask and the filter one
        """
        # No mask
        if filters_mask is None:
            return None

        # Initially, the mask consists of the geographic mask if present
        mask: np.ndarray | None = geo_mask

        # Also apply the filtering mask if present
        if mask is None:
            mask = filters_mask
        else:
            mask = mask & filters_mask

        if np.ma.is_masked(mask):
            mask = np.ma.filled(mask, fill_value=False)
        mask = np.array(mask, dtype=np.uint8)
        return mask

    def __process_data(
        self, i_driver: dtm_driver.DtmDriver, o_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor
    ) -> None:
        """
        Browse all layers and apply the reset cell process on them.
        """

        # Initialize output file
        process_util.initialize_output_file(i_driver, o_driver, process_name=self.__class__.__name__)

        o_file = o_driver.dataset

        # Used for the log
        count = 0
        n = len(i_driver.get_layers())
        monitor.set_work_remaining(n + 1)

        # Geographic mask.
        geo_mask: np.ndarray | None = None
        if self.mask_files:
            geo_mask = compute_geo_mask_from_dtm(i_driver.get_file_path(), self.mask_files, self.reverse_mask)
            # convert to boolean array (geo_mask is set to 1)
            geo_mask = geo_mask > 0

        for layer_name, variable in i_driver.get_layers().items():
            if layer_name in DtmConstants.LAYERS:
                count += 1
                log.info_progress_layer(self.logger, "layer", layer_name, count, n)

                # Create variable in the o_files[ind].
                o_file.createVariable(
                    layer_name, variable.datatype, variable.dimensions, compression=nc_util.DEFAULT_COMPRESSION_LIB
                )

                # Computing the mask to apply to the layer
                mask: np.ndarray | None = None

                if geo_mask is None and not self.filters:
                    # No filter at all. Reset all cells
                    mask = np.full(variable.shape, 1, dtype=np.uint8)
                if geo_mask is not None and not self.filters:
                    # No filter. Applying only the geographic mask
                    mask = np.array(geo_mask, dtype=np.uint8)
                elif self.filters:
                    filters_mask = self.__create_filter_mask(i_driver, layer_name)
                    if filters_mask is not None:
                        # Filter exists. Applying it and geographic mask (if any)
                        mask = self.__create_mask(geo_mask, filters_mask)

                if mask is None:
                    # No mask for this layer. Keeps all cells
                    mask = np.full(variable.shape, 0, dtype=np.uint8)

                self.__process_layer(i_driver, layer_name, o_file, mask)

            elif layer_name == DtmConstants.CDI:
                # Copy cdi layer, it will be cleaned later
                count += 1
                log.info_progress_layer(self.logger, "layer", layer_name, count, n)
                o_driver.create_cdi_reference_variable(cdi_util.trim_string_array(variable[:]))

            monitor.worked(1)

        # now once everything is processed, clean cdi
        cdi_util.clean_cdi(o_file)
        monitor.worked(1)

    def __process_layer(
        self, i_driver: dtm_driver.DtmDriver, layer_name: str, o_file: nc.Dataset, mask: np.ndarray
    ) -> None:
        """
        For each True in mask array, reset the corresponding cell in the specified layer.
        """
        # copy variable attributes all at once via dictionary
        o_file[layer_name].setncatts(i_driver[layer_name].__dict__)

        # Initialisation
        o_data = o_file[layer_name][:].data
        i_data = i_driver[layer_name][:].data
        m_val = i_driver[layer_name]._FillValue

        # Reset selected cells
        o_file[layer_name][:] = nb.reset_layer(o_data, i_data, m_val, mask)

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
