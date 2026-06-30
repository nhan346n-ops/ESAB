"""Module for Ping detection signal support
A 2D matrix of signals indexed on ping detection dimension

based on a xarray dataset and xsf variables (dataArray can be added or substracted)

It could be obtained from a variable of a xsf file,
* If variable is 2 on ping detection, value is returned
* If variable is of dimension 1 with a ping dimension, values are extended to detection dimension
* If variable is of dimension 2 with a ping antenna dimension, values are resampled with the use of detection_rx_antenna variable
* If variable is of dimension 2 with a ping beam dimension, values are resampled with the use of detection_rx_beam variable
"""
from typing import List, Callable, Mapping, Any

import numpy as np
import xarray as xr

from pyat.sonarscope.common.xsf_utils import get_detection_antenna_coords, get_detection_tx_beam_coords, \
    get_detection_count
from pyat.sonarscope.model.signal.abstract_signal import BaseSignal, PVariable

from pyat.sonarscope.model.constants import DEFAULT_BEAM_GROUP_IDENT
from pyat.sonarscope.model.sounder_format_dictionary.sounder_format_factories import get_variables_dictionary
from pyat.utils.exceptions.exception_list import UnexpectedError
from pyat.xsf.xsf_driver import XsfDriver

from pyat.sonarscope.model.constants import VariableKeys as KeyDef, VariableDim as DimDef
from pyat.sonarscope.model.sounder_format_dictionary.common_dictionary import VariablesContainer


class PingDetectionSignal(BaseSignal):
    def __init__(self, xsf_dataset: XsfDriver, beam_group=DEFAULT_BEAM_GROUP_IDENT):
        super().__init__(xsf_dataset=xsf_dataset)
        dictionary = get_variables_dictionary(xsf_dataset)
        self.default_variables_book = dictionary.PingDetectionVariables(beam_group=beam_group)
        self.computed_variables_book = dictionary.ComputedPingDetectionVariables()
        self.ping_time_book = dictionary.PingTimeVariables(beam_group=beam_group)

    def read_new_variables(
        self, variable_to_load: List[str], reductor_function: Callable, ignore_unknown_variables: bool
    ):
        new_variables = {}
        for key in variable_to_load:
            v = None
            if key in self.default_variables_book:
                v = self._load_ping_detection_variable(
                    key=key,
                    container=self.default_variables_book,
                    reductor_function=reductor_function,
                    ignore_unknown_variables=ignore_unknown_variables,
                )
            if key in self.computed_variables_book:
                v = self._load_ping_detection_variable(
                    key=key,
                    container=self.computed_variables_book,
                    reductor_function=reductor_function,
                    ignore_unknown_variables=ignore_unknown_variables,
                )
            if key in self.ping_time_book:
                v = self._load_ping_detection_variable(
                    key=key,
                    container=self.ping_time_book,
                    reductor_function=reductor_function,
                    ignore_unknown_variables=ignore_unknown_variables,
                )
            if v is not None:
                new_variables[key] = v
        return new_variables

    def get_mandatory_variable(self) -> List[str]:
        return []

    def read_coordinates(self) -> Mapping[Any, Any]:
        attributes = self.ping_time_book.get_attributes(key=KeyDef.PING_TIME, nc_dataset=self.xsf_dataset.dataset)
        data = self.ping_time_book.get_values(key=KeyDef.PING_TIME, nc_dataset=self.xsf_dataset.dataset)

        ping_time_variable = xr.Variable(data=data, dims=DimDef.PING_DIM, attrs=attributes)
        detection_z_dims = self.default_variables_book.get_dimensions(
            key=KeyDef.DETECTION_BACKSCATTER, nc_dataset=self.xsf_dataset.dataset
        )

        detection_variable = xr.Variable(
            data=np.arange(0, detection_z_dims[DimDef.DETECTION_DIM].size), dims=DimDef.DETECTION_DIM
        )
        return {DimDef.PING_DIM: ping_time_variable, DimDef.DETECTION_DIM: detection_variable}

    def _load_ping_detection_variable(
        self,
        key: str,
        container: VariablesContainer,
        reductor_function=np.nanmean,
        ignore_unknown_variables=False,
    ):
        expected_dimensions = [DimDef.PING_DIM, DimDef.DETECTION_DIM]
        try:
            dimensions = container.get_dimensions(key=key, nc_dataset=self.xsf_dataset.dataset)
            data = container.get_values(key=key, nc_dataset=self.xsf_dataset.dataset)
            attributes = container.get_attributes(key=key, nc_dataset=self.xsf_dataset.dataset)

            def check_dim(name):
                """Check that dimension 'name' is defined"""
                if name not in dimensions:
                    # Variable does not have a ping time dimensions,
                    self.logger.error(f"Variable {key} does not have a {name} dimension")
                    raise UnexpectedError(f"Variable {key} does not have a {name} dimension")

            check_dim(DimDef.PING_DIM)
            if DimDef.DETECTION_DIM in dimensions:
                ping_count = dimensions[DimDef.PING_DIM].size
                detection_count = dimensions[DimDef.DETECTION_DIM].size
                # reduce dimension if needed
                axis = []  # list of dimension different from time dimension
                i = 0
                for dim in dimensions:
                    if dim not in expected_dimensions:
                        axis.append(i)
                    i += 1
                if data is None:
                    raise UnexpectedError(f"Variable {key} does not exist")
                if data.size == 0:  # one of the variable dimension is empty (typically no WC)
                    data = np.full(shape=(ping_count, detection_count), fill_value=np.NAN, dtype=np.float64)
                elif len(axis) > 0:
                    data = reductor_function(data, axis=tuple(axis))
                # end of dimension reduction
            elif DimDef.RX_ANTENNA_DIM in dimensions:
                data = data[get_detection_antenna_coords(self.xsf_dataset.dataset)]
            elif DimDef.TX_BEAM_DIM in dimensions:
                data = data[get_detection_tx_beam_coords(self.xsf_dataset.dataset)]
            elif len(dimensions) == 1: #ping_time variable
                data = np.repeat(data[:, None], get_detection_count(self.xsf_dataset.dataset), axis=1)

            # create a temporary storage for variables data
            return PVariable(data=data, dimensions=expected_dimensions, attributes=attributes)

        except IndexError as e:  # raise when a variable is missing in dataset
            if ignore_unknown_variables:
                self.logger.warning(f"variable {key} is missing in dataset")
                return PVariable(
                    data=np.full(shape=(ping_count, detection_count), fill_value=np.NAN, dtype=np.float64),
                    dimensions=expected_dimensions,
                    attributes={},
                )

            else:
                self.logger.error(f"variable {key} is missing in dataset", e)
                raise e
