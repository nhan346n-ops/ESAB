"""Module for Ping time signal support
A ping time signal is a 1D values indexed on ping time

Ping time signal can be seen as a cache file, it is based on a xarray dataset and xsf variables (dataArray can be added or substracted)

It could be obtained from a variable of a xsf file,
* If variable is 2 or more dimension, a sampling method is defined to reduce data to a single value per ping
* If variable is of dimension 1 and ping time indexed it is returned
* If variable is not ping time indexed but time indexed, resampling method can be provided


"""

from typing import Any, Callable, List, Mapping

import numpy as np
from scipy.interpolate import interp1d
from sonar_netcdf.sonar_groups import PositionSubGroup, RuntimeGrp

from pyat.sonarscope.model.constants import DEFAULT_BEAM_GROUP_IDENT
from pyat.sonarscope.model.constants import VariableDim as DimDef
from pyat.sonarscope.model.constants import VariableKeys as KeyDef
from pyat.sonarscope.model.signal.abstract_signal import BaseSignal, PVariable
from pyat.sonarscope.model.sounder_format_dictionary.common_dictionary import (
    VariablesContainer,
)
from pyat.sonarscope.model.sounder_format_dictionary.sounder_format_factories import (
    get_variables_dictionary,
)
from pyat.utils.exceptions.exception_list import UnexpectedError
from pyat.xsf.xsf_driver import XsfDriver


class PingSignal(BaseSignal):

    def __init__(self, xsf_dataset: XsfDriver, beam_group=DEFAULT_BEAM_GROUP_IDENT):
        super().__init__(xsf_dataset=xsf_dataset)
        dictionary = get_variables_dictionary(xsf_dataset)
        self.default_variables_book = dictionary.PingTimeVariables(beam_group=beam_group)
        self.runtime_variables_book = dictionary.RunTimeVariables()
        self.position_variables_book = dictionary.PositionVariables(root_dataset=xsf_dataset.dataset)
        self.computed_variables = dictionary.ComputedPingVariables()

    def get_mandatory_variable(self):
        # Add ping_time (mandatory)
        return [KeyDef.PING_TIME]

    def _get_raw_ping_time_data(self):
        """Load ping_time variable (without ping time cf decoding"""
        data = self.default_variables_book.get_values(key=KeyDef.PING_TIME, nc_dataset=self.xsf_dataset.dataset)
        return data

    def read_coordinates(self) -> Mapping[Any, Any]:
        """Return a mapping filled with coordinates variables"""
        # return None, variable ping_time is automatically added in classical variable list
        return None

    def _load_ping_time_variable(
        self,
        key: str,
        ping_time_data: np.array,
        container: VariablesContainer,
        reductor_function=np.nanmean,
        ignore_unknown_variables=False,
    ):
        try:
            dimensions = container.get_dimensions(key=key, nc_dataset=self.xsf_dataset.dataset)
            data = container.get_values(key=key, nc_dataset=self.xsf_dataset.dataset)
            attributes = container.get_attributes(key=key, nc_dataset=self.xsf_dataset.dataset)

            if DimDef.PING_DIM not in dimensions:
                # Variable does not have a ping time dimensions,
                self.logger.error(f"Variable {key} does not have a {DimDef.PING_DIM} dimension")
                raise UnexpectedError(f"Variable {key} does not have a {DimDef.PING_DIM} dimension")

            axis = []  # list of dimension different from time dimension
            i = 0
            for dim in dimensions:
                if dim != DimDef.PING_DIM:
                    axis.append(i)
                i += 1
            if data.size == 0:  # one of the variable dimension is empty (typically no WC)
                data = np.full_like(ping_time_data, fill_value=np.NAN, dtype=np.float64)
            elif len(axis) > 0:
                data = reductor_function(data, axis=tuple(axis))

            # dimension is now just "ping_time"
            # create a temporary storage for variables data
            return PVariable(data=data, dimensions=DimDef.PING_DIM, attributes=attributes)
        except IndexError as e:  # raise when a variable is missing in dataset
            if ignore_unknown_variables:
                self.logger.warning(f"variable {key} is missing in dataset")
                return PVariable(
                    data=np.full_like(ping_time_data, fill_value=np.nan), dimensions="ping_time", attributes={}
                )

            else:
                self.logger.error(f"variable {key} is missing in dataset", e)
                raise e

    def _load_interpol_variable(
        self,
        key: str,
        time_data: np.array,
        ping_time_data: np.array,
        time_dimension_name: str,
        container: VariablesContainer,
        ignore_unknown_variables=False,
        extrapolation_direction: str | None = None,
    ):
        """
        Interpolate a variable on ping_time.

        Parameters
        ----------
        key : Variable to interpolate
        time_data : source times
        ping_time_data : destination times
        time_dimension_name : name of time dimension
        container : VariableContainer
        ignore_unknown_variables : bool
        extrapolation_direction : None or an element of {"forward", "backward", "both"}

        """
        try:
            dimensions = container.get_dimensions(key=key, nc_dataset=self.xsf_dataset.dataset)
            data = container.get_values(key=key, nc_dataset=self.xsf_dataset.dataset)
            attributes = container.get_attributes(key=key, nc_dataset=self.xsf_dataset.dataset)

            if time_dimension_name not in dimensions:
                # Variable does not have a ping time dimensions,
                self.logger.error(f"Variable {key} does not have a {time_dimension_name} dimension")
                raise UnexpectedError(f"Variable {key} does not have a {time_dimension_name} dimension")

            if len(dimensions) > 1:
                raise UnexpectedError(f"Only 1D time indexed dataset are supported")

            fill_value = container.get_fill_values(key, nc_dataset=self.xsf_dataset.dataset)
            # using data without mask for interpolation/extrapolation allows extrapolation of fillValue
            raw_data = np.array(data)
            extrapolation_value = fill_value
            if extrapolation_direction == "forward":
                extrapolation_value = (fill_value, raw_data[-1])
            elif extrapolation_direction == "backward":
                extrapolation_value = (raw_data[0], fill_value)
            elif extrapolation_direction == "both":
                extrapolation_value = (raw_data[0], raw_data[-1])

            # now expand data to match ping_time
            interpolator = interp1d(
                time_data, raw_data, kind="previous", bounds_error=False, fill_value=extrapolation_value
            )

            values = interpolator(ping_time_data.data)
            # we need to keep type as they were defined (possible since interpolation is taking the previous value)
            values = values.astype(data.dtype)
            # dimension is now just "ping_time"
            # create a temporary storage for variables data
            return PVariable(data=values, dimensions="ping_time", attributes=attributes)
        except IndexError as e:  # raise when a variable is missing in dataset
            if ignore_unknown_variables:
                self.logger.warning(f"variable {key} is missing in dataset")
                return PVariable(
                    data=np.full_like(ping_time_data, fill_value=np.nan), dimensions="ping_time", attributes={}
                )

            else:
                self.logger.error(f"variable {key} is missing in dataset", e)
                raise e
        return None

    def read_dimensions_values(self):
        # load or retrieve ping time dimension
        ping_time_data = self._get_raw_ping_time_data()
        return {"ping_time_data": ping_time_data}

    def read_new_variables(
        self, variable_to_load: List[str], reductor_function: Callable, ignore_unknown_variables: bool
    ):
        runtime_time_data = None
        position_time_data = None
        ping_time_data = self._get_raw_ping_time_data()
        new_variables = {}
        for key in variable_to_load:
            v = None

            if key in self.runtime_variables_book:
                # if we do not have runtime time variable, read it
                if runtime_time_data is None:
                    runtime_time_data = self.runtime_variables_book.get_values(
                        key=KeyDef.RUNTIME_TIME, nc_dataset=self.xsf_dataset.dataset
                    )
                # now read variable
                v = self._load_interpol_variable(
                    key=key,
                    time_data=runtime_time_data,
                    ping_time_data=ping_time_data,
                    time_dimension_name=RuntimeGrp.RUNTIME_COUNT_DIM_NAME,
                    ignore_unknown_variables=ignore_unknown_variables,
                    container=self.runtime_variables_book,
                    extrapolation_direction="both",
                )
            elif key in self.default_variables_book:
                v = self._load_ping_time_variable(
                    key=key,
                    ping_time_data=ping_time_data,
                    container=self.default_variables_book,
                    reductor_function=reductor_function,
                    ignore_unknown_variables=ignore_unknown_variables,
                )
            elif key in self.position_variables_book:
                if position_time_data is None:
                    position_time_data = self.position_variables_book.get_values(
                        key=KeyDef.POSITION_SENSOR_TIME, nc_dataset=self.xsf_dataset.dataset
                    )
                # now read variable
                v = self._load_interpol_variable(
                    key=key,
                    time_data=position_time_data,
                    ping_time_data=ping_time_data,
                    time_dimension_name=PositionSubGroup.TIME_DIM_NAME,
                    ignore_unknown_variables=ignore_unknown_variables,
                    container=self.position_variables_book,
                )
            elif key in self.computed_variables:
                v = self._load_ping_time_variable(
                    key=key,
                    ping_time_data=ping_time_data,
                    container=self.computed_variables,
                    reductor_function=reductor_function,
                    ignore_unknown_variables=ignore_unknown_variables,
                )
            if v is not None:
                new_variables[key] = v
        return new_variables
