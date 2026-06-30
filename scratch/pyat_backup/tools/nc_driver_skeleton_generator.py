#! /usr/bin/env python3
# coding: utf-8

#
# Module to generate the code of Netcdf4 Driver (Mbg...)
#

import ntpath

import netCDF4 as nc
import numpy as np

# Global parameters
nc_file = "E:/temp/filtri_Pr_023_tide_COR__MAREE.mbg"
driver_name = "Mbg"
layer_prefix = "mb"  # None if variables does start by any prefix

BYTE_MAX = (1 << 7) - 1
SHORT_MAX = (1 << 15) - 1
INT_MAX = (1 << 31) - 1


def camel_to_snake(s: str) -> str:
    return "".join(["_" + c.lower() if c.isupper() else c for c in s]).lstrip("_")


def variable_to_layer(variable_name: str) -> str:
    if layer_prefix is not None and variable_name.startswith(layer_prefix):
        variable_name = variable_name[len(layer_prefix) :]
    return camel_to_snake(variable_name).upper()


def print_plain_variable_accessor(variable_name: str) -> None:
    layer_name = variable_to_layer(variable_name)
    print(
        f"""

    def read_{layer_name.lower()}(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        \"""
        return the data of the variable {variable_name } as a numpy array.
        Generated with {ntpath.basename(__file__)}
        \"""
        return self.__read_layer({layer_name}, from_index, to_index) """
    )


def print_variable_accessor(variable: nc.Variable, numpy_dtype) -> None:
    layer_name = variable_to_layer(variable.name)
    if "scale_factor" in variable.ncattrs() and isinstance(variable.scale_factor, float):
        print(
            f"""

    def read_{layer_name.lower()}(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        \"""
        return the data of the variable {variable.name} as a numpy array.
        Generated with {ntpath.basename(__file__)}
        \"""
        return self.__read_layer_as({layer_name}, np.{numpy_dtype.__name__}, float, from_index, to_index)"""
        )

        return

    print(
        f"""

    def read_{layer_name.lower()}(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        \"""
        return the data of the variable {variable.name} as a numpy array.
        Generated with {ntpath.basename(__file__)}
        \"""
        return self.__read_layer_as({layer_name}, np.{numpy_dtype.__name__}, None, from_index, to_index)"""
    )


with nc.Dataset(nc_file) as dataset:

    print(
        """
#! /usr/bin/env python3
# coding: utf-8

import netCDF4 as nc
import numpy as np

"""
    )
    print("# Dimensions")
    dimensions = list(dataset.dimensions)
    dimensions.sort()
    for variable_name in dimensions:
        print(f'{variable_to_layer(variable_name)} = "{variable_name}"')

    print("# Layers")
    variables = list(dataset.variables.keys())
    variables.sort()
    for variable_name in variables:
        print(f'{variable_to_layer(variable_name)} = "{variable_name}"')

    # Class definition
    print(
        f"""

class {driver_name}Driver():
    \"""
    Driver class of {driver_name} files to encapsulate the Netcdf4 access
    \"""
    @property
    def dataset(self) -> nc.Dataset:
    return self._dataset
    """
    )

    # Layers accessors
    for variable_name in variables:
        variable = dataset.variables[variable_name]

        # Check max value of signed variable
        if "valid_maximum" in variable.ncattrs():
            if variable.dtype == np.short and variable.valid_maximum > SHORT_MAX:
                # short variable have to be process as unsigned short
                print_variable_accessor(variable, np.ushort)
                continue
            if variable.dtype == int and variable.valid_maximum > INT_MAX:
                # int variable have to be process as unsigned int
                print_variable_accessor(variable, np.uintc)
                continue
            if variable.dtype == np.byte and variable.valid_maximum > BYTE_MAX:
                # byte variable have to be process as unsigned byte
                print_variable_accessor(variable, np.ubyte)
                continue

        if variable.dtype == np.dtype("|S1"):
            print_variable_accessor(variable, np.int8)
            continue

        print_plain_variable_accessor(variable_name)

    # Factorized methods
    print(
        f"""
    def __apply_offset_and_scale(self, variable : nc.Variable, data : np.ndarray) -> None:
        \"""
        Apply the offset and scale if present
        Generated with {ntpath.basename(__file__)}
        \"""
        if "scale_factor" in variable.ncattrs():
            np.multiply(data, variable.scale_factor, out=data)
        if "add_offset" in variable.ncattrs():
            np.add(data, variable.add_offset, out=data)

    def __read_layer(self, layer_name:str, from_index: int = None, to_index: int = None) -> np.ndarray:
        \"""
        return the data of the specified variable as a numpy array.
        Generated with {ntpath.basename(__file__)}
        \"""
        if from_index is None and to_index is None:
            return self.dataset[layer_name][:]
        if from_index is not None and to_index is None:
            return self.dataset[layer_name][from_index:]
        if from_index is None and to_index is not None:
            return self.dataset[layer_name][:to_index]
        return self.dataset[layer_name][from_index:to_index]

    def __read_layer_as(
        self, layer_name: str, from_numpy_dtype=np.int8, to_numpy_dtype=None, from_index: int = None, to_index: int = None
    ) -> np.ndarray:
        \"""
        return the data of the specified variable as a numpy array of a specific type.
        Generated with {ntpath.basename(__file__)}
        \"""
        variable = self.dataset[layer_name]
        variable.set_auto_maskandscale(False)

        data = self.__read_layer(layer_name, from_index, to_index)
        result = np.frombuffer(data, dtype=from_numpy_dtype).reshape(data.shape)

        if to_numpy_dtype is not None:
            result = result.astype(to_numpy_dtype)

        self.__apply_offset_and_scale(variable, result)

        return result
        """
    )
