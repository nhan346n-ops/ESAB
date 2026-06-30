"""
Interpolation module for 2D grids with NaN values.

This module provides various interpolation algorithms to fill
missing values (NaN) in 2D grids along the horizontal axis.
"""

from enum import Enum
from typing import Protocol, Union

import numpy as np
import xarray as xr


class InterpolationMethod(str, Enum):
    """Enumeration of available interpolation methods."""

    LINEAR = "linear"


class InterpolationFunction(Protocol):
    """Protocol defining the signature of interpolation functions."""

    def __call__(self, grid: np.ndarray) -> np.ndarray:
        """
        Interpolate NaN values in a 2D grid.

        Args:
            grid: 2D array of shape (height, length) containing NaN values

        Returns:
            Grid with interpolated NaN values
        """


def _interpolate_linear(grid: np.ndarray) -> np.ndarray:
    """
    Linear interpolation of NaN values along the horizontal axis.

    Args:
        grid: 2D array of shape (height, length) containing NaN values

    Returns:
        Grid with NaN values linearly interpolated
    """
    data_array = xr.DataArray(grid, dims=("height", "length"))
    interpolated = data_array.interpolate_na(dim="length", method="linear")
    return interpolated.values


# Registry of interpolation functions
_INTERPOLATION_FUNCTIONS: dict[InterpolationMethod, InterpolationFunction] = {
    InterpolationMethod.LINEAR: _interpolate_linear,
}


def interpolate(grid: np.ndarray, method: Union[InterpolationMethod, str] = InterpolationMethod.LINEAR) -> np.ndarray:
    """
    Interpolate NaN values in a 2D grid using the chosen method.

    This function is the main entry point of the module. It allows
    applying different interpolation algorithms to a 2D grid
    containing missing values (NaN).

    Args:
        grid: 2D NumPy array of shape (height, length) containing NaN values
        method: Interpolation method to use. Can be:
            - A member of the InterpolationMethod enumeration
            - A string corresponding to the method name (e.g., "linear")

    Returns:
        Grid with NaN values interpolated according to the chosen method

    Raises:
        ValueError: If the specified method is not supported
    """
    # Convert string to Enum if necessary
    if isinstance(method, str):
        method = InterpolationMethod(method)

    # Retrieve the corresponding interpolation function
    if method not in _INTERPOLATION_FUNCTIONS:
        raise ValueError(f"Method {method.value} not implemented")

    interpolation_func = _INTERPOLATION_FUNCTIONS[method]
    return interpolation_func(grid)
