#! /usr/bin/env python3
# coding: utf-8

from typing import Optional

import numpy as np

import pyat.dtm.transform.interpolation.coronis.heightmap_interpolation as heightmap_interpolation_process
from pyat.dtm.transform.interpolation.coronis.abstract_interpolation import InterpolationProcessAdapter


class CubicParameters(heightmap_interpolation_process.HeightmapParameters):
    """
    Adds the set of options to Cubic interpolation (Coronis).
        - fill_value : Value used to fill in for requested points outside of the convex hull of the input points. If not provided, the default is NaN
        - rescale : Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.
        - tolerance : Absolute/relative tolerance for gradient estimation
        - max_iters : Maximum number of iterations in gradient estimation
    """

    def __init__(self, **kwargs) -> None:
        self.fill_value: float = np.nan
        self.rescale: bool = False
        self.tolerance: float = 1e-6
        self.max_iters: int = 400

        # Init super attributes and grab values of attributes present in kwargs
        super().__init__(**kwargs)


class CubicInterpolationProcess(heightmap_interpolation_process.HeightmapInterpolationProcess):
    """
    Parameters used to invoke a cubic coronis interpolation.
    """

    def __init__(self, cubic_parameters: Optional[CubicParameters] = None, **kwargs):
        """
        Constructor.
        """
        parameters = cubic_parameters if cubic_parameters is not None else CubicParameters(**kwargs)
        super().__init__("cubic", parameters)


class CubicInterpolationProcessAdapter(InterpolationProcessAdapter):
    """
    Adapts an instance of CubicInterpolationProcess to launch a cubic interpolation on DTMs.
    """

    def __init__(self, **kwargs):
        super().__init__(CubicInterpolationProcess(**kwargs), **kwargs)
