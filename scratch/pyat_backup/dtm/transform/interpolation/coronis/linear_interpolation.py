#! /usr/bin/env python3
# coding: utf-8

from typing import Optional

import pyat.dtm.transform.interpolation.coronis.heightmap_interpolation as heightmap_interpolation_process
from pyat.dtm.transform.interpolation.coronis.abstract_interpolation import InterpolationProcessAdapter


class LinearParameters(heightmap_interpolation_process.HeightmapParameters):
    """
    Adds the set of options to linear interpolation (Coronis).
        - rescale : Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.
    """

    def __init__(self, **kwargs) -> None:
        self.rescale: bool = False

        # Init super attributes and grab values of attributes present in kwargs
        super().__init__(**kwargs)


class LinearInterpolationProcess(heightmap_interpolation_process.HeightmapInterpolationProcess):
    """
    Process used to invoke a linear coronis interpolation.
    """

    def __init__(self, linear_parameters: Optional[LinearParameters] = None, **kwargs):
        """
        Constructor.
        """
        parameters = linear_parameters if linear_parameters is not None else LinearParameters(**kwargs)
        super().__init__("linear", parameters)


class LinearInterpolationProcessAdapter(InterpolationProcessAdapter):
    """
    Adapts an instance of LinearInterpolationProcess to launch a linear interpolation on DTMs.
    """

    def __init__(self, **kwargs):
        super().__init__(LinearInterpolationProcess(**kwargs), **kwargs)
