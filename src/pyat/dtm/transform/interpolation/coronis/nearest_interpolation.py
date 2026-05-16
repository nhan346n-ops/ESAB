#! /usr/bin/env python3
# coding: utf-8

from typing import Optional

import pyat.dtm.transform.interpolation.coronis.heightmap_interpolation as heightmap_interpolation_process
from pyat.dtm.transform.interpolation.coronis.abstract_interpolation import InterpolationProcessAdapter


class NearestParameters(heightmap_interpolation_process.HeightmapParameters):
    """
    Adds the set of options to nearest interpolation (Coronis).
        - rescale : Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.
    """

    def __init__(self, **kwargs) -> None:
        self.rescale: bool = False

        # Init super attributes and grab values of attributes present in kwargs
        super().__init__(**kwargs)


class NearestInterpolationProcess(heightmap_interpolation_process.HeightmapInterpolationProcess):
    """
    Process used to invoke a linear coronis interpolation.
    """

    def __init__(self, linear_parameters: Optional[NearestParameters] = None, **kwargs):
        """
        Constructor.
        """
        parameters = linear_parameters if linear_parameters is not None else NearestParameters(**kwargs)
        super().__init__("nearest", parameters)


class NearestInterpolationProcessAdapter(InterpolationProcessAdapter):
    """
    Adapts an instance of NearestInterpolationProcess to launch a nearest interpolation on DTMs.
    """

    def __init__(self, **kwargs):
        super().__init__(NearestInterpolationProcess(**kwargs), **kwargs)
