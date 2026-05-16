#! /usr/bin/env python3
# coding: utf-8

from typing import Optional

import pyat.dtm.transform.interpolation.coronis.heightmap_interpolation as heightmap_interpolation_process
from pyat.dtm.transform.interpolation.coronis.abstract_interpolation import InterpolationProcessAdapter


class TeleaParameters(heightmap_interpolation_process.HeightmapParameters):
    """
    Adds the set of options to Telea interpolation (Coronis).
        - radius : Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
    """

    def __init__(self, **kwargs) -> None:
        self.radius: int = 25

        # Init super attributes and grab values of attributes present in kwargs
        super().__init__(**kwargs)


class TeleaInterpolationProcess(heightmap_interpolation_process.HeightmapInterpolationProcess):
    """
    Process used to invoke a Telea Coronis interpolation.
    """

    def __init__(self, telea_parameters: Optional[TeleaParameters] = None, **kwargs):
        """
        Constructor.
        """
        parameters = telea_parameters if telea_parameters is not None else TeleaParameters(**kwargs)
        super().__init__("telea", parameters)


class TeleaInterpolationProcessAdapter(InterpolationProcessAdapter):
    """
    Adapts an instance of TeleaInterpolationProcess to launch a linear interpolation on DTMs.
    """

    def __init__(self, **kwargs):
        super().__init__(TeleaInterpolationProcess(**kwargs), **kwargs)
