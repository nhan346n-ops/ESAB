#! /usr/bin/env python3
# coding: utf-8

from typing import Optional

import pyat.dtm.transform.interpolation.coronis.heightmap_interpolation as heightmap_interpolation_process
from pyat.dtm.transform.interpolation.coronis.abstract_interpolation import InterpolationProcessAdapter


class NavierStokesParameters(heightmap_interpolation_process.HeightmapParameters):
    """
    Adds the set of options to navier-stokes interpolation (Coronis).
        - radius : Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
    """

    def __init__(self, **kwargs) -> None:
        self.radius: int = 25

        # Init super attributes and grab values of attributes present in kwargs
        super().__init__(**kwargs)


class NavierStokesInterpolationProcess(heightmap_interpolation_process.HeightmapInterpolationProcess):
    """
    Process used to invoke a navier-stokes Coronis interpolation.
    """

    def __init__(self, navier_stokes_parameters: Optional[NavierStokesParameters] = None, **kwargs):
        """
        Constructor.
        """
        parameters = (
            navier_stokes_parameters if navier_stokes_parameters is not None else NavierStokesParameters(**kwargs)
        )
        super().__init__("navier-stokes", parameters)


class NavierStokesInterpolationProcessAdapter(InterpolationProcessAdapter):
    """
    Adapts an instance of NavierStokesInterpolationProcess to launch a linear interpolation on DTMs.
    """

    def __init__(self, **kwargs):
        super().__init__(NavierStokesInterpolationProcess(**kwargs), **kwargs)
