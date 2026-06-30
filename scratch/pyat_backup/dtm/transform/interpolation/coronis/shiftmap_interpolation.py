#! /usr/bin/env python3
# coding: utf-8

from typing import Optional

import pyat.dtm.transform.interpolation.coronis.heightmap_interpolation as heightmap_interpolation_process


class ShiftmapInterpolationProcess(heightmap_interpolation_process.HeightmapInterpolationProcess):
    """
    Process used to invoke a Shiftmap Coronis interpolation.
    """

    def __init__(
        self, telea_parameters: Optional[heightmap_interpolation_process.HeightmapParameters] = None, **kwargs
    ):
        """
        Constructor.
        """
        parameters = (
            telea_parameters
            if telea_parameters is not None
            else heightmap_interpolation_process.HeightmapParameters(**kwargs)
        )
        super().__init__("shiftmap", parameters)
