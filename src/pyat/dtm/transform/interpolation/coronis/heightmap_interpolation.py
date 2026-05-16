#! /usr/bin/env python3
# coding: utf-8

from abc import ABC
from typing import List, Optional

import heightmap_interpolation.apps.interpolate_netcdf4 as heightmap_interpolation
import numpy as np

import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.utils.pyat_logger as log


class HeightmapParameters:
    """
    Common parameters of Coronis interpolation process.
        - areas : KML file containing the areas that will be interpolated
        - verbose : Verbosity flag, activate it to have feedback of the current steps of the process in the command line
        - show : Show interpolation problem and results on screen
    """

    def __init__(self, **kwargs) -> None:
        self.areas: Optional[str] = None
        self.verbose: Optional[bool] = None
        self.show: Optional[bool] = None

        # Dynamically change value of attribut with kwargs
        for key, value in kwargs.items():
            if key in self.__dict__:
                self.__setattr__(key, value)


class HeightmapInterpolationProcess(ABC):
    """
    Abstract class representing a Coronis heightmap_interpolation process.
    """

    def __init__(self, scattered_method: str, parameters: HeightmapParameters):
        """
        Constructor.
        """
        self.scattered_method = scattered_method
        self.parameters = parameters
        self.scattered_method_args: List[str] = []
        self._format_scattered_method_args(**parameters.__dict__)
        self.logger = log.logging.getLogger(__file__)

    def _format_scattered_method_args(self, **kwargs) -> None:
        for key, key_value in kwargs.items():
            if key in ["areas", "verbose", "show"]:
                continue
            value = key_value[0] if isinstance(key_value, tuple) else key_value
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        self.scattered_method_args.append("--" + key)
                elif isinstance(value, float):
                    if value != np.nan:
                        self.scattered_method_args.extend(["--" + key, str(value)])
                else:
                    self.scattered_method_args.extend(["--" + key, str(value)])

    def interpolates(self, i_path: str, o_path: str, kml_path: str | None) -> None:
        """
        Invoke the Coronis process to perform the interpolation of the input DTM into the o_path
        """
        # Output file
        arg_to_parse = ["-o", o_path]

        # Geo mask
        if kml_path is not None:
            arg_to_parse.extend(["--areas", kml_path])

        # Feedback parameters
        if self.parameters.verbose:
            arg_to_parse.append("-v")
        if self.parameters.show:
            arg_to_parse.append("-s")

        # Specify layer names
        arg_to_parse.extend(["--elevation_var", DtmConstants.ELEVATION_NAME])
        # arg_to_parse.extend(["--interpolation_flag_var", DtmConstants.INTERPOLATION_FLAG])

        # Interpolation parameters
        arg_to_parse.append(self.scattered_method)
        arg_to_parse.extend(self.scattered_method_args)

        # File to interpolate
        arg_to_parse.append(i_path)

        if self.parameters.verbose:
            self.logger.info(f"Launching heightmap_interpolation with parameters : {' '.join(arg_to_parse)}")

        # Call interpolation process
        heightmap_interpolation.interpolate(heightmap_interpolation.parse_args(arg_to_parse))
