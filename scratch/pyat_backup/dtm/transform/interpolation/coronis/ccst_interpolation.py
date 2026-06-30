#! /usr/bin/env python3
# coding: utf-8

from typing import Optional

from pyat.dtm.transform.interpolation.coronis.abstract_interpolation import InterpolationProcessAdapter
from pyat.dtm.transform.interpolation.coronis.harmonic_interpolation import (
    InpainterParameters,
)
from pyat.dtm.transform.interpolation.coronis.heightmap_interpolation import (
    HeightmapInterpolationProcess,
)


class CcstParameters(InpainterParameters):
    """
    Adds the set of options for Harmonic process.
        - update_step_size : Update step size
        - term_thres : If the relative change between the inpainted elevations in the current and a previous step is smaller than this value, the optimization will stop
        - tension : Tension parameter weighting the contribution between a harmonic and a biharmonic interpolation (see the docs and the original reference for more details)
    """

    def __init__(self, **kwargs) -> None:
        self.update_step_size: float = 0.01
        self.term_thres: float = 1e-8
        self.tension: float = 0.3
        super().__init__(**kwargs)


class CcstInterpolationProcess(HeightmapInterpolationProcess):
    """
    Parameters used to invoke the Harmonic inpainter (coronis).
        - areas : KML file containing the areas that will be interpolated
        - verbose : Verbosity flag, activate it to have feedback of the current steps of the process in the command line
        - show : Show interpolation problem and results on screen
    """

    def __init__(self, ccst_parameters: Optional[CcstParameters] = None, **kwargs):
        """
        Constructor.
        """
        parameters = ccst_parameters if ccst_parameters is not None else CcstParameters(**kwargs)
        super().__init__("ccst", parameters)


class CcstInterpolationProcessAdapter(InterpolationProcessAdapter):
    """
    Adapts an instance of CcstInterpolationProcess to launch a Harmonic interpolation on DTMs.
    """

    def __init__(self, **kwargs):
        super().__init__(CcstInterpolationProcess(**kwargs), **kwargs)
