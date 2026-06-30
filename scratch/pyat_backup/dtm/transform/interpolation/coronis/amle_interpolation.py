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


class AMLEParameters(InpainterParameters):
    """
    Same parameters than InpainterParameters adding :
        - update_step_size : Update step size
        - term_thres : If the relative change between the inpainted elevations in the current and a previous step is smaller than this value, the optimization will stop
        - convolve_in_1d : Perform 1D convolutions instead of using the 2D convolution indicated in --convolver
    """

    def __init__(self, **kwargs) -> None:
        self.update_step_size: float = 0.01
        self.term_thres: float = 1e-7
        self.convolve_in_1d: bool = True
        super().__init__(**kwargs)


class AMLEInterpolationProcess(HeightmapInterpolationProcess):
    """
    Process used to invoke the Absolutely Minimizing Lipschitz Extension (AMLE) inpainter (coronis).
        - areas : KML file containing the areas that will be interpolated
        - verbose : Verbosity flag, activate it to have feedback of the current steps of the process in the command line
        - show : Show interpolation problem and results on screen
    """

    def __init__(self, amle_parameters: Optional[AMLEParameters] = None, **kwargs):
        """
        Constructor.
        """
        parameters = amle_parameters if amle_parameters is not None else AMLEParameters(**kwargs)
        super().__init__("amle", parameters)


class AmleInterpolationProcessAdapter(InterpolationProcessAdapter):
    """
    Adapts an instance of AMLEInterpolationProcess to launch a AMLE interpolation on DTMs.
    """

    def __init__(self, **kwargs):
        super().__init__(AMLEInterpolationProcess(**kwargs), **kwargs)
