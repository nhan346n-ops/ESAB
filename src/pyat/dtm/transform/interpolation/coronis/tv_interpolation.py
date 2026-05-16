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


class TVParameters(InpainterParameters):
    """
    Same parameters than InpainterParameters adding :
        - update_step_size : Update step size
        - term_thres : If the relative change between the inpainted elevations in the current and a previous step is smaller than this value, the optimization will stop
        - epsilon : A small value to be added when computing the norm of the gradients during optimization, to avoid a division by zero
    """

    def __init__(self, **kwargs) -> None:
        self.update_step_size: float = 0.225
        self.term_thres: float = 1e-5
        self.epsilon: float = 1.0
        super().__init__(**kwargs)


class TVInterpolationProcess(HeightmapInterpolationProcess):
    """
    Process used to invoke the Harmonic inpainter (coronis).
    """

    def __init__(self, tv_parameters: Optional[TVParameters] = None, **kwargs):
        """
        Constructor.
        """
        parameters = tv_parameters if tv_parameters is not None else TVParameters(**kwargs)
        super().__init__("tv", parameters)


class TvInterpolationProcessAdapter(InterpolationProcessAdapter):
    """
    Adapts an instance of TVInterpolationProcess to launch a TV interpolation on DTMs.
    """

    def __init__(self, **kwargs):
        super().__init__(TVInterpolationProcess(**kwargs), **kwargs)
