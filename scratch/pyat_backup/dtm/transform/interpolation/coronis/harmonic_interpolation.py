#! /usr/bin/env python3
# coding: utf-8

from typing import Optional

import pyat.dtm.transform.interpolation.coronis.heightmap_interpolation as heightmap_interpolation_process
from pyat.dtm.transform.interpolation.coronis.abstract_interpolation import InterpolationProcessAdapter


class InpainterParameters(heightmap_interpolation_process.HeightmapParameters):
    """
    Adds the set of options common to all FD-PDE inpainting methods (Coronis).
        - term_check_iters : Number of iterations in the optimization after which we will check if the relative tolerance is below the threshold
        - max_iters : Maximum number of iterations in the optimization
        - relaxation : Set to >1 to perform over-relaxation at each iteration
        - backend : The desired backend where computations should take place. If the requested backend is not available in the machine, will fallback to 'cpu'. Options: 'cpu', 'gpu'
        - ti_arch : When '--backend' is 'gpu', this parameter sets the actual GPU architecture to use. Available options: 'cpu' (i.e., runs the GPU implementation in the CPU), 'gpu', 'cuda', 'vulkan', 'metal'
        - print_progress_iters : If '--print_progress True', the optimization progress will be shown after this number of iterations
        - mgs_levels : Levels of the Multi-grid solver. I.e., number of levels of detail used in the solving pyramid
        - mgs_min_res : If during the construction of the pyramid of the Multi-Grid Solver one of the dimensions of the grid drops below this size, the pyramid construction will stop at that level
        - init_with : Initialize the unknown values to inpaint using a simple interpolation function. If using a MGS, this will be used with the lowest level on the pyramid. Available initializers: 'nearest' (default), 'linear', 'cubic', 'harmonic'
        - convolver : The convolution method to use. Available: 'opencv' (default),'scipy-signal', 'scipy-ndimage', 'masked', 'masked-parallel'
        - debug_dir : If set, debugging information will be stored in this directory (useful to visualize the inpainting progress)
    """

    def __init__(self, **kwargs) -> None:
        self.term_check_iters: int = 1000
        self.max_iters: int = 1000000
        self.relaxation: float = 0.0
        self.backend: str = "cpu"
        self.ti_arch: str = "gpu"
        self.print_progress_iters: int = 1000
        self.mgs_levels: int = 1
        self.mgs_min_res: int = 100
        self.init_with: str = "nearest"
        self.convolver: str = "opencv"
        self.debug_dir: Optional[str] = None

        # Init super attributes and grab values of attributes present in kwargs
        super().__init__(**kwargs)


class HarmonicParameters(InpainterParameters):
    """
    Adds the set of options for Harmonic process.
        - update_step_size : Update step size
        - term_thres : If the relative change between the inpainted elevations in the current and a previous step is smaller than this value, the optimization will stop
    """

    def __init__(self, **kwargs) -> None:
        self.update_step_size = 0.2
        self.term_thres = 1e-5
        super().__init__(**kwargs)


class HarmonicInterpolationProcess(heightmap_interpolation_process.HeightmapInterpolationProcess):
    """
    Process used to invoke the Harmonic inpainter (coronis).
    """

    def __init__(self, harmonic_parameters: Optional[HarmonicParameters] = None, **kwargs):
        """
        Constructor.
        """
        parameters = harmonic_parameters if harmonic_parameters is not None else HarmonicParameters(**kwargs)
        super().__init__("harmonic", parameters)


class HarmonicInterpolationProcessAdapter(InterpolationProcessAdapter):
    """
    Adapts an instance of HarmonicInterpolationProcess to launch a Harmonic interpolation on DTMs.
    """

    def __init__(self, **kwargs):
        super().__init__(HarmonicInterpolationProcess(**kwargs), **kwargs)
