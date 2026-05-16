#! /usr/bin/env python3
# coding: utf-8

from typing import Optional

import pyat.dtm.transform.interpolation.coronis.heightmap_interpolation as heightmap_interpolation_process
from pyat.dtm.transform.interpolation.coronis.abstract_interpolation import InterpolationProcessAdapter


class RbfParameters(heightmap_interpolation_process.HeightmapParameters):
    """
    Parameters used to invoke a Radial Basis Function interpolation (Coronis).
        - query_block_size : Apply the interpolant using maximum this number of points at a time to avoid large memory consumption
        - rbf_distance_type: Distance type. Available: euclidean (default), haversine, vincenty
        - rbf_type : RBF type. Available: linear, cubic, quintic, gaussian, multiquadric, green, regularized, tension, thinplate, wendland
        - rbf_epsilon : Epsilon parameter of the RBF. Please check each RBF documentation for its meaning. Required just for the following RBF types: gaussian, multiquadric, regularized, tension, wendland
        - rbf_regularization : Regularization scalar to use while creating the RBF interpolant (optional)
        - rbf_polynomial_degree : Degree of the global polynomial fit used in the RBF formulation. Valid: -1 (no polynomial fit), 0 (constant), 1 (linear), 2 (quadric), 3 (cubic)
    """

    def __init__(self, **kwargs) -> None:
        self.query_block_size: int = 1000
        self.rbf_distance_type: str = "euclidean"
        self.rbf_type: str = "thinplate"
        self.rbf_epsilon: float = 1.0
        self.rbf_regularization: float = 0.0
        self.rbf_polynomial_degree: int = 1

        # Init super attributes and grab values of attributes present in kwargs
        super().__init__(**kwargs)


class RbfInterpolationProcess(heightmap_interpolation_process.HeightmapInterpolationProcess):
    """
    Process used to invoke a Radial Basis Function interpolation (Coronis).
    """

    def __init__(self, rbf_parameters: Optional[RbfParameters] = None, **kwargs):
        """
        Constructor.
        """
        parameters = rbf_parameters if rbf_parameters is not None else RbfParameters(**kwargs)
        super().__init__("rbf", parameters)


class RbfInterpolationProcessAdapter(InterpolationProcessAdapter):
    """
    Adapts an instance of RbfInterpolationProcess to launch a rbf interpolation on DTMs.
    """

    def __init__(self, **kwargs):
        super().__init__(RbfInterpolationProcess(**kwargs), **kwargs)
