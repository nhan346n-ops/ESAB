#! /usr/bin/env python3
# coding: utf-8

from typing import Optional

from pyat.dtm.transform.interpolation.coronis.abstract_interpolation import InterpolationProcessAdapter
from pyat.dtm.transform.interpolation.coronis.rbf_interpolation import RbfParameters

import pyat.dtm.transform.interpolation.coronis.heightmap_interpolation as heightmap_interpolation_process


class PurbfParameters(RbfParameters):
    """
    Parameters extending RbfParameters, used to invoke a Partition of Unity Radial Basis Function (Coronis).
        - pu_overlap : Overlap factor between circles in neighboring sub-domains in the partition. The radius of a QuadTree cell, computed as half its diagonal, is enlarged by this factor
        - pu_min_point_in_cell : Minimum number of points in a QuadTree cell
        - pu_min_cell_size_percent : Minimum cell size, specified as a percentage [0..1] of the max(width, height) of the query domain
        - pu_overlap_increment : If, after creating the QuadTree, a cell contains less than pu_min_point_in_cell, the radius will be iteratively incremented until this condition is satisfied. This parameter specifies how much the radius of a cell increments at each iteration
    """

    def __init__(self, **kwargs) -> None:
        self.pu_overlap: float = 0.25
        self.pu_min_point_in_cell: int = 1000
        self.pu_min_cell_size_percent: float = 0.005
        self.pu_overlap_increment: float = 0.001

        # Init super attributes and grab values of attributes present in kwargs
        super().__init__(**kwargs)


class PurbfInterpolationProcess(heightmap_interpolation_process.HeightmapInterpolationProcess):
    """
    Process used to invoke a Partition of Unity Radial Basis Function interpolation (Coronis).
    """

    def __init__(self, purbf_parameters: Optional[PurbfParameters] = None, **kwargs):
        """
        Constructor.
        """
        parameters = purbf_parameters if purbf_parameters is not None else PurbfParameters(**kwargs)
        super().__init__("purbf", parameters)


class PurbfInterpolationProcessAdapter(InterpolationProcessAdapter):
    """
    Adapts an instance of PurbfInterpolationProcess to launch a purbf interpolation on DTMs.
    """

    def __init__(self, **kwargs):
        super().__init__(PurbfInterpolationProcess(**kwargs), **kwargs)
