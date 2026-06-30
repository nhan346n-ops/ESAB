#! /usr/bin/env python3
# coding: utf-8

from abc import ABC
from typing import List, Optional

from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

from pyat.dtm.transform.interpolation.coronis.heightmap_interpolation import (
    HeightmapInterpolationProcess,
)
from pyat.dtm.transform.interpolation.coronis.interpolation import interpolate_dtms


class InterpolationProcessAdapter(ABC):
    """
    Callable used by application utils to perform an interpolation on DTMs.
    """

    def __init__(
        self,
        interpolation_process_delegate: HeightmapInterpolationProcess,
        i_paths: List,
        o_paths: List,
        areas: Optional[str] = None,
        cdi_interpolation_algo: str = "closest_neighbor",  # or most_common_neighbor
        overwrite: bool = False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        interpolate_dtms(
            i_paths=i_paths,
            o_paths=o_paths,
            interpolation_algo=interpolation_process_delegate.interpolates,
            cdi_interpolation_algo=cdi_interpolation_algo,
            overwrite=overwrite,
            areas=areas,
            monitor=monitor,
        )
