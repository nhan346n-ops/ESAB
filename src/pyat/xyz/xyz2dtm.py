#! /usr/bin/env python3
# coding: utf-8

import os
from typing import Dict, List, Optional

from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.csv.csv_constants as CSV
import pyat.dtm.dtm_standard_constants as DTM
from pyat.dtm.convert.gridded_csv_to_dtm import GriddedCsvToDtm
from pyat.dtm.cdi.set_cdi_process import SetCdiProcess


class Xyz2Dtm:
    """Utility class to export an XYZ file as a dtm (netcdf4 format)."""

    def __init__(
        self,
        i_paths: List[str],
        o_paths: Optional[List[str]] = None,
        target_resolution: float = 1.0 / 3600.0,
        coord: Optional[Dict[str, float]] = None,
        cdi: Optional[str] = None,
        overwrite: bool = False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        self.i_paths = i_paths
        self.cdi = cdi
        self.target_resolution = target_resolution
        self.coord = coord
        # Create output name from the input with the nc extension if necessary.
        self.o_paths = (
            o_paths if not o_paths is None else [path[: path.rfind(".")] + DTM.EXTENSION for path in self.i_paths]
        )
        self.overwrite = overwrite
        self.monitor = monitor

    def __call__(self):
        """
        Run method.
        Perform the conversion and the set the CDI
        """
        to_dtm_o_paths = self.o_paths
        # Temp DTM file if set CDI required
        if not self.cdi is None:
            to_dtm_o_paths = [path[: path.rfind(".")] + "_without_cdi." + DTM.EXTENSION for path in self.i_paths]

        exporter = GriddedCsvToDtm(
            i_paths=self.i_paths,
            indexes={CSV.COL_LONGITUDE: 0, CSV.COL_LATITUDE: 1, CSV.COL_ELEVATION: 2},
            target_resolution=self.target_resolution,
            coord=self.coord,
            o_paths=to_dtm_o_paths,
            overwrite=self.overwrite,
            monitor=self.monitor if self.cdi is None else self.monitor.split(50),
            recompute_geobox=True,
            auto_rounding_arcmin=False,
        )
        exporter()

        if not self.cdi is None:
            try:
                set_cdi_process = SetCdiProcess(
                    i_paths=to_dtm_o_paths,
                    cdi=self.cdi,
                    o_paths=self.o_paths,
                    suffix="",
                    overwrite=self.overwrite,
                    monitor=self.monitor,
                )
                set_cdi_process()
            finally:
                for tmp_file in to_dtm_o_paths:
                    os.remove(tmp_file)
