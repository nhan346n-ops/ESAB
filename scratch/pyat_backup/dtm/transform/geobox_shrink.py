#! /usr/bin/env python3
# coding: utf-8

from typing import Tuple

import numpy as np
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.pyat_logger as log
from pyat.dtm.transform.update_boundingbox import ReprojectProcess
from pyat.utils.argument_utils import Geobox


class ShrinkProcess:
    """Class used for the shrink process of a dtm file."""

    def __init__(
        self,
        i_paths: list,
        o_paths: list = None,
        realign: bool = False,
        overwrite: bool = False,
        monitor=DefaultMonitor,
    ):
        """ """
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.realign = realign
        self.overwrite = overwrite
        self.monitor = monitor

        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __process_data(
        self, i_driver: dtm_driver.DtmDriver, o_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor
    ) -> None:
        """Process the shrink process on the specified DTM"""
        i_dtm_file = i_driver.dtm_file

        self.logger.info(
            f"Input geobox {Geobox(i_dtm_file.north, i_dtm_file.south, i_dtm_file.west, i_dtm_file.east, i_dtm_file.spatial_reference):DMS}"
        )

        col_from, col_to, row_from, row_to = self.__compute_shrinked_index(i_driver)
        geobox = self.__compute_shrinked_geobox(i_driver, col_from, col_to, row_from, row_to)
        # Something to shrink ?
        if col_to - col_from + 1 == i_dtm_file.col_count and row_to - row_from + 1 == i_dtm_file.row_count:
            self.logger.warning("No empty cells detected at the border.")
        else:
            self.logger.info(
                f"Shrinking to columns [from {col_from} to {col_to}] and rows [from {row_from} to {row_to}]"
            )

        self.logger.info(f"Reprojection DTM to {geobox:DMS}")

        reprojectProcess = ReprojectProcess(
            i_paths=[i_driver.get_file_path()],
            o_paths=[o_driver.get_file_path()],
            coord=geobox.to_dict(),
            suffix="",
            overwrite=self.overwrite,
        )
        reprojectProcess.process_data(i_driver, o_driver, monitor)

    def __compute_shrinked_index(self, i_driver: dtm_driver.DtmDriver) -> Tuple[int, int, int, int]:
        """Determines the slice of index containing elevation"""

        # Computes the first and last index of the column/row where the elevation is present
        i_elev = i_driver[DtmConstants.ELEVATION_NAME][:]
        nb_value_per_col = i_elev.count(axis=0)
        non_empty_cols = np.argwhere(nb_value_per_col > 0)
        nb_value_per_row = i_elev.count(axis=1)
        non_empty_rows = np.argwhere(nb_value_per_row > 0)
        col_from = non_empty_cols[0][0]
        col_to = non_empty_cols[-1][0]
        row_from = non_empty_rows[0][0]
        row_to = non_empty_rows[-1][0]

        return col_from, col_to, row_from, row_to

    def __compute_shrinked_geobox(
        self, i_driver: dtm_driver.DtmDriver, col_from: int, col_to: int, row_from: int, row_to: int
    ) -> Geobox:
        """Use the grid mapping variables to create the shrinking geobox"""
        i_dtm_file = i_driver.dtm_file
        spatial_reference = i_dtm_file.spatial_reference
        geobox = Geobox(
            left=float(i_driver.get_x_axis()[col_from]) - 0.5 * i_dtm_file.spatial_resolution_x,
            right=float(i_driver.get_x_axis()[col_to]) + 0.5 * i_dtm_file.spatial_resolution_x,
            lower=float(i_driver.get_y_axis()[row_from]) - 0.5 * i_dtm_file.spatial_resolution_y,
            upper=float(i_driver.get_y_axis()[row_to]) + 0.5 * i_dtm_file.spatial_resolution_y,
            spatial_reference=spatial_reference,
        )

        # Realign bounding box
        if self.realign:
            if spatial_reference.IsProjected():
                # Align Geobox on a multiple of spatial_resolution
                geobox.realign(i_dtm_file.spatial_resolution_x, i_dtm_file.spatial_resolution_y)
            else:
                # Align Geobox on arcmin
                geobox.realign()

        return geobox

    def __call__(self) -> None:
        process_util.process_each_input_dtm_to_output_dtm(
            self.__class__.__name__,
            self.i_paths,
            self.__process_data,
            self.logger,
            self.o_paths,
            "shrink",
            self.overwrite,
            self.monitor,
        )
