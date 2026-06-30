#! /usr/bin/env python3
# coding: utf-8

import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pygws.service.execution_context as exec_ctx
from osgeo import osr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.csv.csv_constants as CSV
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.utils.exceptions.exception_list import ProcessingError


class GeoboxEvaluator:
    """
    Read the whole CSV file to extract the Geobox.
    Geobox is composed of the maximum and minimum values on each axis
    """

    def __init__(
        self,
        i_paths: List[str],
        indexes: Optional[Dict[str, int]] = None,
        delimiter: str = ";",
        decimal_point: str = ".",
        skip_rows: int = 0,
        spatial_reference: str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        evaluate_spatial_resolution: bool = False,
        spatial_resolution=0,
        auto_rounding=False,  # if lat/lon enable auto rounding to nearest arcmin
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor.
        """
        self.i_paths = i_paths

        self.delimiter = delimiter
        self.decimal_point = decimal_point
        self.skip_rows = arg_util.parse_int("skip_rows", skip_rows, 0)

        self.spatial_reference = osr.SpatialReference()
        self.spatial_reference.ImportFromProj4(spatial_reference)
        self.evaluate_spatial_resolution = evaluate_spatial_resolution
        self.auto_rounding = auto_rounding
        self.spatial_resolution = spatial_resolution

        # XYZ by default
        if indexes is None:
            self.indexes = {CSV.COL_LONGITUDE: 0, CSV.COL_LATITUDE: 1, CSV.COL_ELEVATION: 2}
        else:
            self.indexes = {key: int(value) for (key, value) in indexes.items()}
            if not all(column in self.indexes for column in (CSV.COL_LONGITUDE, CSV.COL_LATITUDE)):
                raise AttributeError(f"Columns {CSV.COL_LONGITUDE} and {CSV.COL_LATITUDE} are mandatory.")

        # Prefer to use RSocket monitor if available
        self.monitor = monitor
        if exec_ctx.get_root_progress_monitor() is not None:
            self.monitor = exec_ctx.get_root_progress_monitor()

        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __call__(self) -> Dict | None:
        """Run method."""
        begin = datetime.datetime.now()
        file_in_error: List[str] = []

        self.monitor.begin_task("Evalutating the geobox", 100)

        self._estimate_extent_and_resolution()
        self.monitor.worked(50)

        self._apply_rounding()
        self.monitor.worked(40)

        process_util.log_result(self.logger, begin, file_in_error)
        self.monitor.done()

        return self._report_result()

    def _estimate_extent_and_resolution(self) -> None:
        """
        Read the CSV file and estimate the geobox and spatial resolution
        Initialize attributes self.spatial_resolution and self.geobox
        Raised exception : IOError when error occurs while parsing the file
        """
        self.logger.info("Opening CSV file, extracting extent......")

        line_count = 0
        first_chunk = True
        geobox_builder = arg_util.GeoBoxBuilder(self.spatial_reference)
        for csv_path in self.i_paths:
            for lines in self.__open_csv(csv_path):
                if lines.shape[0] < 2:
                    raise ProcessingError("Bad CSV file : Not enough row")
                line_count = line_count + lines.shape[0]
                self.logger.info(f"Number of lines processed : {line_count}")

                lons = lines[CSV.COL_LONGITUDE][:].to_numpy()
                lats = lines[CSV.COL_LATITUDE][:].to_numpy()
                geobox_builder.add_lons_lats(lons, lats)

                if self.evaluate_spatial_resolution and first_chunk:
                    # Compute the spatial_resolution on the first chunk
                    deltaLon = abs(lons[0:-1] - lons[1:])
                    deltaLon[deltaLon == 0] = np.nan
                    deltaLat = abs(lats[0:-1] - lats[1:])
                    deltaLat[deltaLat == 0] = np.nan

                    self.spatial_resolution = np.nanmin([np.nanmin(deltaLon), np.nanmin(deltaLat)])
                    if self.spatial_resolution == 0:
                        raise ProcessingError("Cannot estimate spatial resolution")
            first_chunk = False

        self.geobox = geobox_builder.build()
        self.geobox.fix_if_180th_meridian()
        self.logger.info(f"Number of lines in the csv file : {line_count}")

    def _apply_rounding(self):
        """Round to the lowest and highest arc min if applicable"""
        if self.auto_rounding:
            self.geobox.expand_to_arcmin()

    def __open_csv(self, csv_path: str):
        nb_cols = max(self.indexes.values()) + 1
        names = ["COL_" + str(index) for index in range(nb_cols)]
        names[self.indexes[CSV.COL_LONGITUDE]] = CSV.COL_LONGITUDE
        names[self.indexes[CSV.COL_LATITUDE]] = CSV.COL_LATITUDE

        dtype = {layer: np.dtype(str) for layer in names}
        dtype[CSV.COL_LONGITUDE] = np.float64
        dtype[CSV.COL_LATITUDE] = np.float64

        usecols = [self.indexes[CSV.COL_LONGITUDE], self.indexes[CSV.COL_LATITUDE]]

        return pd.read_csv(
            csv_path,
            chunksize=1_000_000,
            sep=r"\s+" if self.delimiter == "…" else self.delimiter,
            decimal=self.decimal_point,
            names=names,
            usecols=usecols,
            dtype=dtype,
            header=None,
            skiprows=self.skip_rows,
            index_col=False,
        )

    def _report_result(self) -> Dict | None:
        """
        Prepare the result for the report JSON file
        """

        result = {
            "top": self.geobox.upper,
            "bottom": self.geobox.lower,
            "left": self.geobox.left,
            "right": self.geobox.right,
        }

        if self.evaluate_spatial_resolution:
            result["spatial_resolution"] = self.spatial_resolution

        self.logger.info("Result")
        self.logger.info(str(result))

        # Using rsocket (if present) to send the result
        rsocket_msg_emitter = exec_ctx.get_rsocket_msg_emitter()
        if rsocket_msg_emitter is not None:
            rsocket_msg_emitter.emit_map_of_double(result)
            return None
        else:
            return {"result": result}


class ExtentEvaluator(GeoboxEvaluator):
    """
    Read the whole CSV file to extract the extent and the resolution
    The extent represents the envelope of all cells in the DTM

    First, evaluates the geobox of the CSV and then compute the extent of the DTM's grid according to the "pos_in_cell" argument
    """

    def __init__(
        self,
        i_paths: List[str],
        indexes: Optional[Dict[str, int]] = None,
        delimiter: str = ";",
        decimal_point: str = ".",
        skip_rows: int = 0,
        spatial_reference: str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        pos_in_cell: str = "center",
        evaluate_spatial_resolution=True,
        spatial_resolution=0,
        auto_rounding=False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        super().__init__(
            i_paths=i_paths,
            indexes=indexes,
            delimiter=delimiter,
            decimal_point=decimal_point,
            skip_rows=skip_rows,
            spatial_reference=spatial_reference,
            evaluate_spatial_resolution=evaluate_spatial_resolution,
            spatial_resolution=spatial_resolution,
            auto_rounding=auto_rounding,
            monitor=monitor,
        )
        self.pos_in_cell = pos_in_cell

    def _adapt_geobox_to_position_in_cell(self) -> None:
        """
        Adapt the extents of coods with the specified pos_in_cell attribute and infer the geobox of the DTM
        """
        colCount = round(self.geobox.get_delta_x() / self.spatial_resolution) + 1
        lineCount = round(self.geobox.get_delta_y() / self.spatial_resolution) + 1
        if self.pos_in_cell == "upper-left":
            self.geobox.right = self.geobox.left + colCount * self.spatial_resolution
            self.geobox.lower = self.geobox.upper - lineCount * self.spatial_resolution
        elif self.pos_in_cell == "upper-right":
            self.geobox.left = self.geobox.right - colCount * self.spatial_resolution
            self.geobox.lower = self.geobox.upper - lineCount * self.spatial_resolution
        elif self.pos_in_cell == "lower-left":
            self.geobox.right = self.geobox.left + colCount * self.spatial_resolution
            self.geobox.upper = self.geobox.lower + lineCount * self.spatial_resolution
        elif self.pos_in_cell == "lower-right":
            self.geobox.left = self.geobox.right - colCount * self.spatial_resolution
            self.geobox.upper = self.geobox.lower + lineCount * self.spatial_resolution
        else:
            # center
            self.geobox.left = self.geobox.left - 0.5 * self.spatial_resolution
            self.geobox.right = self.geobox.left + colCount * self.spatial_resolution
            self.geobox.upper = self.geobox.upper + 0.5 * self.spatial_resolution
            self.geobox.lower = self.geobox.upper - lineCount * self.spatial_resolution

        self.geobox.normalize_degrees()

    def __call__(self) -> Dict | None:
        """Run method."""
        begin = datetime.datetime.now()
        file_in_error: List[str] = []

        self.monitor.begin_task("Evaluating the extent", 100)
        self.monitor.worked(10)

        self._estimate_extent_and_resolution()
        self.monitor.worked(40)

        self._adapt_geobox_to_position_in_cell()
        self.monitor.worked(40)

        process_util.log_result(self.logger, begin, file_in_error)
        self.monitor.worked(10)

        return self._report_result()


class ExtentEvaluatorAuto(ExtentEvaluator):
    """Evaluate extent of a dtm with spatial resolution defined and some auto expand to integer number of minute"""

    def __init__(
        self,
        i_paths: List[str],
        spatial_resolution: float = np.float64(3.75 / 3600),
        auto_rounding: bool = True,  # auto round bounding box to highest number of min
        indexes: Optional[Dict[str, int]] = None,
        delimiter: str = ";",
        decimal_point: str = ".",
        skip_rows: int = 0,
        spatial_reference: str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        pos_in_cell: str = "center",
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        super().__init__(
            i_paths=i_paths,
            indexes=indexes,
            delimiter=delimiter,
            decimal_point=decimal_point,
            skip_rows=skip_rows,
            spatial_reference=spatial_reference,
            evaluate_spatial_resolution=False,
            spatial_resolution=spatial_resolution,
            auto_rounding=auto_rounding,
            monitor=monitor,
        )
        self.pos_in_cell = pos_in_cell

    def __call__(self) -> None:
        """Run method."""
        begin = datetime.datetime.now()
        file_in_error: List[str] = []

        self.logger.info("Evaluating the extent")
        self._estimate_extent_and_resolution()
        self._adapt_geobox_to_position_in_cell()
        self._apply_rounding()

        process_util.log_result(self.logger, begin, file_in_error)
        return self._report_result()
