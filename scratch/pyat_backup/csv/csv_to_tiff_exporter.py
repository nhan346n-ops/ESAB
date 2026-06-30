#! /usr/bin/env python3
# coding: utf-8

import datetime
import os
from os import path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from osgeo import osr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.csv.csv_constants as CSV
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.tiff.tiff_gridder import TiffGridder


class CsvToTiffExporter:
    """
    Utility class to convert CSV files as Tiff
    """

    def __init__(
        self,
        i_paths: List[str],
        o_paths: Union[List[str], str],
        indexes: Dict[str, str],
        target_resolution: float,
        coord: Dict[str, float],
        target_fillvalue: float = None,
        target_compression: bool = True,
        headers_types: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
        delimiter: str = ";",
        decimal_point: str = ".",
        skip_rows: int = 0,
        depth_sign: float = 1.0,
        spatial_reference: str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor.
        """
        self.logger = log.logging.getLogger(self.__class__.__name__)
        self.i_paths = i_paths

        if isinstance(o_paths, str):
            folder = o_paths
            if not path.exists(folder):
                os.makedirs(folder)
            # Create output name from the input with the tiff extension.
            self.o_paths = [os.path.join(folder, path[: path.rfind(".")] + ".tif") for path in i_paths]
        else:
            self.o_paths = o_paths
        arg_util.check_output_paths(self.i_paths, self.o_paths)

        self.overwrite = overwrite

        self.indexes = {key: int(value) for (key, value) in indexes.items()}
        if not all(column in self.indexes for column in (CSV.COL_LONGITUDE, CSV.COL_LATITUDE)):
            raise AttributeError(f"Columns {CSV.COL_LONGITUDE} and {CSV.COL_LATITUDE} are mandatory.")

        self.headers_types: Dict[str, str] = headers_types if not headers_types is None else {}
        if self.headers_types is None or len(self.headers_types) == 0:
            self.headers_types = {column_name: "float" for column_name in indexes.keys()}

        if not all(column_name in self.headers_types for column_name in indexes.keys()):
            raise AttributeError("Argument headers_types mismatches argument indexes")

        self.delimiter = delimiter
        self.decimal_point = decimal_point
        self.skip_rows = arg_util.parse_int("skip_rows", skip_rows, 0)
        self.depth_sign = arg_util.parse_float("depth_sign", depth_sign, 1.0)
        self.spatial_reference = osr.SpatialReference()
        self.spatial_reference.ImportFromProj4(spatial_reference)
        self.spatial_resolution = arg_util.parse_float("target_resolution", target_resolution, 1.0 / 3600.00)

        self.fill_value = target_fillvalue
        self.compression = target_compression

        self.geobox = arg_util.parse_geobox("coord", coord)
        self.geobox.spatial_reference = self.spatial_reference

        self.monitor = monitor

    def __export_data(self, csv_file: str, column_name: str, tiff_path: str, monitor: ProgressMonitor) -> None:
        """
        Launch the export of the file.
        Raised exception : IOError when error occurs while parsing the file
        """
        monitor.set_work_remaining(100)

        # Create a DtmBuilder with a DtmDriver opened in write mode
        tiff_gridder = TiffGridder(tiff_path, self.geobox, self.spatial_resolution, monitor)

        col_type = self.headers_types[column_name]
        tiff_gridder.initialize_tiff_file(dtype=float if col_type == "float" else int, fill_value=self.fill_value)
        monitor.worked(10)

        # Process all lines by chunck
        line_count = 0
        for lines in self.__open_csv(csv_file, column_name):
            # First, compute columns and rows index
            columns, rows = tiff_gridder.project_coords(
                lines[CSV.COL_LONGITUDE][:].to_numpy(), lines[CSV.COL_LATITUDE][:].to_numpy()
            )
            # Then, process values
            tiff_gridder.grid_keep_last(columns, rows, lines[column_name][:].to_numpy(), self.depth_sign)

            line_count = line_count + lines.shape[0]
            self.logger.info(f"Number of processed lines : {line_count}")
        monitor.worked(80)

        tiff_gridder.finalize_tiff(compression=self.compression)

    def __open_csv(self, csv_file: str, column_name):
        nb_cols = max(self.indexes.values()) + 1
        names = ["COL_" + str(index) for index in range(nb_cols)]
        dtype = {}
        for layer, column in self.indexes.items():
            names[column] = layer
            if layer in [CSV.COL_LATITUDE, CSV.COL_LONGITUDE]:
                dtype[layer] = np.float64
            elif self.headers_types[layer] == "float":
                dtype[layer] = np.float32
            elif self.headers_types[layer] == "int":
                dtype[layer] = np.int32
            else:
                dtype[layer] = np.dtype(str)

        usecols = [self.indexes[col] for col in [CSV.COL_LATITUDE, CSV.COL_LONGITUDE, column_name]]

        return pd.read_csv(
            csv_file,
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

    def __call__(self) -> None:
        """Run method."""
        begin = datetime.datetime.now()

        self.monitor.set_work_remaining(len(self.i_paths))
        file_in_error = []
        for csv_file, tiff_file in zip(self.i_paths, self.o_paths):
            sub_monitor = self.monitor.split(1)
            try:
                self.logger.info(f"Starting to convert {csv_file} to {tiff_file}")
                now = datetime.datetime.now()
                col_value = "Value"
                if self.headers_types[col_value] in ["float", "int"]:
                    if not self.overwrite and os.path.exists(tiff_file):
                        self.logger.warning("File exists and overwrite is not allowed. Convertion aborted.")
                    else:
                        self.__export_data(csv_file, col_value, tiff_file, sub_monitor)
                else:
                    self.logger.warning(
                        f"Expecting float or int type for values, not {self.headers_types[col_value]}. Convertion aborted."
                    )

                self.logger.info(f"End of conversion for {csv_file} : {datetime.datetime.now() - now} time elapsed\n")

            except Exception as e:
                file_in_error.append(csv_file)
                self.logger.error(f"An exception was thrown : {str(e)}", exc_info=True, stack_info=True)
            finally:
                sub_monitor.done()

        self.monitor.done()
        process_util.log_result(self.logger, begin, file_in_error)
