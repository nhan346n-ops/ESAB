#! /usr/bin/env python3
# coding: utf-8

import datetime
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from osgeo import osr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.csv.csv_constants as CSV
import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DTM
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.dtm.dtm_gridder import DtmGridder
from pyat.dtm.cdi.cdi_layer_util import check_undefined_cdi
from pyat.function.evaluate_csv_grid import ExtentEvaluatorAuto


class GriddedCsvToDtm:
    """
    Utility class to convert CSV files as DTM (netcdf4 format)
    """

    def __init__(
        self,
        i_paths: List[str],
        indexes: Dict[str, int],
        headers_types: Optional[Dict[str, str]] = None,
        o_paths: Optional[List[str]] = None,
        overwrite: bool = False,
        delimiter: str = ";",
        decimal_point: str = ".",
        skip_rows: int = 0,
        depth_sign: float = 1.0,
        spatial_reference: str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        target_resolution: float = 1.0 / 3600.0,
        coord: Optional[Dict[str, float]] = None,
        title: str = "",
        institution: str = "",
        source: str = "",
        references: str = "",
        comment: str = "",
        recompute_geobox=False,
        auto_rounding_arcmin=False,
        pos_in_cell="center",
        allow_undefined_cdi=False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor.
        :param : i_paths : path of the input file to convert
        :param : o_paths : resulting dtm file path

        Raised exceptions :
            - FileNotFoundError when emoFilePath does not exist
            - PermissionError when emoFilePath is not readable or dtmFilePath is not writable
            - IOError when emoFilePath is not a suitable emo file
        """
        self.logger = log.logging.getLogger(self.__class__.__name__)

        if isinstance(i_paths, list):
            self.i_paths = i_paths
        else:
            self.i_paths = [i_paths]

        if o_paths is None or len(o_paths) == 0:
            # Create output name from the input with the nc extension.
            self.o_paths = [path[: path.rfind(".")] + DTM.EXTENSION for path in self.i_paths]
        elif isinstance(o_paths, list):
            self.o_paths = o_paths
        else:
            self.o_paths = [o_paths]
        if len(self.o_paths) != len(self.i_paths):
            raise AttributeError("Number of Output/Input paths must be the same.")

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

        if not coord is None:
            self.geobox = arg_util.parse_geobox("coord", coord)
        else:
            self.geobox = arg_util.Geobox(0, 0, 0, 0)
            self.recompute_geobox = True
        self.geobox.spatial_reference = self.spatial_reference

        self.title = title
        self.institution = institution
        self.source = source
        self.references = references
        self.comment = comment
        self.auto_rounding_arcmin = auto_rounding_arcmin
        self.recompute_geobox = recompute_geobox
        self.pos_in_cell = pos_in_cell

        self.allow_undefined_cdi = allow_undefined_cdi

        self.monitor = monitor

    def __export_data(self, csv_file: str, o_dtm_driver: dtm_driver.DtmDriver) -> None:
        """
        Launch the export of the file.
        Raised exception : IOError when error occurs while parsing the file
        """
        self.logger.info(f"Spatial resolution is {self.spatial_resolution}")
        self.logger.info(str(self.geobox))

        # Create a DtmBuilder with a DtmDriver opened in write mode. average_elevations=False => keep last elevation
        dtm_gridder = DtmGridder(
            o_dtm_driver, self.geobox, self.spatial_resolution, depth_factor=self.depth_sign, average_elevations=False
        )

        # Add layer provided by CSV
        for column_name in self.indexes.keys():
            if column_name in CSV.COL_TO_LAYER:
                dtm_gridder.add_layer(CSV.COL_TO_LAYER[column_name])
            elif column_name not in [CSV.COL_LATITUDE, CSV.COL_LONGITUDE, CSV.COL_ELEVATION]:
                data_type = self.headers_types[column_name]
                dtm_gridder.add_layer(column_name, data_type, CSV.COL_DEFAULT_VALUES[data_type])

        dtm_gridder.initialize_dtm_file(
            history=f"Created from {os.path.basename(csv_file)}",
            title=self.title,
            institution=self.institution,
            source=self.source,
            references=self.references,
            comment=self.comment,
        )

        # Process all lines by chunck
        line_count = 0
        for lines in self.__open_csv(csv_file):
            # First, compute columns and rows index
            x, y = dtm_gridder.project_coords(
                lines[CSV.COL_LONGITUDE][:].to_numpy(), lines[CSV.COL_LATITUDE][:].to_numpy()
            )
            # Then, process elevations (mandatory)
            dtm_gridder.grid_elevations(x, y, lines[CSV.COL_ELEVATION][:].to_numpy())

            columns = np.floor(x).astype(int)
            rows = np.floor(y).astype(int)

            # Process other layers but CDI
            for column_name in lines:
                if column_name not in [CSV.COL_CDI, CSV.COL_CPRD]:
                    layer_name = CSV.COL_TO_LAYER[column_name] if column_name in CSV.COL_TO_LAYER else column_name
                    self.logger.info(f"Process column '{column_name}' to layer '{layer_name}'...")
                    if layer_name in dtm_gridder.layer_desc:
                        dtm_gridder.grid_keep_last(
                            layer_name=layer_name,
                            values=lines[column_name][:].to_numpy(),
                            columns=columns,
                            rows=rows,
                        )
            # Process CDI layer
            if CSV.COL_CDI in lines:
                self.logger.info("Process CDI column...")
                dtm_gridder.grid_cdi(
                    cdis=lines[CSV.COL_CDI][:].to_numpy(dtype=str, na_value=""),
                    columns=columns,
                    rows=rows,
                    cdi_or_cprd_prefix="SDN:CDI:LOCAL:",
                )

            # Process CPRD
            if CSV.COL_CPRD in lines:
                self.logger.info("Process CPRD column...")
                dtm_gridder.grid_cdi(
                    cdis=lines[CSV.COL_CPRD][:].to_numpy(dtype=str, na_value=""),
                    columns=columns,
                    rows=rows,
                    cdi_or_cprd_prefix="SDN:CPRD:LOCAL:",
                )
            # ensure value count layer will be added if related column is present
            if CSV.COL_VALUE_COUNT in lines:
                dtm_gridder.deal_with(DTM.VALUE_COUNT)

            line_count = line_count + lines.shape[0]
            self.logger.info(f"Number of lines processed : {line_count}")

        dtm_gridder.finalize_dtm()

        # Check presence of undefined CDI
        check_undefined_cdi(o_dtm_driver.dataset, self.allow_undefined_cdi)

    def __open_csv(self, csv_file: str):
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
                dtype[layer] = np.float32  # can't use np.int32 because of missing values
            else:
                dtype[layer] = np.dtype(str)

        usecols = list(range(nb_cols))

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

    def __estimate_extent_and_resolution(self, csv_file: str) -> None:
        """
        Read the CSV file and estimate the geobox and spatial resolution
        Initialize attributes self.spatial_resolution and self.geobox
        Raised exception : IOError when error occurs while parsing the file
        """
        evaluator = ExtentEvaluatorAuto(
            i_paths=[csv_file],
            indexes=self.indexes,
            spatial_resolution=self.spatial_resolution,
            auto_rounding=self.auto_rounding_arcmin,
            delimiter=self.delimiter,
            decimal_point=self.decimal_point,
            skip_rows=self.skip_rows,
            spatial_reference=self.spatial_reference.ExportToProj4(),
            pos_in_cell=self.pos_in_cell,
        )
        evaluator()
        self.geobox = evaluator.geobox
        self.spatial_resolution = evaluator.spatial_resolution
        self.logger.info(f"{csv_file} spatial_resolution:{self.spatial_resolution}, computed geobox {self.geobox}")

    def __call__(self) -> None:
        """Run method."""
        begin = datetime.datetime.now()
        self.monitor.set_work_remaining(len(self.i_paths))
        file_in_error = []
        for ind, csv_file in enumerate(self.i_paths):
            try:
                self.logger.info(f"Starting to convert {csv_file} to {self.o_paths[ind]}")
                if not self.overwrite and os.path.exists(self.o_paths[ind]):
                    self.logger.warning("File exists and overwrite is not allowed. Convertion aborted.")
                else:
                    now = datetime.datetime.now()
                    if self.recompute_geobox:
                        self.__estimate_extent_and_resolution(csv_file)
                    with dtm_driver.open_dtm(self.o_paths[ind], "w") as o_dtm_driver:
                        self.__export_data(csv_file, o_dtm_driver)

                    self.logger.info(
                        f"End of conversion for {csv_file} : {datetime.datetime.now() - now} time elapsed\n"
                    )

            except Exception as e:
                os.remove(self.o_paths[ind])
                file_in_error.append(csv_file)
                self.logger.error(f"An exception was thrown : {str(e)}", exc_info=True, stack_info=True)
        self.monitor.done()
        process_util.log_result(self.logger, begin, file_in_error)


if __name__ == "__main__":
    exporter = GriddedCsvToDtm(
        i_paths=["E:/temp/test_merc.csv"],
        overwrite=True,
        indexes={CSV.COL_LONGITUDE: 0, CSV.COL_LATITUDE: 1, CSV.COL_MIN_ELEVATION: 2, CSV.COL_ELEVATION: 4},
        headers_types={
            CSV.COL_LONGITUDE: "float",
            CSV.COL_LATITUDE: "float",
            CSV.COL_MIN_ELEVATION: "float",
            CSV.COL_ELEVATION: "float",
        },
        decimal_point=".",
        delimiter=",",
        skip_rows=1,
    )
    exporter()
