#! /usr/bin/env python3
# coding: utf-8

import datetime
import os
from typing import Any, Dict, List, Optional

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
from pyat.dtm.transform.interpolation import gap_filling as gap_filling_process
from pyat.dtm.cdi.cdi_layer_util import check_undefined_cdi
from pyat.dtm.mask import compute_geo_mask_from_dtm
from pyat.function.evaluate_csv_grid import GeoboxEvaluator


class CsvToDtm:
    """
    Utility class to convert CSV files as DTM (netcdf4 format)
    """

    def __init__(
        self,
        i_paths: List[str],
        indexes: Dict[str, str],
        target_resolution: float = 0.0,
        coord: Optional[Dict[str, float]] = None,
        headers_types: Optional[Dict[str, str]] = None,
        o_paths: Optional[List[str]] = None,
        overwrite: bool = False,
        delimiter: str = ";",
        decimal_point: str = ".",
        skip_rows: int = 0,
        depth_sign: float = 1.0,
        spatial_reference: str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        auto_layers: Optional[List[str]] = None,
        gap_filling: bool = False,
        mask_size: int = 3,
        mask: Optional[List[str]] = None,
        cdi: Optional[Dict[str, str]] = None,
        allow_undefined_cdi=False,
        spatial_antialiasing: bool = False,
        min_elevation: float = float("-inf"),
        max_elevation: float = float("inf"),
        min_sounds: int = 0,
        title: str = "",
        institution: str = "",
        source: str = "",
        references: str = "",
        comment: str = "",
        recompute_geobox=False,
        auto_rounding_arcmin=False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor.
        :param : i_paths : path of the input files to convert
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

        # Target spatial resolution to apply to all files.
        self.spatial_resolution = arg_util.parse_float("target_resolution", target_resolution, 0.0)

        if not coord is None:
            self.geobox = arg_util.parse_geobox("coord", coord)
        else:
            self.geobox = arg_util.Geobox(0, 0, 0, 0)
            self.recompute_geobox = True

        self.geobox.spatial_reference = osr.SpatialReference()
        self.geobox.spatial_reference.ImportFromProj4(spatial_reference)

        self.auto_layers = arg_util.parse_list_of_str(auto_layers)

        self.gap_filling = str.upper(str(gap_filling)) == "TRUE"
        self.mask_size = arg_util.parse_int("mask_size", mask_size, default=3, min_value=3, max_value=31)
        self.mask_files = arg_util.parse_list_of_files("mask", mask) if self.gap_filling else []

        self.cdi = cdi
        self.allow_undefined_cdi = allow_undefined_cdi

        self.min_elevation = arg_util.parse_float("min_elevation", min_elevation, float("-inf"))
        self.max_elevation = arg_util.parse_float("max_elevation", max_elevation, float("inf"))
        self.min_sounds = arg_util.parse_int("min_sounds", min_sounds, 0)

        self.title = title
        self.institution = institution
        self.source = source
        self.references = references
        self.comment = comment
        self.auto_rounding_arcmin = auto_rounding_arcmin
        self.recompute_geobox = recompute_geobox
        self.spatial_antialiasing = spatial_antialiasing
        self.monitor = monitor

    def __infer_cdi(self, sounder_file_path: str) -> Optional[str]:
        if self.cdi is None or len(self.cdi) == 0:
            return None
        sounder_file_name = os.path.basename(sounder_file_path)
        if sounder_file_name in self.cdi:
            self.logger.info(f"CDI of {sounder_file_name} is {self.cdi[sounder_file_name]}")
            return self.cdi[sounder_file_name]
        return None

    def __export_data(self, csv_files: List[str], o_dtm_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor) -> None:
        """
        Launch the export of the file.
        Raised exception : IOError when error occurs while parsing the file
        """
        monitor.set_work_remaining(60 * len(csv_files) + 40)

        target_resolution = self.spatial_resolution
        if self.recompute_geobox:
            evaluator = GeoboxEvaluator(
                i_paths=csv_files,
                indexes=self.indexes,
                spatial_resolution=self.spatial_resolution,
                evaluate_spatial_resolution=self.spatial_resolution == 0.0,
                auto_rounding=self.auto_rounding_arcmin,
                delimiter=self.delimiter,
                decimal_point=self.decimal_point,
                skip_rows=self.skip_rows,
                spatial_reference=self.geobox.spatial_reference.ExportToProj4(),
            )
            evaluator()
            self.geobox = evaluator.geobox
            if target_resolution == 0.0:
                # Apply default resolution (1/16 arc-minute for geographic data, 100 meters for projected data)
                target_resolution = 1 / 16 / 60 if self.geobox.spatial_reference.IsGeographic() else 100

        if len(csv_files) == 1:
            self.logger.info(f"{csv_files[0]} spatial_resolution:{target_resolution}, geobox {self.geobox}")
        else:
            self.logger.info(f"Spatial resolution:{target_resolution}, geobox {self.geobox}")

        # Create a DtmBuilder with a DtmDriver opened in write mode
        dtm_gridder = DtmGridder(
            o_dtm_driver,
            self.geobox,
            target_resolution,
            depth_factor=self.depth_sign,
            average_elevations=True,
            spatial_antialiasing=self.spatial_antialiasing,
        )

        # Add layer provided by CSV
        for column_name in self.indexes.keys():
            if column_name in CSV.COL_TO_LAYER:
                dtm_gridder.add_layer(CSV.COL_TO_LAYER[column_name])
            elif column_name not in [CSV.COL_LATITUDE, CSV.COL_LONGITUDE, CSV.COL_ELEVATION]:
                data_type = self.headers_types[column_name]
                dtm_gridder.add_layer(column_name, data_type, CSV.COL_DEFAULT_VALUES[data_type])

        # Ask gridder to compute some other layers if they are not provided by the CSV
        if DTM.ELEVATION_MIN in self.auto_layers:
            dtm_gridder.deal_with(DTM.ELEVATION_MIN)
        if DTM.ELEVATION_MAX in self.auto_layers:
            dtm_gridder.deal_with(DTM.ELEVATION_MAX)
        if DTM.STDEV in self.auto_layers:
            dtm_gridder.deal_with(DTM.STDEV)
        if DTM.VALUE_COUNT in self.auto_layers:
            dtm_gridder.deal_with(DTM.VALUE_COUNT)
        if DTM.FILTERED_COUNT in self.auto_layers:
            dtm_gridder.deal_with(DTM.FILTERED_COUNT)

        if self.min_elevation != float("-inf") or self.max_elevation != float("inf"):
            dtm_gridder.restrict_elevations(self.min_elevation, self.max_elevation)

        dtm_gridder.initialize_dtm_file(
            history=f"Created from {os.path.basename(csv_files[0])}",
            title=self.title,
            institution=self.institution,
            source=self.source,
            references=self.references,
            comment=self.comment,
        )
        monitor.worked(10)

        # Process all lines by chunck
        line_count = 0
        for csv_file in csv_files:
            cdi = self.__infer_cdi(csv_file)
            for lines in self.__open_csv(csv_file):
                # First, compute grid coords
                x, y = dtm_gridder.project_coords(
                    lines[CSV.COL_LONGITUDE][:].to_numpy(), lines[CSV.COL_LATITUDE][:].to_numpy()
                )
                # Then, process elevations (mandatory)
                dtm_gridder.grid_elevations(
                    x, y, lines[CSV.COL_ELEVATION][:].to_numpy(), cdi if CSV.COL_CDI not in lines else None
                )
                # Second transform coords to grid index
                columns = np.floor(x).astype(int)
                rows = np.floor(y).astype(int)

                # Process other layers but CDI
                for column_name in lines:
                    if column_name not in [CSV.COL_CDI, CSV.COL_CPRD]:
                        layer_name = CSV.COL_TO_LAYER[column_name] if column_name in CSV.COL_TO_LAYER else column_name
                        if layer_name in dtm_gridder.layer_desc:
                            dtm_gridder.grid_keep_last(
                                layer_name=layer_name,
                                values=lines[column_name][:].to_numpy(),
                                columns=columns,
                                rows=rows,
                            )
                # Process CDI
                if CSV.COL_CDI in lines:
                    dtm_gridder.grid_cdi(
                        cdis=lines[CSV.COL_CDI][:].to_numpy(dtype=str, na_value=""),
                        columns=columns,
                        rows=rows,
                        cdi_or_cprd_prefix="SDN:CDI:LOCAL:",
                    )

                # Process CPRD
                if CSV.COL_CPRD in lines:
                    dtm_gridder.grid_cdi(
                        cdis=lines[CSV.COL_CPRD][:].to_numpy(dtype=str, na_value=""),
                        columns=columns,
                        rows=rows,
                        cdi_or_cprd_prefix="SDN:CPRD:LOCAL:",
                    )

                # Standard deviation
                if DTM.STDEV in self.auto_layers:
                    dtm_gridder.grid_standard_deviation(columns, rows, lines[CSV.COL_ELEVATION][:].to_numpy())

                line_count = line_count + lines.shape[0]
                self.logger.info(f"Number of processed lines : {line_count}")
            monitor.worked(70)

        monitor.worked(10)

        if self.min_sounds > 0:
            dtm_gridder.reset_cell(self.min_sounds)
        monitor.worked(10)

        dtm_gridder.finalize_dtm()

        # Check presence of undefined CDI
        check_undefined_cdi(o_dtm_driver.dataset, self.allow_undefined_cdi)

    def __open_csv(self, csv_file: str):
        nb_cols = max(self.indexes.values()) + 1
        names = ["COL_" + str(index) for index in range(nb_cols)]
        dtype: Dict[str, Any] = {}
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

    def __export_csv_to_dtm(self) -> None:
        """
        Export each CSV file in one DTM file
        """
        begin = datetime.datetime.now()
        self.monitor.set_work_remaining(len(self.i_paths))
        file_in_error = []
        for csv_file, dtm_file in zip(self.i_paths, self.o_paths):
            sub_monitor = self.monitor.split(1)
            try:
                self.logger.info(f"Starting to convert {csv_file} to {dtm_file}")
                if not self.overwrite and os.path.exists(dtm_file):
                    self.logger.warning("File exists and overwrite is not allowed. Convertion aborted.")
                else:
                    now = datetime.datetime.now()
                    with dtm_driver.open_dtm(dtm_file, "w") as o_dtm_driver:
                        self.__export_data([csv_file], o_dtm_driver, sub_monitor)
                    if self.gap_filling:
                        self.logger.info("Starting interpolation process (Fill Gap)")
                        self.__fill_gap(dtm_file)

                    self.logger.info(
                        f"End of conversion for {csv_file} : {datetime.datetime.now() - now} time elapsed\n"
                    )

            except Exception as e:
                os.remove(dtm_file)
                file_in_error.append(csv_file)
                self.logger.error(f"An exception was thrown : {str(e)}", exc_info=True, stack_info=True)
            finally:
                sub_monitor.done()

        self.monitor.done()
        process_util.log_result(self.logger, begin, file_in_error)

    def __merge_csv_to_dtm(self) -> None:
        """
        Merge all CSV files in one DTM file
        """
        self.monitor.set_work_remaining(1)
        try:
            dtm_file = self.o_paths[0]
            self.logger.info(f"Merging all CSV files to {dtm_file}")
            if not self.overwrite and os.path.exists(dtm_file):
                self.logger.warning("File exists and overwrite is not allowed. Convertion aborted.")
            else:
                now = datetime.datetime.now()
                with dtm_driver.open_dtm(dtm_file, "w") as o_dtm_driver:
                    self.__export_data(self.i_paths, o_dtm_driver, self.monitor)
                if self.gap_filling:
                    self.logger.info("Starting interpolation process (Fill Gap)")
                    self.__fill_gap(dtm_file)

                self.logger.info(f"End of conversion : {datetime.datetime.now() - now} time elapsed\n")

        except Exception as e:
            os.remove(dtm_file)
            self.logger.error(f"An exception was thrown : {str(e)}", exc_info=True, stack_info=True)

        self.monitor.done()

    def __fill_gap(self, path: str) -> None:
        mask = compute_geo_mask_from_dtm(path, self.mask_files)
        with dtm_driver.open_dtm(path, "r+") as o_dtm_driver:
            gap_filling_process.process(o_dtm_driver, self.mask_size, mask, self.logger, 0, 2)

    def __call__(self) -> None:
        """Run method."""
        if len(self.i_paths) > 1 and len(self.o_paths) == 1:
            self.__merge_csv_to_dtm()
        else:
            self.__export_csv_to_dtm()


if __name__ == "__main__":
    exporter = CsvToDtm(
        i_paths=["e:/temp/test.emo"],
        o_paths=["e:/temp/test.dtm.nc"],
        coord={"north": 52.4920, "south": 52.4900, "west": -9.6840, "east": -9.6820},
        target_resolution=0.001,
        indexes={"Longitude/X": "0", "Latitude/Y": "1", "Elevation": "2"},
        headers_types={"Longitude/X": "float", "Latitude/Y": "float", "Elevation": "float"},
        delimiter=";",
        skip_rows=1,
        cdi="Test_CDI",
    )
    exporter()
