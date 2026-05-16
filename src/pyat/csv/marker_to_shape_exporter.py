#! /usr/bin/env python3
# coding: utf-8

import datetime
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from osgeo import ogr, osr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log

TERRAIN_MARKER_COLUMNS = {
    "ID": "marker_id",
    "LATITUDE_DEG": "y",
    "LONGITUDE_DEG": "x",
    "LATITUDE_DMD": "lat_dmd",
    "LONGITUDE_DMD": "lon_dmd",
    "HEIGHT_ABOVE_SEA_SURFACE": "z",
    "SEA_FLOOR_LAYER": "sea_fl_la",
    "MARKER_COLOR": "color",
    "MARKER_SIZE": "size",
    "MARKER_SHAPE": "shape",
    "GROUP": "group",
    "CLASS": "class",
    "COMMENT": "comment",
}

WC_MARKER_COLUMNS = {
    "ID": "marker_id",
    "LAYER": "layer",
    "PING": "ping",
    "LATITUDE_DEG": "y",
    "LONGITUDE_DEG": "x",
    "LATITUDE_DMD": "lat_dmd",
    "LONGITUDE_DMD": "lon_dmd",
    "HEIGHT_ABOVE_SEA_SURFACE": "z",
    "HEIGHT_ABOVE_SEA_FLOOR": "ht_sea_fl",
    "SEA_FLOOR_ELEVATION": "sea_fl_el",
    "SEA_FLOOR_LAYER": "sea_fl_la",
    "DATE": "date",
    "TIME": "time",
    "MARKER_COLOR": "color",
    "MARKER_SIZE": "size",
    "MARKER_SHAPE": "shape",
    "GROUP": "group",
    "CLASS": "class",
    "COMMENT": "comment",
}


class MarkersToShapefileExporter:
    """
    Utility class to convert CSV marker files to shape files
    """

    def __init__(
        self,
        i_paths: List[str],
        o_paths: Optional[List[str]] = None,
        overwrite: bool = False,
        monitor: ProgressMonitor = DefaultMonitor,
        **metadata,
    ):
        """
        Constructor.

        Metadata is the dictionary of additional fields to be created on the shapefile's layer
        They are declared in the json of configuration, gathered in the "Shapefile options" page
        """
        self.logger = log.logging.getLogger(self.__class__.__name__)
        self.i_paths = arg_util.parse_list_of_files("i_paths", i_paths)
        if o_paths:
            self.o_paths = list(o_paths)
        else:
            # Create output name from the input with the nc extension.
            self.o_paths = [path[: path.rfind(".")] + ".shp" for path in i_paths]
        if len(self.o_paths) != len(self.i_paths):
            raise AttributeError("Number of Output/Input paths must be the same.")

        self.overwrite = overwrite
        self.metadata = metadata
        self.monitor = monitor

    def _read_markers(self, markers_file: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Parse the CSV and check the format
        """
        result = pd.read_csv(markers_file, delimiter=";")
        column_names = list(result)
        if "PING" in column_names:
            self.logger.info("File contains markers of Water Column.")
            marker_columns = WC_MARKER_COLUMNS
        else:
            self.logger.info("File contains markers of Terrain.")
            marker_columns = TERRAIN_MARKER_COLUMNS

        if len(column_names) != len(marker_columns) or any((column not in column_names for column in marker_columns)):
            raise IOError("Marker file has a wrong format")

        return result, marker_columns

    def _create_fields(self, marker_columns: Dict[str, str], markers: pd.DataFrame, layer: ogr.Layer) -> None:
        """
        Create a field in the shapefile for all column names
        marker_columns is TERRAIN_MARKER_COLUMNS or WC_MARKER_COLUMNS
        """
        for column_name, field_name in marker_columns.items():
            field_type = markers.dtypes[column_name]
            if np.issubdtype(field_type, float):
                layer.CreateField(ogr.FieldDefn(field_name, ogr.OFTReal))
            elif np.issubdtype(field_type, np.int64):
                layer.CreateField(ogr.FieldDefn(field_name, ogr.OFTInteger))
            else:
                layer.CreateField(ogr.FieldDefn(field_name, ogr.OFTString))

        # Add optional field
        for field, field_value in self.metadata.items():
            if field_value:
                layer.CreateField(ogr.FieldDefn(field, ogr.OFTString))

    def _set_fields(
        self, marker_columns: Dict[str, str], markers: pd.DataFrame, row: int, feature: ogr.Feature
    ) -> None:
        """
        Set the value of all fields in the shapefile with the ones of the CSV file
        marker_columns is TERRAIN_MARKER_COLUMNS or WC_MARKER_COLUMNS
        """
        for column_name, field_name in marker_columns.items():
            if not markers[column_name].isnull()[row]:
                field_value = markers[column_name][row]
                field_type = markers.dtypes[column_name]
                if np.issubdtype(field_type, float):
                    feature.SetField(field_name, float(field_value))
                elif np.issubdtype(field_type, np.int64):
                    feature.SetField(field_name, int(field_value))
                else:
                    feature.SetField(field_name, str(field_value))

        # Add optional field
        for field, field_value in self.metadata.items():
            if field_value:
                feature.SetField(field, str(field_value))

    def _convert_to_shapefile(self, i_file: str, data_source: ogr.DataSource) -> None:
        """
        Launch the conversion.
        """
        markers, marker_columns = self._read_markers(i_file)
        if len(markers) == 0:
            self.logger.warning("Marker file is empty. Convertion aborted.")
            return
        self.logger.info(f"Number of markers : {len(markers)}.")

        # creation layer
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = data_source.CreateLayer("Markers", srs=srs, geom_type=ogr.wkbPointZM)
        if layer is None:
            raise IOError("Could not create layer in the Shapefile")

        # Define fields
        self._create_fields(marker_columns, markers, layer)

        # Browse each line of the CSV file
        for i, (x, y, z) in enumerate(
            zip(markers["LONGITUDE_DEG"], markers["LATITUDE_DEG"], markers["HEIGHT_ABOVE_SEA_SURFACE"])
        ):
            # For each marker, create a feature with one point
            point = ogr.Geometry(ogr.wkbPointZM)
            point.AddPoint(x, y, z)
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetGeometry(point)
            # Transfert values in fields
            self._set_fields(marker_columns, markers, i, feature)
            layer.CreateFeature(feature)

    def __call__(self) -> None:
        """Run method."""
        shp_driver = ogr.GetDriverByName("ESRI Shapefile")
        if shp_driver is None:
            self.logger.error("Shapefile not supported. Conversion aborted")
            return

        begin = datetime.datetime.now()
        self.monitor.set_work_remaining(len(self.i_paths))
        file_in_error = []
        for i_file, o_file in zip(self.i_paths, self.o_paths):
            data_source = None
            try:
                self.logger.info(f"Starting to convert {i_file} to {o_file}")
                if os.path.exists(o_file):
                    if self.overwrite:
                        shp_driver.DeleteDataSource(o_file)
                    else:
                        self.logger.warning("File exists and overwrite is not allowed. Convertion aborted.")
                        file_in_error.append(i_file)
                        continue
                now = datetime.datetime.now()
                data_source = shp_driver.CreateDataSource(o_file)
                if data_source is None:
                    raise IOError("Unable to create the Shapefile. Marker file skipped")
                self._convert_to_shapefile(i_file, data_source)
                self.logger.info(f"End of conversion for {i_file} : {datetime.datetime.now() - now} time elapsed\n")
                self.monitor.worked(1)
            except IOError as ioerror:
                file_in_error.append(i_file)
                self.logger.error(str(ioerror))
            except Exception:
                file_in_error.append(i_file)
                self.logger.error("An exception was thrown!", exc_info=True, stack_info=True)
            finally:
                if data_source is None:
                    del data_source

        self.monitor.done()
        process_util.log_result(self.logger, begin, file_in_error)


if __name__ == "__main__":
    converter = MarkersToShapefileExporter(i_paths=["e:/temp/shape/markers1.csv"], overwrite=True)
    converter()
