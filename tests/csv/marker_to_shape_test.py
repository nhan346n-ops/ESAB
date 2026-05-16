#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile as tmp

import fiona
from osgeo import ogr

from pyat.csv.marker_to_shape_exporter import MarkersToShapefileExporter, WC_MARKER_COLUMNS, TERRAIN_MARKER_COLUMNS


def test_terrain_markers():
    """
    Convert a CSV (Terrain Marker format) to Shapefile
    """
    # CSV file with 2 terrain markers
    csv_content = """ "ID";"LATITUDE_DEG";"LONGITUDE_DEG";"LATITUDE_DMD";"LONGITUDE_DMD";"HEIGHT_ABOVE_SEA_SURFACE";"SEA_FLOOR_LAYER";"MARKER_COLOR";"MARKER_SIZE";"MARKER_SHAPE";"GROUP";"CLASS";"COMMENT"
                     "001";"41.761976"   ;"4.209475"     ;"41__45,717"  ;"4__12,567"    ;"-2540.668"               ;"CALMAR97"       ;"#ffff00ff"   ;"50"         ;"Cylinder"    ;"BIO"  ;"Cetac";"Com"
                     "002";"41.761976"   ;"4.209475"     ;"41__45,717"  ;"4__12,567"    ;"-2540.668"               ;"CALMAR97"       ;"#ffff00ff"   ;"50"         ;"Cylinder"    ;"BIO"  ;"Cetac";"Com"
        """
    try:
        path_markers = tmp.mktemp(suffix=".csv")
        path_shapefile = tmp.mktemp(suffix=".shp")
        with open(path_markers, "w") as markers_file:
            markers_file.write(csv_content.replace(" ", ""))

        # Export CSV -> DTM
        exporter = MarkersToShapefileExporter(i_paths=[path_markers], o_paths=[path_shapefile])
        exporter()

        with fiona.open(path_shapefile) as shape:
            # Check CRS
            assert 'AUTHORITY["EPSG","6326"]' in shape.crs_wkt
            for i, point in enumerate(shape.values()):
                assert point["geometry"]["type"] == "Point"
                assert point["geometry"]["coordinates"] == (4.209475, 41.761976, -2540.668)
                assert point["properties"][TERRAIN_MARKER_COLUMNS["ID"]] == i + 1
                assert point["properties"][TERRAIN_MARKER_COLUMNS["LONGITUDE_DEG"]] == 4.209475
                assert point["properties"][TERRAIN_MARKER_COLUMNS["LATITUDE_DEG"]] == 41.761976
                assert point["properties"][TERRAIN_MARKER_COLUMNS["HEIGHT_ABOVE_SEA_SURFACE"]] == -2540.668
                assert point["properties"][TERRAIN_MARKER_COLUMNS["LATITUDE_DMD"]] == "41__45,717"
                assert point["properties"][TERRAIN_MARKER_COLUMNS["LONGITUDE_DMD"]] == "4__12,567"
                assert point["properties"][TERRAIN_MARKER_COLUMNS["SEA_FLOOR_LAYER"]] == "CALMAR97"
                assert point["properties"][TERRAIN_MARKER_COLUMNS["MARKER_COLOR"]] == "#ffff00ff"
                assert point["properties"][TERRAIN_MARKER_COLUMNS["MARKER_SIZE"]] == 50
                assert point["properties"][TERRAIN_MARKER_COLUMNS["MARKER_SHAPE"]] == "Cylinder"
                assert point["properties"][TERRAIN_MARKER_COLUMNS["GROUP"]] == "BIO"
                assert point["properties"][TERRAIN_MARKER_COLUMNS["CLASS"]] == "Cetac"
                assert point["properties"][TERRAIN_MARKER_COLUMNS["COMMENT"]] == "Com"

    finally:
        os.remove(path_markers)
        ogr.GetDriverByName("ESRI Shapefile").DeleteDataSource(path_shapefile)


def test_wc_markers():
    """
    Convert a CSV (Water columr Marker format) to Shapefile
    """
    # CSV file with 2 terrain markers
    csv_content = """   "ID";"LAYER";"PING";"LATITUDE_DEG";"LONGITUDE_DEG";"LATITUDE_DMD";"LONGITUDE_DMD";"HEIGHT_ABOVE_SEA_SURFACE";"HEIGHT_ABOVE_SEA_FLOOR";"SEA_FLOOR_ELEVATION";"SEA_FLOOR_LAYER";"DATE"      ;"TIME"        ;"MARKER_COLOR";"MARKER_SIZE";"MARKER_SHAPE";"GROUP";"CLASS";"COMMENT"
                       "001";"wc_01";"4"   ;"47.143708"   ;"-9.184005"    ;"47__8,617"   ;"-9__11,033"   ;"-2801.757"               ;"1754.243"              ;"-4556"              ;"Globe_layer"    ;"2018-09-07";"10:49:29.756";"#004000ff"   ;"50"         ;"Sphere"      ;"FLUID";"IC"   ;""
        """
    try:
        path_markers = tmp.mktemp(suffix=".csv")
        path_shapefile = tmp.mktemp(suffix=".shp")
        with open(path_markers, "w") as markers_file:
            markers_file.write(csv_content.replace(" ", ""))

        # Export CSV -> DTM
        exporter = MarkersToShapefileExporter(i_paths=[path_markers], o_paths=[path_shapefile])
        exporter()

        with fiona.open(path_shapefile) as shape:
            # Check CRS
            assert 'AUTHORITY["EPSG","6326"]' in shape.crs_wkt
            for i, point in enumerate(shape.values()):
                assert point["geometry"]["type"] == "Point"
                assert point["geometry"]["coordinates"] == (-9.184005, 47.143708, -2801.757)
                assert point["properties"][WC_MARKER_COLUMNS["ID"]] == i + 1
                assert point["properties"][WC_MARKER_COLUMNS["LAYER"]] == "wc_01"
                assert point["properties"][WC_MARKER_COLUMNS["PING"]] == 4
                assert point["properties"][WC_MARKER_COLUMNS["LONGITUDE_DEG"]] == -9.184005
                assert point["properties"][WC_MARKER_COLUMNS["LATITUDE_DEG"]] == 47.143708
                assert point["properties"][WC_MARKER_COLUMNS["HEIGHT_ABOVE_SEA_SURFACE"]] == -2801.757
                assert point["properties"][WC_MARKER_COLUMNS["LATITUDE_DMD"]] == "47__8,617"
                assert point["properties"][WC_MARKER_COLUMNS["LONGITUDE_DMD"]] == "-9__11,033"
                assert point["properties"][WC_MARKER_COLUMNS["HEIGHT_ABOVE_SEA_FLOOR"]] == 1754.243
                assert point["properties"][WC_MARKER_COLUMNS["SEA_FLOOR_ELEVATION"]] == -4556
                assert point["properties"][WC_MARKER_COLUMNS["SEA_FLOOR_LAYER"]] == "Globe_layer"
                assert point["properties"][WC_MARKER_COLUMNS["DATE"]] == "2018-09-07"
                assert point["properties"][WC_MARKER_COLUMNS["TIME"]] == "10:49:29.756"
                assert point["properties"][WC_MARKER_COLUMNS["MARKER_COLOR"]] == "#004000ff"
                assert point["properties"][WC_MARKER_COLUMNS["MARKER_SIZE"]] == 50
                assert point["properties"][WC_MARKER_COLUMNS["MARKER_SHAPE"]] == "Sphere"
                assert point["properties"][WC_MARKER_COLUMNS["GROUP"]] == "FLUID"
                assert point["properties"][WC_MARKER_COLUMNS["CLASS"]] == "IC"
                assert point["properties"][WC_MARKER_COLUMNS["COMMENT"]] is None
    finally:
        os.remove(path_markers)
        ogr.GetDriverByName("ESRI Shapefile").DeleteDataSource(path_shapefile)
