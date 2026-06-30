#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List, Tuple

from osgeo import ogr

import pyat.utils.pyat_logger as log

logger = log.logging.getLogger("kml_splitter")


def extract_polygons_from_kml(input_kml, output_dir) -> List[str]:
    """
    Reads a KML file and creates a separate KML file for each polygon.

    Args:
        input_kml: Path to the source KML file
        output_dir: Output directory for KML files (default: "output_polygons")
    """

    result: List[str] = []

    ogr.DontUseExceptions()

    # Open the source KML file
    ds_in = ogr.Open(input_kml, 0)  # 0 = read-only
    driver = ogr.GetDriverByName("KML")

    if ds_in is None:
        logger.warning(f"Unable to open file {input_kml}")
        return [input_kml]

    # Iterate through all layers in the KML
    polygon_count = 0

    for layer_idx in range(ds_in.GetLayerCount()):
        layer = ds_in.GetLayerByIndex(layer_idx)
        layer_name = layer.GetName()

        logger.debug(f"Processing layer: {layer_name}")

        # Iterate through all features in the layer
        for feature in layer:
            geom = feature.GetGeometryRef()

            if geom is None:
                continue

            # Check if it's a polygon or contains polygons
            geom_type = geom.GetGeometryName()

            # Process different geometry types
            polygons = []

            if geom_type == "POLYGON":
                poly = geom.Clone()
                poly.CloseRings()
                polygons.append(poly)
            elif geom_type == "MULTIPOLYGON":
                for i in range(geom.GetGeometryCount()):
                    poly = geom.GetGeometryRef(i).Clone()
                    poly.CloseRings()
                    polygons.append(poly)

            # Create a KML file for each polygon
            for polygon in polygons:
                polygon_count += 1

                # Output filename
                output_filename = os.path.join(output_dir, f"polygon_{polygon_count:04d}.kml")

                # Create the new KML file
                ds_out = driver.CreateDataSource(output_filename)

                # Create a layer with the same projection
                srs = layer.GetSpatialRef()
                layer_out = ds_out.CreateLayer("polygon", srs, ogr.wkbPolygon)

                # Copy fields from the original feature
                layer_defn = layer.GetLayerDefn()
                for i in range(layer_defn.GetFieldCount()):
                    field_defn = layer_defn.GetFieldDefn(i)
                    layer_out.CreateField(field_defn)

                # Create the new feature
                feature_out = ogr.Feature(layer_out.GetLayerDefn())
                feature_out.SetGeometry(polygon)

                # Copy attributes
                for i in range(layer_defn.GetFieldCount()):
                    field_name = layer_defn.GetFieldDefn(i).GetName()
                    feature_out.SetField(field_name, feature.GetField(i))

                # Add the feature to the layer
                layer_out.CreateFeature(feature_out)

                # Release resources
                feature_out = None
                ds_out = None

                result.append(output_filename)
                logger.debug(f"  Created: {output_filename}")

    # Close the source file
    ds_in = None

    logger.debug(f"{polygon_count} polygon(s) extracted to '{output_dir}' directory")

    return result


def extract_polylines_from_kml(file_path: str) -> List[List[Tuple[float, float]]]:
    """
    Extract the polylines from a KML or SHP file using GDAL.

    Args:
        file_path: Path to the KML or SHP file

    Returns:
        List of (longitude, latitude) tuples representing the polyline coordinates,
    """

    result: List[List[Tuple[float, float]]] = []
    dataset = None
    try:
        # Open the vector file with GDAL/OGR
        dataset = ogr.Open(file_path)

        if dataset is not None:
            # Get the first layer
            layer = dataset.GetLayer(0)
            if layer is not None:
                # Iterate through features to find the first polyline
                for feature in layer:
                    geometry = feature.GetGeometryRef()
                    if geometry is not None:
                        result.extend(_extract_linestrings(geometry))
            else:
                logger.warning("No layer found in file")
        else:
            logger.warning(f"Failed to open file: {file_path}")

        return result

    except Exception as e:
        logger.error(f"Error reading file with GDAL: {e}")
        return result

    finally:
        # Close the dataset
        if dataset is not None:
            dataset = None


def _extract_linestrings(geometry) -> List[List[Tuple[float, float]]]:
    """
    Extract the polylines from the geometry.
    """
    result: List[List[Tuple[float, float]]] = []

    # Check for LineString geometry (type code: 2)
    geom_type = geometry.GetGeometryType()
    if geom_type in (ogr.wkbLineString, ogr.wkbLineString25D):
        coordinates = _extract_linestring_coords(geometry)
        if coordinates:
            result.append(coordinates)

    # Check for MultiLineString geometry (type code: 5)
    elif geom_type in (ogr.wkbMultiLineString, ogr.wkbMultiLineString25D):
        # Extract all LineStrings from the MultiLineString
        num_geometries = geometry.GetGeometryCount()
        logger.debug(f"Found MultiLineString with {num_geometries} LineStrings")

        for i in range(num_geometries):
            line = geometry.GetGeometryRef(i)
            coordinates = _extract_linestring_coords(line)
            if coordinates:
                result.append(coordinates)

    return result


def _extract_linestring_coords(geometry) -> List[Tuple[float, float]]:
    """
    Extract coordinates from a LineString geometry.

    Args:
        geometry: OGR LineString geometry object

    Returns:
        List of (longitude, latitude) tuples
    """
    coordinates = []

    # Get the number of points in the linestring
    point_count = geometry.GetPointCount()

    # Extract each point (ignoring Z coordinate if present)
    for i in range(point_count):
        lon = geometry.GetX(i)
        lat = geometry.GetY(i)
        coordinates.append((lon, lat))

    return coordinates
