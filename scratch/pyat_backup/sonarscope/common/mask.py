#! /usr/bin/env python3
# coding: utf-8

import fiona
import geopandas as gpd
import numpy as np
import shapely

fiona.supported_drivers["LIBKML"] = "rw"
fiona.supported_drivers["KML"] = "rw"


def compute_geo_mask_from_lon_lat(longitudes: np.ndarray, latitudes: np.ndarray, mask_files: list) -> np.ndarray:
    """Compute a mask of the area which must be processed.
    The area is set to 1 if data shall be processed, to 0 if it shall be ignored

    Arguments:
        longitudes {ndarray} -- longitudes of data to be processed.
        latitudes {ndarray} -- longitudes of data to be processed.
        mask_files {list} -- list of files (**kml, *.shp)

    Raises:
        ProcessingError: failed to rasterize vector file

    Returns:
        np.ndarray -- mask array
    """
    # Get size of the mask
    geo_mask = np.full_like(longitudes, fill_value=True, dtype=bool)
    geometric_points = gpd.points_from_xy(longitudes.ravel(), latitudes.ravel(), crs="EPSG:4326")

    # Read the KML file using geopandas
    for mask_file in mask_files:
        # read geometry file (kml,shp)
        kml_map = gpd.read_file(mask_file)
        kml_map = kml_map.to_crs(crs="EPSG:4326")
        # retrieve a unique geometry as union of included geometry
        poly = kml_map.geometry.unary_union
        # Call prepare for performance
        shapely.prepare(poly)
        # Flag points contained in mask geometry
        new_mask = poly.contains(geometric_points).reshape(geo_mask.shape)
        geo_mask = geo_mask & new_mask

    return geo_mask
