import logging as log
import os
from contextlib import ExitStack, contextmanager
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from pynvi import nvi_driver
from pytechsas.navigation.techsas_navigation import (
    TechsasFileNavigation,
    TechsasGpsFileNavigation,
    TechsasNavFileNavigation,
    TechsasSubnavFileNavigation,
)

from pyat.navigation.abstract_navigation import AbstractNavigation
from pyat.navigation.navigation_data import NavigationData
from pyat.navigation.navigation_utils import merge
from pyat.navigation.shapefile_navigation import ShapefileNavigation
from pyat.sounder.sounder_file_navigation import SounderFileNavigation
from pyat.utils.path_utils import ext_of_fname

logger = log.getLogger("navigation_factory")


def from_arrays(
    name: str,
    times: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    headings: Optional[np.ndarray] = None,
    altitudes: Optional[np.ndarray] = None,
    speeds: Optional[np.ndarray] = None,
    courses_over_ground: Optional[np.ndarray] = None,
) -> AbstractNavigation:
    """
    Builds navigation model from data arrays.
    """
    return NavigationData(
        name=name,
        times=times,
        latitudes=latitudes,
        longitudes=longitudes,
        headings=headings,
        altitudes=altitudes,
        speeds=speeds,
        courses_over_ground=courses_over_ground,
    )


@contextmanager
def from_file(file_path: str, filtered: bool = True):
    """
    Reads a file to get navigation data.

    Use this method in a "with...as...:" to properly release the resource after use.

    @param file_path: navigation data file path
    @param filtered: True by default, return navigation with only valid data.
    """
    file_name = os.path.basename(file_path)
    extension = ext_of_fname(file_path)
    navigation = None

    match extension:
        case "shp":
            logger.info(f"Get navigation from {file_name}, read as Shapefile file.")
            navigation = ShapefileNavigation(file_path)
        case "nvi" | "nvi.nc":
            logger.info(f"Get navigation from {file_name}, read as NVI file (filter on Quality Flag : {filtered}).")
            navigation = nvi_driver.get_nvi_driver(file_path, filtered)
            navigation.nc_open()
        case "mbg" | "xsf.nc":
            logger.info(f"Get navigation from {file_name}, read as Sounder file.")
            navigation = SounderFileNavigation(file_path)
        case "nav" | "nav.nc":
            logger.info(f"Get navigation from {file_name}, read as TECHSAS navigation file.")
            navigation = TechsasNavFileNavigation(file_path)
        case "subnav" | "subnav.nc":
            logger.info(f"Get navigation from {file_name}, read as TECHSAS sub navigation file.")
            navigation = TechsasSubnavFileNavigation(file_path)
        case "gps" | "gps.nc":
            logger.info(f"Get navigation from {file_name}, read as TECHSAS gps file.")
            navigation = TechsasGpsFileNavigation(file_path)
        case _:
            if extension.endswith("nc"):
                logger.info(f"Get navigation from {file_name}, read as TECHSAS file.")
                navigation = TechsasFileNavigation(file_path)
            else:
                logger.info(f"No navigation found for {file_name} (extension = {extension}).")

    try:
        yield navigation
    finally:
        if hasattr(navigation, "close") and callable(navigation.close):
            navigation.close()


def from_files(file_paths: List[str]) -> AbstractNavigation:
    """
    Builds navigation from several files.
    """
    with ExitStack() as stack:
        return merge([stack.enter_context(from_file(file_path)) for file_path in file_paths])


def from_geodataframe(gdf: gpd.GeoDataFrame, name: str = "") -> AbstractNavigation:
    """
    Builds navigation model from GeoDataFrame.
    """
    # check GeoDataframe data structure
    check_geodataframe(gdf)
    # Build navigation
    result = from_arrays(
        name=name, times=gdf.index.values, latitudes=gdf.geometry.y.values, longitudes=gdf.geometry.x.values
    )
    # List navigation arrays
    properties = list(vars(result).keys())
    # Find corresponding columns name in the GeoDataFrame
    attributes = gdf.columns.intersection(properties)
    # populate NavigationFileProxy properties with GeoDataFrame data
    if attributes is not None:
        for attr in attributes:
            setattr(result, attr, gdf[attr].values)
    return result


def check_geodataframe(gdf: gpd.GeoDataFrame) -> None:
    """
    Check if GeoDataFrame is conformed to init a NavigationFileProxy object.
    It should at least be time-indexed, and contains point geometries in EPSG:4326 CRS
    If OK, returns list of column names to use, else None
    """
    # Check it is datetime indexed -> times property
    if not isinstance(gdf.index, pd.DatetimeIndex):
        raise IndexError("The GeoDataFrame is not datetime indexed.")
    # check it contains geometry in EPSG:4326
    if gdf.crs.to_string() != "EPSG:4326":
        raise AttributeError("The GeoDataFrame is not in EPSG:4326 CRS.")
    # Check if all geometries are Points
    if not all(gdf.geom_type.values == "Point"):
        raise AttributeError("The GeoDataFrame active geometry is not Point.")
