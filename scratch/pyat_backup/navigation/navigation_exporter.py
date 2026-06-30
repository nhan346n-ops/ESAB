from typing import List

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from pynvi.version_2 import export_nvi

from pyat.navigation.abstract_navigation import AbstractNavigation
from pyat.navigation.navigation_data import NavigationData


def to_nvi(
    nav: AbstractNavigation, o_path: str, source_filenames: List[str] | None = None, overwrite: bool = False
) -> None:
    """
    Export a AbstractNavigation to a NVI file (.nvi.nc)
    """
    args = export_nvi.ExportNviArg(
        o_path=o_path,
        time=nav.get_times().astype(np.uint64),
        latitude=nav.get_latitudes(),
        longitude=nav.get_longitudes(),
        heading=nav.get_headings(),
        height_above_reference_ellipsoid=nav.get_altitudes(),
        vertical_offset=nav.get_vertical_offsets(),
        source_filenames=source_filenames,
        quality_flag=np.zeros_like(nav.get_latitudes()),  # corresponding to no_quality_control flag in SeaDataNet vocab
        overwrite=overwrite,
    )

    export_nvi.exports_with_ExportNviArg(args)


def to_geodataframe(nav: AbstractNavigation, index_on_time: bool = True) -> gpd.GeoDataFrame:
    """
    Convert NavigationFileProxy to a time-indexed GeoDataFrame with point geometries in EPSG:4326 CRS
    Filename attribute is lost
    By default, set 'times' as index.
    """
    # Discard longitudes and latitudes values as they will be stored within geometry
    filtered_out = ["longitudes", "latitudes"]
    # Filter out empty NavigationFileProxy attributes
    min_size = len(nav.get_longitudes())
    # Copy navigation data to local arrays.
    nav = NavigationData.copy_from(nav)
    # Build data from navigation arrays (always keep "times" even if not ndarray).
    data = {
        k: v
        for k, v in vars(nav).items()
        if k == "times" or (isinstance(v, np.ndarray) and len(v) == min_size and k not in filtered_out)
    }
    # Populate geometry with points computed with lat/lon
    data["points"] = [Point(xy) for xy in zip(nav.get_longitudes(), nav.get_latitudes())]
    # define a GeoDataFrame that represents the NavigationFileProxy as points
    gdf = gpd.GeoDataFrame(
        data=data,
        geometry="points",
        crs="EPSG:4326",
    )
    # Set 'times' as index
    if index_on_time:
        gdf = gdf.set_index("times")
    return gdf
