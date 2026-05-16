import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import pygws.service.execution_context as exec_ctx
from osgeo import osr

from pyat.xsf.navshift.isobath_registration import (
    IsobathRegistrationProcess,
)
from pyat.navigation.navigation_data import NavigationData
from pyat.navigation.navigation_exporter import to_geodataframe
from pyat.utils.gdal_utils import GDALDataset


def compute_nav_shift(
    start_period: datetime,
    end_period: datetime,
    lat_file_path: str,
    lon_file_path: str,
    datetime_file_path: str,
    ref_contour_shp_path: List[str],
    profiles: str,
) -> None | List[dict]:
    """
    Computes navigation shift vectors that match source to target isobath
    from temporary data file produced by Navigation Shift Editor (GLOBE).

    @param start_period: datetime of the start of processed period.
    @param end_period: datetime of the end of processed period.
    @param lat_file_path: path of the temporary file containing navigation latitude data (double array).
    @param lon_file_path: path of the temporary file containing navigation longitude data (double array).
    @param datetime_file_path: path of the temporary file containing navigation datetime data (epoch time in ms, long array).
    @param ref_contour_shp_path: path of the contour file (.shp) used as reference.
    @param profiles: profiles to process (JSON with fields : name, contour_file_path, start_nav_index and end_nav_index).

    @return: produced shift vectors.
    """
    profiles_to_process = json.loads(profiles)

    # progress monitor init
    monitor = exec_ctx.get_root_progress_monitor()
    monitor.begin_task("Registration process", len(profiles_to_process))

    # Convert target isobath to GeoDataFrame (aka GDF)
    # TODO : manage several reference isobaths dataset (for now takes only first DTM into account)
    trgt_isobaths = isobaths_to_gdf(ref_contour_shp_path[0])

    # Convert navigation and time NavShift-Editor data file to GeoDataFrame
    srce_nav = nav_to_gdf(lat=lat_file_path, lon=lon_file_path, time=datetime_file_path, start=start_period)

    shift_vectors_list = []

    for profile in profiles_to_process:
        monitor.logger.info(f"Process profile {profile['name']}...")
        # Convert source isobaths to GDF
        srce_isobaths = isobaths_to_gdf(profile["contour_file_path"])

        # Start and end navigation indexes for the current profile
        start_nav_index = profile["start_nav_index"]
        end_nav_index = profile["end_nav_index"]

        # Init nav_shift process
        nav_shift_process = IsobathRegistrationProcess(
            trgt_isobaths=trgt_isobaths,
            srce_isobaths=srce_isobaths,
            srce_nav=srce_nav.iloc[start_nav_index:end_nav_index].loc[start_period:end_period],
        )
        # Call nav_shift process and populate shift vectors list
        shift_vectors = nav_shift_process()
        shift_vectors_list.extend(shift_vectors)

        monitor.worked(1)
    monitor.done()

    # Convert the list of ShiftVector objects to a list of dictionaries
    shift_vectors_dict = [shift_vector.to_dict() for shift_vector in shift_vectors_list]
    # Using rsocket (if present) to send the result
    rsocket_msg_emitter = exec_ctx.get_rsocket_msg_emitter()
    if rsocket_msg_emitter is not None:
        rsocket_msg_emitter.emit_dict(shift_vectors_dict)
        return None
    else:
        return shift_vectors_dict


def nav_to_gdf(lat: str, lon: str, time: str, start: datetime) -> gpd.GeoDataFrame:
    """
    Returns a GeoDataFrame from a set of temporary file produced by the NavShift-Editor.
    Files are big-endian sequence of navigation points coordinates and time (lat, lon, time).
    The resulting geodataframe is clipped between start and stop.
    """
    lat_arr = np.fromfile(lat, np.dtype(">d"))
    lon_arr = np.fromfile(lon, np.dtype(">d"))
    time_arr = np.fromfile(time, np.dtype(">i8"))

    # start and stop ar timezone-aware, convert gdf index to match timezone
    dt_series = pd.to_datetime(time_arr, unit="ms", origin="unix").tz_localize(start.tzname())
    nav = NavigationData(times=dt_series, latitudes=lat_arr, longitudes=lon_arr)
    nav_gdf = to_geodataframe(nav)

    return nav_gdf


def isobaths_to_gdf(filename: str) -> gpd.GeoDataFrame:
    """
    Returns a GeoDataFrame from a set of temporary shape file produced by the NavShift-Editor.
    Note : For some reason, files CSR is missing, hence the need for get_csr() method.
    """
    isobaths_gdf = gpd.read_file(filename)
    isobaths_gdf.crs = get_csr(filename)

    return isobaths_gdf


def get_csr(filename: str) -> str:
    """
    Workaround method that retrieves the missing isobath CSR from original DTM tiff file.
    We suppose that the DTM tiff file and isobath shp file were generated at the same time.
    """
    # get the corresponding tiff file name
    DTM_tif_file = sorted(
        list(Path(filename).parents[1].glob("*.tif")),
        key=lambda x: abs(os.path.getmtime(filename) - os.path.getmtime(x)),
    )[0]
    # extract CSR
    with GDALDataset(str(DTM_tif_file)) as rasterDs:
        input_projection = rasterDs.GetProjection()
        input_spatial_reference = osr.SpatialReference()
        input_spatial_reference.ImportFromWkt(input_projection)

        return input_spatial_reference.ExportToProj4()
