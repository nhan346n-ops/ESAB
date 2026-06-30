#! /usr/bin/env python3
# coding: utf-8

import datetime as dt
import re
from logging import Logger
from os import PathLike
from typing import List

import geopandas as gpd
import pandas as pd
import pytz
from sonar_netcdf.utils import nc_merger as nc_m
from sonar_netcdf.utils.nc_merger import Timeline

from pyat.navigation.navigation_exporter import to_geodataframe
from pyat.navigation.navigation_factory import from_file
from pyat.utils.path_utils import basename_of_fname, ext_of_fname


def parse_cut_file(cut_file_path: str, logger: Logger) -> List[Timeline]:
    result: List[Timeline] = []
    cut_regex = re.compile(
        r".*(\d{2}/\d{2}/\d{4}\s*\d{2}:\d{2}:\d{2}\.\d{3})\s*(\d{2}/\d{2}/\d{4}\s*\d{2}:\d{2}:\d{2}\.\d{3})\s*(.*)"
    )
    try:
        with open(cut_file_path, "r", encoding="utf8") as cut_file:
            cut_lines = cut_file.read()
            cut_matches = cut_regex.findall(cut_lines)

        for cut_match in cut_matches:
            if len(cut_match) == 3:
                date_start = pytz.utc.localize(dt.datetime.strptime(cut_match[0], r"%d/%m/%Y %H:%M:%S.%f"))
                date_stop = pytz.utc.localize(dt.datetime.strptime(cut_match[1], r"%d/%m/%Y %H:%M:%S.%f"))
                result.append(Timeline(cut_match[2], date_start, date_stop))
    except OSError as e:
        logger.error(f"Unparsable cut file : {str(e)}")
    logger.info(f"Number of cut lines found : {len(result)}")
    return result


def create_cut_file_from_ncfile_set(file_list: List[str], cut_file_path: str) -> None:
    """
    Generate Cut file from Sonar_netcdf file list

    """
    # Get start and stop time and time sort file list
    sorted_file_list = nc_m.time_sort_files(file_list)

    with open(cut_file_path, "w", encoding="utf8") as cut_file:
        digit_count = len(str(len(sorted_file_list)))
        for k, file in enumerate(sorted_file_list):
            start_datetime = file[1].strftime("%d/%m/%Y  %H:%M:%S.%f")[:-3]
            stop_datetime = file[2].strftime("%d/%m/%Y  %H:%M:%S.%f")[:-3]
            cut_file.write(f"> {start_datetime}  {stop_datetime}  line_{k + 1:0{digit_count}}\n")


def create_cut_lines_from_files(
    i_paths: List[PathLike],
    o_paths: List[PathLike],
    i_geo_mask_path: PathLike,
    reverse_geo_mask: bool = False,
) -> List[Timeline]:
    """
    Creates timelines (start/end dates) by clipping navigation files with geographic mask.
    """
    # Read input mask file (.shp...).
    mask_gdf = gpd.read_file(i_geo_mask_path)

    result: List[Timeline] = []
    for i_path, o_path in zip(i_paths, o_paths):
        # Read navigation from input file to cut.
        with from_file(i_path) as nav:
            nav_gdf = to_geodataframe(nav, index_on_time=False).set_crs(mask_gdf.crs)  # navigation CRS must be the same
            # Computes cut lines.
            result += cut_with_geo_mask(
                nav_gdf=nav_gdf,
                geo_mask_gdf=mask_gdf,
                reverse_geo_mask=reverse_geo_mask,
                line_prefix=f"{basename_of_fname(o_path)}",
                line_suffix=f".{ext_of_fname(i_path)}",
            )

    return result


def cut_with_geo_mask(
    nav_gdf: gpd.GeoDataFrame,
    geo_mask_gdf: gpd.GeoDataFrame,
    reverse_geo_mask: bool = False,
    line_prefix: str = "Line",
    line_suffix: str = "",
) -> List[Timeline]:
    """
    Creates timelines (start/end dates) by clipping navigation data with geographic mask.

    Arguments:
        reverse_geo_mask -- if true, keep data outside geographic mask.
    """
    # Clip nav with geographic mask.
    clipped_nav_gdf = nav_gdf.clip(geo_mask_gdf).sort_index()
    if reverse_geo_mask:
        clipped_nav_gdf = nav_gdf.drop(clipped_nav_gdf.index)

    # Get cut lines starts and ends.
    timelines_df = pd.DataFrame(
        {
            "start": clipped_nav_gdf[clipped_nav_gdf.index.diff() != 1].times.values,
            "stop": clipped_nav_gdf[clipped_nav_gdf.index.diff(-1) != -1].times.values,
        }
    )

    # Remove line where start == end.
    timelines_df = timelines_df[timelines_df["start"] != timelines_df["stop"]]

    # Localize to UTC if naive.
    def localize_if_naive(series):
        return series.apply(lambda x: pytz.utc.localize(x) if x.tzinfo is None else x)

    timelines_df["start"] = localize_if_naive(timelines_df["start"])
    timelines_df["stop"] = localize_if_naive(timelines_df["stop"])

    # Build result timelines.
    return [
        Timeline(
            name=f"{line_prefix}_{(idx + 1)}{line_suffix}" if len(timelines_df) > 1 else f"{line_prefix}{line_suffix}",
            start=line.start,
            stop=line.stop,
        )
        for idx, line in timelines_df.iterrows()
    ]
