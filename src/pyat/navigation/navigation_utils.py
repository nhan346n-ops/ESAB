import warnings
from typing import List

import numpy as np

from pyat.navigation.abstract_navigation import AbstractNavigation
from pyat.navigation.navigation_data import NavigationData


def interpolate(navigation_data: NavigationData, interpolation_time: np.ndarray) -> NavigationData:
    """Interpolate navigation data based on the given time sensor
    :param navigation_data: input navigation data
    :param interpolation_time: time values used to get interpolated data
    :return: interpolated navigation data
    """
    # interpolate data
    time_navigation = navigation_data.times
    if time_navigation[0] > interpolation_time[0]:
        raise Exception(
            f"Not supported case : Starting date ({time_navigation[0]}) from navigation file is higher than starting date of time sensor ({interpolation_time[0]})"
        )

    if time_navigation[-1] < interpolation_time[-1]:
        raise Exception(
            f"Not supported case : Last date  ({time_navigation[-1]}) from navigation file is less than last date of time sensor ({interpolation_time[-1]})"
        )

    # We need to change time to float in order to be able to interpolate data
    # We substract the lowest date as a reference date
    reference_date = time_navigation[0]
    time_navigation_float = (time_navigation - reference_date) / np.timedelta64(1, "s")
    time_sensor_float = (interpolation_time - reference_date) / np.timedelta64(1, "s")

    # Need to check for interpolation around 180/-180 for longitudes
    longitudes = np.interp(time_sensor_float, time_navigation_float, navigation_data.longitudes)
    latitudes = np.interp(time_sensor_float, time_navigation_float, navigation_data.latitudes)

    altitudes = (
        np.interp(time_sensor_float, time_navigation_float, navigation_data.altitudes)
        if navigation_data.altitudes is not None
        else None
    )

    heading = None
    if navigation_data.headings is not None:
        # use 'unwrap' method to get a correct interpolation of heading angles
        heading_unwrapped = navigation_data.headings
        heading_unwrapped[~np.isnan(heading_unwrapped)] = np.unwrap(
            heading_unwrapped[~np.isnan(heading_unwrapped)], period=360
        )
        heading = np.interp(time_sensor_float, time_navigation_float, heading_unwrapped) % 360

    speed = (
        np.interp(time_sensor_float, time_navigation_float, navigation_data.speeds)
        if navigation_data.get_speeds() is not None
        else None
    )

    result = NavigationData(
        name=navigation_data.name,
        times=interpolation_time,
        latitudes=latitudes,
        longitudes=longitudes,
        headings=heading,
        altitudes=altitudes,
        speeds=speed,
    )
    return result


def merge(navigation_list: List[AbstractNavigation]) -> AbstractNavigation:
    """Merges several navigation."""
    aggr_names = []
    aggr_times_nav = []
    aggr_longitudes_nav = []
    aggr_latitudes_nav = []
    aggr_altitudes_nav = []
    aggr_vertical_offsets_nav = []
    aggr_speeds = []
    aggr_headings = []

    # read navigation from files
    for nav_data in navigation_list:
        aggr_names.append(nav_data.get_name())

        times = nav_data.get_times()
        aggr_times_nav.extend(times)

        longitudes = nav_data.get_longitudes()
        aggr_longitudes_nav.extend(longitudes)

        latitudes = nav_data.get_latitudes()
        aggr_latitudes_nav.extend(latitudes)

        headings = nav_data.get_headings()
        if headings is not None:
            aggr_headings.extend(headings)

        altitudes = nav_data.get_altitudes()
        if altitudes is not None:
            aggr_altitudes_nav.extend(altitudes)

        vertical_offsets = nav_data.get_vertical_offsets()
        if vertical_offsets is not None:
            aggr_vertical_offsets_nav.extend(vertical_offsets)

        speeds = nav_data.get_speeds()
        if speeds is not None:
            aggr_speeds.extend(speeds)

    # sort nav points by time
    indexer = np.asarray(aggr_times_nav).argsort()

    warnings.filterwarnings("ignore", message="Warning: converting a masked element to nan")
    merged_nav = NavigationData(
        name="; ".join(aggr_names),
        times=np.asarray(aggr_times_nav)[indexer],
        latitudes=np.asarray(aggr_latitudes_nav)[indexer],
        longitudes=np.asarray(aggr_longitudes_nav)[indexer],
        headings=np.asarray(aggr_headings)[indexer] if aggr_headings else None,
        altitudes=np.asarray(aggr_altitudes_nav)[indexer] if aggr_altitudes_nav else None,
        vertical_offsets=np.asarray(aggr_vertical_offsets_nav)[indexer] if aggr_vertical_offsets_nav else None,
        speeds=np.asarray(aggr_speeds)[indexer] if aggr_speeds else None,
    )
    return merged_nav
