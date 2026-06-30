from typing import List

import numpy as np
import sonarnative

from pyat.xsf.xsf_driver import XsfDriver, WATERLINE_TO_CHART_DATUM


def get_biggest_echoes_count(in_xsfs: List[str], swaths_wanted):
    """
    return the biggest echo count per swath in the input files
    Args:
        in_xsfs: input files
        swaths_wanted: number of swath you want iterate to

    Returns:
        the biggest echo count found in the input files

    """
    echoes_count = 0
    for input_file in in_xsfs:
        try:
            spatializer = sonarnative.open_spatializer(input_file, -1, True)
            swath_count = spatializer.get_swath_count()
            swath_list = list(range(swath_count))
            for i in swath_list[::swaths_wanted]:
                # arg: file / swath index / number of swath
                echoes_count = max(echoes_count, sonarnative.estimate_beam_echo_count(spatializer, i, swaths_wanted))
        finally:
            sonarnative.close_spatializer(spatializer)
    return echoes_count


def get_xsf_statistics(xsf_driver: XsfDriver):
    """
    read the xsf files and get the variables to compute statistics
    Returns: min_across, max_across, min_elevation, max_elevation
    """
    valid = xsf_driver.read_validity_flags(0, xsf_driver.sounder_file.swath_count)
    # ACROSS
    across = xsf_driver.read_across_distances(0, xsf_driver.sounder_file.swath_count)
    across[~valid] = np.nan
    min_across = np.nanmin(across)
    max_across = np.nanmax(across)
    # DEPTH
    waterline_to_chart_datum = xsf_driver[WATERLINE_TO_CHART_DATUM][:]
    transducer_depth = xsf_driver.read_transducer_depth(0, xsf_driver.sounder_file.swath_count)
    detection_depth = xsf_driver.read_fcs_depths(0, xsf_driver.sounder_file.swath_count)
    detection_depth[~valid] = np.nan
    detection_z = xsf_driver.read_vertical_distances(0, xsf_driver.sounder_file.swath_count)
    detection_z[~valid] = np.nan
    max_vertical_distance = np.nanmax(detection_z)
    max_elevation = np.nanmax(waterline_to_chart_datum - transducer_depth)
    min_elevation = -np.nanmax(detection_depth)

    return min_across, max_across, min_elevation, max_elevation, max_vertical_distance
