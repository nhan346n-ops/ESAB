import math

# number of decimal for resolution precision
DTM_PRECISION_DECIMAL_COUNT = 10
DTM_PRECISION = 10 ** (-DTM_PRECISION_DECIMAL_COUNT)


def estimate_row(upper_limit: float, lower_limit: float, spatial_resolution: float) -> int:
    """
    Estimate row number given two limits and a spatial_resolution
    :param upper_limit:
    :param lower_limit:
    :param spatial_resolution:
    :return:
    """
    # we estimated resolution, but to prevent adding a row or col due to precision issues, first compute size with a 8 decimal resolution
    count = abs(round((upper_limit - lower_limit) / spatial_resolution, DTM_PRECISION_DECIMAL_COUNT))
    # then use a ceil to include border cell if there is still any
    return int(math.ceil(count))


def estimate_col(right_or_east: float, left_or_west: float, spatial_resolution: float) -> int:
    """
    Estimate col number given two limits and a spatial_resolution
    :param right_or_east: upper is east or right corner of the bounding box
    :param left_or_west: lower is west or left corner of the bounding box
    :param spatial_resolution:
    :return:
    """
    if spatial_resolution <= 0:
        raise ValueError(f"Unsupported negative spatial resolution {spatial_resolution}")
    # Check if limits span the 180th meridian
    if right_or_east < 0 < left_or_west:
        right_or_east = right_or_east + 360.0
    # we estimated resolution, but to prevent adding a row or col due to precision issues, first compute size with a 8 decimal resolution
    count = abs(round((right_or_east - left_or_west) / spatial_resolution, DTM_PRECISION_DECIMAL_COUNT))
    # then use a ceil to include border cell if there is still any
    return int(math.ceil(count))
