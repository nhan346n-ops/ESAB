#! /usr/bin/env python3
# coding: utf-8

import numpy as np


# class containing all naming, convention for 'new" dtm format, this one is inspired from Gebco dtm format

DIM_LAT: str = "lat"
DIM_LON: str = "lon"
DIM_CDI: str = "cdi_index_count"
DIM_ABSCISSA: str = "x"
DIM_ORDINATE: str = "y"

INTERPOLATED_CDI = "interpolated"


# known attribute
HISTORY_ATTRIB_NAME: str = "history"
VERSION_ATTRIB_NAME: str = "dtm_convention_version"


# Known layer
ELEVATION_NAME: str = "elevation"
ELEVATION_SMOOTHED_NAME: str = "elevation_smoothed"
ELEVATION_MIN: str = "elevation_min"
ELEVATION_MAX: str = "elevation_max"
STDEV: str = "stdev"
VALUE_COUNT: str = "value_count"
FILTERED_COUNT: str = "filtered_sounding"
INTERPOLATION_FLAG: str = "interpolation_flag"
CDI_INDEX: str = "cdi_index"
CDI: str = "cdi_reference"

BACKSCATTER: str = "backscatter"
MIN_ACROSS_DISTANCE: str = "min_across_distance"
MAX_ACROSS_DISTANCE: str = "max_across_distance"
MAX_ACCROSS_ANGLE: str = "max_across_angle"

LAT_NAME: str = "lat"
LON_NAME: str = "lon"
CRS_NAME: str = "crs"
ABSCISSA_NAME: str = "x"
ORDINATE_NAME: str = "y"

# Name of the processed layers
LAYERS = [
    ELEVATION_NAME,
    ELEVATION_SMOOTHED_NAME,
    ELEVATION_MIN,
    ELEVATION_MAX,
    STDEV,
    INTERPOLATION_FLAG,
    VALUE_COUNT,
    FILTERED_COUNT,
    CDI_INDEX,
    BACKSCATTER,
    MIN_ACROSS_DISTANCE,
    MAX_ACROSS_DISTANCE,
    MAX_ACCROSS_ANGLE,
]


# Type of layers
LAYERS_TYPE = {
    ELEVATION_NAME: np.float32,
    ELEVATION_MIN: np.float32,
    ELEVATION_MAX: np.float32,
    VALUE_COUNT: np.int32,
    FILTERED_COUNT: np.int32,
    STDEV: np.float32,
    CDI: str,
    CDI_INDEX: np.int32,
    ELEVATION_SMOOTHED_NAME: np.float32,
    INTERPOLATION_FLAG: np.int8,
    BACKSCATTER: np.float32,
    MIN_ACROSS_DISTANCE: np.float32,
    MAX_ACROSS_DISTANCE: np.float32,
    MAX_ACCROSS_ANGLE: np.float32,
}

FORMAT = "NETCDF4"

EXTENSION = ".dtm.nc"
EXTENSION_NC = ".nc"

# Dimensions mercator
MERCATOR_NAME = "mercator"
X_NAME = "x"
Y_NAME = "y"

# Mapping between mercator and latlon
MAPPING = {
    MERCATOR_NAME: CRS_NAME,
    X_NAME: LAT_NAME,
    Y_NAME: LON_NAME,
    CRS_NAME: MERCATOR_NAME,
    LAT_NAME: X_NAME,
    LON_NAME: Y_NAME,
}
