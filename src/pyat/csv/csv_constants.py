#! /usr/bin/env python3
# coding: utf-8

from typing import Any, Dict

import numpy as np
import pyat.dtm.dtm_standard_constants as DTM

# COLUMNS OF CSV DTM (.emo for example)
COL_LONGITUDE: str = "Longitude/X"
COL_LATITUDE: str = "Latitude/Y"
COL_MIN_ELEVATION: str = "Min elevation"
COL_MAX_ELEVATION: str = "Max elevation"
COL_ELEVATION: str = "Elevation"
COL_STDDEV: str = "Std dev"
COL_VALUE_COUNT: str = "Value count"
COL_INTER_FLAG: str = "Interpolation flag"
COL_ELEVATION_SMOOTHED: str = "Elevation smoothed"
COL_CDI: str = "CDI"
COL_CPRD: str = "CPRD"
COL_BACKSCATTER: str = "backscatter"

# Link between columns and layers
COL_TO_LAYER: Dict[str, str] = {
    COL_MIN_ELEVATION: DTM.ELEVATION_MIN,
    COL_MAX_ELEVATION: DTM.ELEVATION_MAX,
    COL_STDDEV: DTM.STDEV,
    COL_VALUE_COUNT: DTM.VALUE_COUNT,
    COL_ELEVATION_SMOOTHED: DTM.ELEVATION_SMOOTHED_NAME,
    COL_INTER_FLAG: DTM.INTERPOLATION_FLAG,
    COL_CDI: DTM.CDI_INDEX,
    COL_CPRD: DTM.CDI_INDEX,
    COL_BACKSCATTER: DTM.BACKSCATTER,
}

# Default values by data type
COL_DEFAULT_VALUES: Dict[str, Any] = {"float": np.nan, "int": np.iinfo(np.int32).max, "CDI": -1}
