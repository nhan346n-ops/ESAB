import logging
from typing import List, Tuple

import numpy as np
import pytechsas.sensor.sensor_constant as sc
import xarray as xr

logger = logging.getLogger(__name__)


def read_elevation_range(sensor_data: List[xr.Dataset]) -> Tuple[float, float] | None:
    """
    Compute elevation range in the sensor data

    Returns:
        min and max elevations
    """

    # Compute elevation range from sensor data
    min_elevation = np.nan
    max_elevation = np.nan

    for sensor_ds in sensor_data:
        if sc.SENSOR_VAR_DEPTH not in sensor_ds.data_vars:
            logger.warning(f"Dataset missing depth variable: {sc.SENSOR_VAR_DEPTH}")
            continue

        # Convert depth to elevation (negative depth)
        elevations = -sensor_ds[sc.SENSOR_VAR_DEPTH].values

        # Update elevation range
        valid_elevations = elevations[~np.isnan(elevations)]
        if len(valid_elevations) > 0:
            max_elevation = np.nanmax([max_elevation, float(np.nanmax(valid_elevations))])
            min_elevation = np.nanmin([min_elevation, float(np.nanmin(valid_elevations))])

    # Check if we found valid elevation data
    if not np.isfinite(min_elevation) or not np.isfinite(max_elevation):
        logger.error("No valid elevation data found in sensor files")
        return None

    logger.info(f"Elevation range: [{min_elevation:.3f}, {max_elevation:.3f}] m")
    return (min_elevation, max_elevation)
