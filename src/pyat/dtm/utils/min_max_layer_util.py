import netCDF4 as nc
import numpy as np


def update_min_max(elevation_layer: nc.Dataset, min_layer: nc.Dataset = None, max_layer: nc.Dataset = None):
    """
    update values for min max layers in order for them to always been lesser than max and higher than min

    """
    if min_layer is not None:
        return np.minimum(elevation_layer[:], min_layer[:])
    if max_layer is not None:
        return np.maximum(elevation_layer[:], max_layer[:])
    return elevation_layer[:]
