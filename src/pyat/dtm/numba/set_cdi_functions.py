#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from numba import prange, njit


@njit(
    ["int32[:,:], int32[:],int32", "int32[:,:], int64[:],int32"],
    parallel=True,
    cache=True,
)
def remap_cdi_index(cdi_index_values, cdi_map, missing_value):
    """Optimized function numba used to remap all cdi index values

    Arguments:
        cdi_index_values {np.array} -- Input array of a layer.
        cdi_map { np.array } -- map matching old index with new index values.

    Returns:
        [np.array] -- Output array of a layer.
    """
    max_index = len(cdi_map) - 1
    for i in prange(cdi_index_values.shape[0]):
        for j in prange(cdi_index_values.shape[1]):
            old_index = cdi_index_values[i, j]
            if old_index < 0 or old_index > max_index:
                cdi_index_values[i, j] = missing_value
            elif old_index != missing_value:
                cdi_index_values[i, j] = cdi_map[old_index]
    return cdi_index_values
