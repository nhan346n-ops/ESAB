#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from numba import prange, njit


# Reset cells process
@njit(
    [
        "float32[:,:], float32[:,:], float32, uint8[:,:]",
        "int32[:,:], int32[:,:], int32, uint8[:,:]",
        "int8[:,:], int8[:,:], int8, uint8[:,:]",
        "float32[:,:], float32[:,:], float32, boolean[:,:]",
        "int32[:,:], int32[:,:], int32, boolean[:,:]",
        "int8[:,:], int8[:,:], int8, boolean[:,:]",
    ],
    parallel=True,
    cache=True,
)
def reset_layer(o_arr, i_arr, m_val, mask):
    """Optimized function numba for all layers. Each selected cell whose the mask = 1, is set
    to m_val. Else just copy le cell from input to output.

    Arguments:
        o_arr {np.array} -- Output array of a layer.
        i_arr {np.array} -- Input array of a layer.
        m_val { - 1 } -- Missing value.
        mask {np.array} -- Mask array.

    Returns:
        [np.array] -- Output array of a layer.
    """
    for i in prange(o_arr.shape[0]):
        for j in prange(o_arr.shape[1]):
            if mask[i, j]:
                o_arr[i, j] = m_val
            else:
                o_arr[i, j] = i_arr[i, j]

    return o_arr


# set cells process
@njit(
    [
        "float32[:,:], float32[:,:], float32, float32, boolean[:,:]",
        "int32[:,:], int32[:,:], int32, int32, boolean[:,:]",
        "int8[:,:], int8[:,:], int8, int8, boolean[:,:]",
    ],
    parallel=True,
    cache=True,
)
def set_layer(o_arr, i_arr, val, m_val, mask):
    """Optimized function numba for all layers. Each selected cell whose the mask = true and not missing, is set
    to val. Else just copy le cell from input to output.

    Arguments:
        o_arr {np.array} -- Output array of a layer.
        i_arr {np.array} -- Input array of a layer.
        val { - 1 } -- Set value.
        m_val { - 1 } -- Missing value.
        mask {np.array} -- Mask array.

    Returns:
        [np.array] -- Output array of a layer.
    """
    for i in prange(o_arr.shape[0]):
        for j in prange(o_arr.shape[1]):
            if mask[i, j] and i_arr[i, j] != m_val and not np.isnan(i_arr[i, j]):
                o_arr[i, j] = val
            else:
                o_arr[i, j] = i_arr[i, j]

    return o_arr
