#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from numba import prange, njit


@njit(
    [
        "int32[:,:], float32[:,:], float32, int64",
        "float32[:,:], float32[:,:], float32, int64",
        "int8[:,:], float32[:,:], float32, int64",
    ],
    parallel=True,
    cache=True,
)
def create_layer(o_arr, i_arr, m_val, mode):
    """Optimized function numba to copy from an array to another array.

    Arguments:
        o_arr {np.array} -- Output array of a layer.
        i_arr {np.array} -- Input array of a layer.

    Returns:
        [np.array] -- Output array of a layer.
    """
    for i in prange(o_arr.shape[0]):
        for j in prange(o_arr.shape[1]):
            if i_arr[i, j] != m_val and not np.isnan(i_arr[i, j]):
                if mode == 1:
                    o_arr[i, j] = 1
                else:
                    o_arr[i, j] = 0

    return o_arr
