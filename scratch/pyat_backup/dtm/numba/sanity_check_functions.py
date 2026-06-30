#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from numba import prange, njit


# Sanity check process
@njit(["int8[:,:], int8[:, :],float32[:,:], int8"], parallel=True, cache=True)
def update_interp(o_arr, i_arr, i_elev, m_val):
    """Optimized function numba for the calcul of the layer interp. Check for interpolation flag layer.
    For each cell, set it to zero (not interpolated) if depth value exists and if its interpoaltion
    flag is set to invalid value.
    Leave it unchanged if it is already set to a valid value.

    Arguments:
        o_arr {np.array} -- Output array.
        i_arr {np.array} -- Input array.
        i_elev {np.array} -- Input elevation array.

    Returns:
        [np.array] -- Output array.
    """
    for i in prange(o_arr.shape[0]):
        for j in prange(o_arr.shape[1]):
            if i_arr[i, j] != m_val:
                o_arr[i, j] = i_arr[i, j]
            elif not np.isnan(i_elev[i, j]):
                o_arr[i, j] = 0

    return o_arr


@njit(["int32[:,:], int32[:,:], float32[:,:], int64, int64"], parallel=True, cache=True)
def update_cdi_index(o_arr, i_arr, i_elev, o_count, i_count):
    """Optimized function numba for the calcul of the layer cdi_index. For each cell egual to
    i_count, set it to o_count if the cell is defined in the elevation layer.

    Arguments:
        o_arr {np.array} -- Output index cdi array.
        i_elev {np.array} -- Input elevation layer.

    Returns:
        [np.array] -- Output index cdi array.
    """
    for i in prange(o_arr.shape[0]):
        for j in prange(o_arr.shape[1]):
            if i_arr[i, j] == i_count and not np.isnan(i_elev[i, j]):
                o_arr[i, j] = o_count

    return o_arr
