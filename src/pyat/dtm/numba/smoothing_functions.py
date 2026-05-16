#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from numba import prange, njit


# Smoothing Process
@njit(
    ["float32[:,:], float32[:,:], uint8[:,:], int32, int32", "int32[:,:], float32[:,:], uint8[:,:], int32, int32"],
    parallel=True,
    cache=True,
)
def smoothing(o_arr, i_arr, mask, rowSize, colSize):
    """Optimized function numba for the smoothing. Do an average between all the cells
    included in the window.

    Arguments:
        o_arr {np.array} -- Output smoothed elevation array.
        i_arr {np.array} -- Input elevation layer.
        mask {np.array} -- Mask geographical array.
        rowSize {np.array} -- Size of the smoothed window.
        colSize {np.array} -- Size of the smoothed window.

    Returns:
        [np.array] -- Output smoothed elevation array.
    """
    for i in prange(o_arr.shape[0]):
        for j in prange(o_arr.shape[1]):
            if mask[i, j] and not np.isnan(i_arr[i, j]):
                rowOffset = int((rowSize - 1) / 2)
                colOffset = int((colSize - 1) / 2)
                value = 0
                count = 0

                for row in prange(rowSize):
                    for col in prange(colSize):
                        r = i + row - rowOffset
                        c = j + col - colOffset
                        if (
                            0 <= r < o_arr.shape[0]
                            and 0 <= c < o_arr.shape[1]
                            and not np.isnan(i_arr[r, c])
                        ):
                            value += i_arr[r, c]
                            count += 1
                o_arr[i, j] = value / count
            else:
                o_arr[i, j] = i_arr[i, j]

    return o_arr
