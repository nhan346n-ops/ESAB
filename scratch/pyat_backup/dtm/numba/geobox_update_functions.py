#! /usr/bin/env python3
# coding: utf-8

from math import floor

import numpy as np
from numba import prange, njit


# Geo box update process
@njit(
    [
        "(float64[:], float64[:], float64[:], float64[:], float32[:,:], float32[:,:],float32)",
        "(float64[:], float64[:], float64[:], float64[:], int32[:,:], int32[:,:],int32)",
        "(float64[:], float64[:], float64[:], float64[:], int8[:,:], int8[:,:],int8)",
    ],
    parallel=True,
    cache=True,
)
def project(i_lat, o_lat, i_lon, o_lon, o_data, i_data, m_val):
    """Optimized function numba which project an array.

    Arguments:
        i_lat {np.array} -- array(1d) of input latittude.
        o_lat {np.array} -- array(1d) of output latittude.
        i_lon {np.array} -- array(1d) of input longitude.
        o_lon {np.array} -- array(1d) of output longitude.
        o_data {np.array} -- array(2d) of output data.
        i_data {np.array} -- array(2d) of input data.
        m_val {} -- the value used for the missing value.

    Returns:
        np.array -- the input array with the good lat, lon and size.
    """
    precision = 10
    resoLat = round(i_lat[1] - i_lat[0], precision)
    rowOffset = int(floor(abs(o_lat[0] - i_lat[0]) / resoLat))

    resoLon = round(i_lon[1] - i_lon[0], precision)
    colOffset = int(floor(abs(o_lon[0] - i_lon[0]) / resoLon))

    temp = np.full(o_data.shape, m_val)

    rowSize = i_data.shape[0]
    colSize = i_data.shape[1]

    for row in prange(o_data.shape[0]):
        for col in prange(o_data.shape[1]):
            rowIn = row + rowOffset
            colIn = col + colOffset

            if 0 <= rowIn < rowSize and 0 <= colIn < colSize:
                temp[row, col] = i_data[rowIn, colIn]

    return temp
