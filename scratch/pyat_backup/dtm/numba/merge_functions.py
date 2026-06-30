#! /usr/bin/env python3
# coding: utf-8
from enum import IntEnum
from math import ceil

import numpy as np
from numba import prange, njit


# Merge process
@njit(parallel=True, cache=True)
def merge_project(i_lat, o_lat, i_lon, o_lon, i_data, m_val, mask):
    """Optimized function numba which project an array. Calcul of the shilft of lat/lon.
    Reshape possible.

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
    rowOffset = round((o_lat[0] - 0.5 * resoLat - i_lat[0]) / resoLat, precision)

    resoLon = round(i_lon[1] - i_lon[0], precision)
    colOffset = round((o_lon[0] - 0.5 * resoLon - i_lon[0]) / resoLon, precision)

    rowScale = round((o_lat[1] - o_lat[0]) / resoLat, precision)
    colScale = round((o_lon[1] - o_lon[0]) / resoLon, precision)

    o_data_shape_0 = o_lat.shape[0]
    o_data_shape_1 = o_lon.shape[0]
    temp = np.full((o_data_shape_0, o_data_shape_1), m_val, dtype=i_data.dtype)

    rowSize = i_data.shape[0]
    colSize = i_data.shape[1]

    for row in prange(o_data_shape_0):
        for col in prange(o_data_shape_1):
            if mask[row, col]:
                rowIn = ceil(row * rowScale + rowOffset)
                colIn = ceil(col * colScale + colOffset)

                if 0 <= rowIn < rowSize and 0 <= colIn < colSize:
                    # Check
                    if rowIn > 0 and colIn > 0:
                        if abs(i_lat[rowIn - 1] - o_lat[row]) < abs(i_lat[rowIn] - o_lat[row]):
                            rowIn -= 1
                        if abs(i_lon[colIn - 1] - o_lon[col]) < abs(i_lon[colIn] - o_lon[col]):
                            colIn -= 1
                    temp[row, col] = i_data[rowIn, colIn]
    return temp


@njit(
    ["float32[:,:], float32[:,:], float32", "int32[:,:], int32[:,:], int32", "int8[:,:], int8[:,:], int8"],
    parallel=True,
    cache=True,
)
def merge_fill(o_arr, i_arr, m_val):
    """Optimized function numba used for the merge "FILL". Just copy values, if the output
    cell is not defined.

    Arguments:
        o_arr {np.array} -- Output array of a layer.
        i_arr {np.array} -- Input array of a layer.
        m_val {} -- Value used for the missing value.

    Returns:
        [np.array] -- Output array of a layer.
    """
    for i in prange(o_arr.shape[0]):
        for j in prange(o_arr.shape[1]):
            # TODO : Check if the condition condIn is really necessary
            condIn = i_arr[i, j] != m_val and not np.isnan(i_arr[i, j])
            condOut = o_arr[i, j] == m_val or np.isnan(o_arr[i, j])
            if condIn and condOut:
                o_arr[i, j] = i_arr[i, j]

    return o_arr


@njit(["float32[:,:], float32[:,:], int8"], parallel=True, cache=True)
def merge_simple_max_min(o_arr, i_arr, is_max):
    """Optimized function numba used in the merge SIMPLE for the min/max elevation layers.

    Arguments:
        o_arr {np.array} -- Output array of a layer.
        i_arr {np.array} -- Input array of a layer.
        i_max {int} -- 1 for max, 0 for min.

    Returns:
        [np.array] -- Output array of a layer.
    """
    for i in prange(o_arr.shape[0]):
        for j in prange(o_arr.shape[1]):
            if not (np.isnan(i_arr[i, j]) or np.isnan(o_arr[i, j])):
                if is_max == 1:
                    o_arr[i, j] = max((i_arr[i, j], o_arr[i, j]))
                else:
                    o_arr[i, j] = min((i_arr[i, j], o_arr[i, j]))
            elif not np.isnan(i_arr[i, j]) and np.isnan(o_arr[i, j]):
                o_arr[i, j] = i_arr[i, j]
    return o_arr


class SlopeOperation(IntEnum):
    INTERPOLATION = 0
    MIN = 1
    MAX = 2
    SUM = 3
    DOMINANT = 4


@njit(
    [
        "float32[:,:], float32[:,:], float32[:,:],float32[:,:], float64, float64, float64, int32",
        "int32[:,:], int32[:,:], int32[:,:],float32[:,:], float64, float64, int32, int32",
        "int64[:,:], int32[:,:], int32[:,:],float32[:,:], float64, float64, int64, int64",
        "int8[:,:], int8[:,:], int8[:,:],float32[:,:], float64, float64, int32, int32",
    ],
    parallel=True,
    cache=True,
)
def merge_operation_with_slope(
    output_array, first_array, second_array, slope_array, min_slope_value, max_slope_value, invalid_value, operation
):
    """ "
    If the conditions are
    respected, then the output cell is set to

    output = first_array if slope < min_slope_value
    output = second_array if slope > max_slope_value
    in other case
        output = op(f) with f= (slope-min_slope_value)(max_slope_value-min_slope_value) and op the operation considered
    output = invalid_value is slope is invalid_value
    """
    # arrays are supposed to have the same size
    for i in prange(output_array.shape[0]):
        for j in prange(output_array.shape[1]):
            if np.isnan(slope_array[i, j]):
                output_array[i, j] = first_array[i, j]
            elif slope_array[i, j] < min_slope_value:
                output_array[i, j] = first_array[i, j]
            elif slope_array[i, j] > max_slope_value:
                output_array[i, j] = second_array[i, j]
            elif first_array[i, j] == invalid_value or np.isnan(first_array[i, j]):
                output_array[i, j] = second_array[i, j]
            elif second_array[i, j] == invalid_value or np.isnan(second_array[i, j]):
                output_array[i, j] = first_array[i, j]
            else:
                if operation == SlopeOperation.INTERPOLATION:
                    f = (slope_array[i, j] - min_slope_value) / (max_slope_value - min_slope_value)
                    output_array[i, j] = f * first_array[i, j] + (1 - f) * second_array[i, j]
                elif operation == SlopeOperation.SUM:
                    output_array[i, j] = first_array[i, j] + second_array[i, j]
                elif operation == SlopeOperation.MIN:
                    output_array[i, j] = min(first_array[i, j], second_array[i, j])
                elif operation == SlopeOperation.MAX:
                    output_array[i, j] = max(first_array[i, j], second_array[i, j])
                elif operation == SlopeOperation.MAX:
                    output_array[i, j] = max(first_array[i, j], second_array[i, j])
                elif operation == SlopeOperation.DOMINANT:
                    f = (slope_array[i, j] - min_slope_value) / (max_slope_value - min_slope_value)
                    if f < 0.5:
                        output_array[i, j] = first_array[i, j]
                    else:
                        output_array[i, j] = second_array[i, j]
                else:
                    output_array[i, j] = invalid_value
    return output_array


@njit(
    [
        "float32[:,:], float32[:,:], float64[:,:], float32, int32",
        "int32[:,:], int32[:,:], float64[:,:], int32, float64",
    ],
    parallel=True,
    cache=True,
)
def merge_simple_1(o_arr, i_arr, count, m_val, factor):
    """Optimized function numba used in the merge SIMPLE. If the conditions are
    respected, then the output cell is set to

    output += input ** 1 (for the average)
    output += input ** 2 (for the standard deviation)

    Arguments:
        o_arr {np.array} -- Output array of a layer.
        i_arr {np.array} -- Input array of a layer.
        count {np.array} -- count used for the division (Second pass) .
        m_val {} -- the value used for the missing value.
        factor {float} -- Possibility to calculate the average (1) or the stdev (2).

    Returns:
        [np.array] -- Output array of a layer.
        [np.array] -- count used for the division (Second passage).
    """
    for i in prange(o_arr.shape[0]):
        for j in prange(o_arr.shape[1]):
            condIn = i_arr[i, j] != m_val and not np.isnan(i_arr[i, j])
            condOut = o_arr[i, j] != m_val and not np.isnan(o_arr[i, j])
            if condIn:
                count[i, j] = count[i, j] + 1
                if condOut:
                    o_arr[i, j] = o_arr[i, j] + (i_arr[i, j]) ** factor
                else:
                    o_arr[i, j] = (i_arr[i, j]) ** factor

    return o_arr, count


@njit(["float32[:,:], float64[:,:], float32, int64, int32"], parallel=True, cache=True)
def merge_simple_2(o_arr, count, m_val, sonde_max, factor):
    """Optimized function numba used in the merge SIMPLE. If the conditions are
    respected, then the output cell is set to

    output = output / count (for the average)
    output = output / count ** (1/2) (for the standard deviation)

    If the sonds number is sup than sonde_max. Set the output to np.nan.

    Arguments:
        o_arr {np.array} -- Output array of a layer.
        count {np.array} -- for each value, amount of file used for the merge .
        m_val {} -- the value used for the missing value.
        sonde_max {int} -- Number of the sonde max. If the count value is sup of it, output = np.nan.
        factor {float} -- Possibility to calculate the average (1) or the stdev (2).

    Returns:
        np.array -- Output array of a layer.
    """
    for i in prange(o_arr.shape[0]):
        for j in prange(o_arr.shape[1]):
            condCount = count[i, j] > 0 and count[i, j] < sonde_max
            condOut = o_arr[i, j] != m_val or not np.isnan(o_arr[i, j])
            if condCount and condOut:
                o_arr[i, j] = (o_arr[i, j] / count[i, j]) ** (1.0 / factor)
            else:
                o_arr[i, j] = np.nan

    return o_arr


@njit(["int8[:,:], int8[:,:], int8"], parallel=True, cache=True)
def merge_simple_interpolation_flag(o_arr, i_arr, m_val):
    """Optimized function numba used for the merge SIMPLE. Set to 1 if one of
    the cell was interpolated.

    Arguments:
        o_arr {np.array} -- Output array of a layer.
        i_arr {np.array} -- Input array of a layer.
        m_val {} -- Value used for the missing value.

    Returns:
        np.array -- Output array of a layer.
    """
    for i in prange(o_arr.shape[0]):
        for j in prange(o_arr.shape[1]):
            if (o_arr[i, j] == m_val and i_arr[i, j] != m_val) or i_arr[i, j] == 1:
                o_arr[i, j] = i_arr[i, j]

    return o_arr


@njit(["int32[:,:], int32[:,:], int64, int64, int32"], parallel=True, cache=True)
def merge_fill_cdi_index(i_arr, o_arr, i_count, o_count, m_val):
    """Optimized function numba used in the merge FILL for the cdi_index layer.
    For each cdi, set the output value by the index of the input value.
    The move of the cdi in the output file is implemented.

    Arguments:
        i_arr {np.array} -- Input array of the layer.
        o_arr {np.array} -- Output array of the layer.
        i_count {[type]} -- Number of the processing cdi in the input layer.
        o_count {[type]} -- Number of the processing cdi in the output layer.
        m_val {} -- Value used for the missing value.

    Returns:
       {np.array} -- Output array of the layer.
       {int} -- Amount of points corresponding to the processing cdi.
    """
    for i in prange(o_arr.shape[0]):
        for j in prange(o_arr.shape[1]):
            if i_arr[i, j] == i_count and o_arr[i, j] == m_val:
                o_arr[i, j] = o_count
    return o_arr


@njit(["int32[:,:], int32[:,:], int32[:,:], float64[:,:], int64, int64, int32"], parallel=True, cache=True)
def merge_simple_cdi_index(i_arr, o_arr, i_value_count_arr, o_value_count_arr, i_count, o_count, m_val):
    """Optimized function numba used in the merge SIMPLE for the cdi_index layer.
    The logic is to set to the output value, the dominant cdi.

    Arguments:
        i_arr {array} -- Array of the layer cdi_index of input data.
        o_arr {array} -- Array of the layer cdi_index of output data.
        i_value_count_arr {array} -- Array of the layer value_count of input data (ex VSOUNDINGS).
        o_value_count_arr {array} -- Array of the layer value_count of output data (ex VSOUNDINGS).
        i_count {int} -- Number in cdi_reference in the input file.
        o_count {int} -- Number in cdi_reference in the output file.
        m_val {int} -- Masked value. (-1)

    Returns:
        array -- Output data.
    """

    for i in prange(o_arr.shape[0]):
        for j in prange(o_arr.shape[1]):
            if i_arr[i, j] == i_count:
                majoritaire = False
                if o_value_count_arr[i, j] != m_val:
                    majoritaire = o_value_count_arr[i, j] <= i_value_count_arr[i, j]

                if o_arr[i, j] == m_val or majoritaire:
                    o_arr[i, j] = o_count
                    o_value_count_arr[i, j] = i_value_count_arr[i, j]

    return o_arr, o_value_count_arr
