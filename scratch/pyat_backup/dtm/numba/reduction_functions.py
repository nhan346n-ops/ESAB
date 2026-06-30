#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from numba import prange, njit


# Reduction process
@njit(
    [
        "(int8[:,:], int8[:,:], int8, int64, int64, int32[:,:])",
        "(float32[:,:], float32[:,:], float32, int64, int64, int32[:,:])",
    ],
    parallel=True,
    cache=True,
)
def reduce_main(o_data, i_data, m_val, r_fact, power, i_vc):
    """Optimized function numba use for the reduction of the layers elevation, elevation_smooth
    et stdev. This function can switch between stdev and average with the factor power(1 for the
    average and 2 for the stdev). Utilisation of the weight for each cell.

    Arguments:
        o_data {np.array} -- Array(2d) of output data.
        i_data {np.array} -- Array(2d) of input data.
        m_val {} -- Value used for the missing value.
        r_fact {int} -- Reduction factor.
        power {int} -- 1 for elevation layers, 2 for stdev.
        i_vc {np.array} -- Array(2d) of input layer value count.

    Returns:
        np.array -- Reduced input array.
    """
    temp = np.full(o_data.shape, m_val)
    max_i_row = i_data.shape[0]
    max_i_col = i_data.shape[1]

    for row in prange(o_data.shape[0]):
        for col in prange(o_data.shape[1]):
            rowIn = int(round(row * r_fact))
            colIn = int(round(col * r_fact))
            count = 0

            for r in prange(r_fact):
                for c in prange(r_fact):
                    r_t = rowIn + r
                    c_t = colIn + c
                    if r_t < max_i_row and c_t < max_i_col:
                        if i_data[r_t, c_t] != m_val and not np.isnan(i_data[r_t, c_t]):
                            value_count = float(i_vc[r_t, c_t]) if i_vc[r_t, c_t] > 0 else 1.0
                            if temp[row, col] != m_val and not np.isnan(temp[row, col]):
                                temp[row, col] += (i_data[r_t, c_t] ** power) * value_count
                            else:
                                temp[row, col] = (i_data[r_t, c_t] ** power) * value_count
                            count += value_count

            if count != 0:
                temp[row, col] = (temp[row, col] / count) ** (1.0 / power)

    return temp


@njit(["(int32[:,:], int32[:,:], int32, int64)"], parallel=True, cache=True)
def reduce_value_count(o_data, i_data, m_val, r_fact):
    """Optimized function numba use for the reduction of the layer value_count.
    For each cell, the algorithm do the sum of all reduced cells.

    Arguments:
        o_data {np.array} -- Array(2d) of output data.
        i_data {np.array} -- Array(2d) of input data.
        m_val {} -- Value used for the missing value.
        r_fact {int} -- Reduction factor.

    Returns:
        np.array -- Reduced input array.
    """
    temp = np.full(o_data.shape, m_val)

    max_i_row = i_data.shape[0]
    max_i_col = i_data.shape[1]

    for row in prange(o_data.shape[0]):
        for col in prange(o_data.shape[1]):
            rowIn = int(round(row * r_fact))
            colIn = int(round(col * r_fact))

            for r in prange(r_fact):
                for c in prange(r_fact):
                    r_t = rowIn + r
                    c_t = colIn + c
                    if r_t < max_i_row and c_t < max_i_col:
                        cond_in = i_data[r_t, c_t] != m_val and not np.isnan(i_data[r_t, c_t])
                        cond_out = temp[row, col] != m_val and not np.isnan(temp[row, col])
                        if cond_in and cond_out:
                            temp[row, col] += i_data[r_t, c_t]
                        else:
                            temp[row, col] = i_data[r_t, c_t]

    return temp


@njit(["(float32[:,:], float32[:,:],float32, int32, int8)"], parallel=True, cache=True)
def reduce_min_max(o_data, i_data, m_val, r_fact, is_max):
    """Optimized function numba for the reduction of layers elevation min/max. The param is_max is the
    condition parameter to switch between the calcul of min and the calcul of max.

    Arguments:
        o_data {np.array} -- Array(2d) of output data.
        i_data {np.array} -- Array(2d) of input data.
        m_val {} -- Value used for the missing value.
        r_fact {int} -- Reduction factor.
        is_max{int} -- 1 for max and 0 for min.

    Returns:
        np.array -- Reduced input array.
    """
    temp = np.full(o_data.shape, m_val)
    max_i_row = i_data.shape[0]
    max_i_col = i_data.shape[1]

    for row in prange(o_data.shape[0]):
        for col in prange(o_data.shape[1]):
            rowIn = int(round(row * r_fact))
            colIn = int(round(col * r_fact))

            for r in prange(r_fact):
                for c in prange(r_fact):
                    r_t = rowIn + r
                    c_t = colIn + c
                    if r_t < max_i_row and c_t < max_i_col:
                        if i_data[r_t, c_t] != m_val and not np.isnan(i_data[r_t, c_t]):
                            if is_max:
                                temp[row, col] = np.nanmax((i_data[r_t, c_t], temp[row, col]))
                            else:
                                temp[row, col] = np.nanmin((i_data[r_t, c_t], temp[row, col]))

    return temp


@njit(["(int32[:,:], int32[:,:], int32[:,:], int32, int64)"], parallel=True, cache=True)
def reduce_cdi(o_data, i_data, i_cv, m_val, r_fact):
    """Optimized function numba use for the reduction of the layer cdi. Set the output cell
    to the input cell with the bigger number of sonds.

    Arguments:
        o_data {np.array} -- Array(2d) of output data.
        i_data {np.array} -- Array(2d) of input data.
        i_vc {np.array} -- Array(2d) of input layer value count.
        m_val {} -- Value used for the missing value.
        r_fact {int} -- Reduction factor.

    Returns:
        np.array -- Reduced input array.
    """
    temp = np.full(o_data.shape, m_val)
    max_i_row = i_data.shape[0]
    max_i_col = i_data.shape[1]

    for row in prange(o_data.shape[0]):
        for col in prange(o_data.shape[1]):
            rowIn = int(round(row * r_fact))
            colIn = int(round(col * r_fact))
            weight = 0

            for r in prange(r_fact):
                for c in prange(r_fact):
                    r_t = rowIn + r
                    c_t = colIn + c
                    if r_t < max_i_row and c_t < max_i_col:
                        cond_in = i_data[r_t, c_t] != m_val and not np.isnan(i_data[r_t, c_t])

                        if cond_in and i_cv[r_t, c_t] >= weight:
                            weight = i_cv[r_t, c_t]
                            temp[row, col] = i_data[r_t, c_t]

    return temp


@njit(["(int8[:,:], int8[:,:] , int8, int64)"], parallel=True, cache=True)
def reduce_interpolation_flag(o_data, i_data, m_val, r_fact):
    """Optimized function numba use for the reduction of the layer interpolation_flag. Set the output cell
    not_interpolated (=0) if one of cells is not interpolated (=0).

    Arguments:
        o_data {np.array} -- Array(2d) of output data.
        i_data {np.array} -- Array(2d) of input data.
        m_val {} -- Value used for the missing value.
        r_fact {int} -- Reduction factor.

    Returns:
        np.array -- Reduced input array.
    """
    temp = np.full(o_data.shape, m_val)
    max_i_row = i_data.shape[0]
    max_i_col = i_data.shape[1]

    for row in prange(o_data.shape[0]):
        for col in prange(o_data.shape[1]):
            rowIn = int(round(row * r_fact))
            colIn = int(round(col * r_fact))

            for r in prange(r_fact):
                for c in prange(r_fact):
                    r_t = rowIn + r
                    c_t = colIn + c
                    if r_t < max_i_row and c_t < max_i_col:
                        cond_in = i_data[r_t, c_t] != m_val and not np.isnan(i_data[r_t, c_t])
                        cond_out = temp[row, col] == m_val or temp[row, col] == 1
                        if cond_in and cond_out:
                            temp[row, col] = i_data[r_t, c_t]
    return temp
