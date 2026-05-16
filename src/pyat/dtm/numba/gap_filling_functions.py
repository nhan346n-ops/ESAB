#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from numba import prange, njit

import pyat.dtm.dtm_standard_constants as DtmConstant
from pyat.dtm import dtm_driver

NO_VALUE_COUNT = dtm_driver.get_missing_value(DtmConstant.VALUE_COUNT)
NO_CDI = dtm_driver.get_missing_value(DtmConstant.CDI_INDEX)


def find_distance(size):
    """Find the distance between cells.

    Arguments:
        size {int} -- Size of the search window.

    Returns:
        np.array -- List of 4 arrays.
    """
    result = np.zeros((4, size, size))
    temp = np.zeros((size, size))
    index = np.zeros(temp.shape)
    distance = np.zeros((size + 1, size + 1))

    for row in range(size + 1):
        for col in range(size + 1):
            distance[row, col] = (row ** 2 + col ** 2) ** 0.5
    temp = distance[1:, :size]

    for count in range(size ** 2):
        count += 1
        e = np.where(temp == np.min(temp))
        if len(e[0]) > 1:
            e = (e[0][1], e[1][1])
        temp[e] = 99
        index[e] = count

    result[0] = index
    result[1] = np.rot90(result[0])  # on tourne d'un quart de tour
    result[2] = np.rot90(result[1])  # on tourne d'un demi de tour
    result[3] = np.rot90(result[2])  # on tourne de trois quarts de tour

    return result


def find_coord(index):
    """Continuation of the find_distance method. Transform the numerotation into
    coordinates(x, y).

    Arguments:
        index {np.array} -- Result of the find_distance method.

    Returns:
        np.array -- List of array of coordinates.
    """
    coord = np.zeros((index.shape[0], index.shape[1] ** 2, 2), dtype=int)
    count = 0
    for q in range(index.shape[0]):
        for count in range(index[q].size):
            count += 1
            a = np.where(index[q] == count)
            coord[q, count - 1, 0] = int(a[0])
            coord[q, count - 1, 1] = int(a[1])

    return coord


@njit(
    [
        "int32(int32[:])",
    ],
    parallel=False,
    cache=True,
)
def argmax(data):
    if len(data) == 0:
        return NO_CDI
    seeked_index = 0
    max_value = data[0]
    for idx, val in enumerate(data):
        if val > max_value:
            seeked_index = idx
            max_value = val

    return seeked_index if max_value > 0 else NO_CDI


@njit(
    parallel=False,
    cache=True,
)
def interpolation(o_elev, i_elev, o_interp, o_cdi, o_val_count, index, m_size, mask):
    """For each mask value which respect condition, do the interpolation. Set the result
    to the corresponding layer.
    First step of the interpolation. The algorithme looks for in each quadran, the nearest cell
    with a correct value.
    Second step of the interpolation. The algorithm use the logic of
    bilinear interpolation.

    Arguments:
        o_elev {np.array} -- Array of the output elevation layer.
        i_elev {np.array} -- Array of the input elevation layer.
        o_interp {np.array} -- Array of the output interpolation_flag layer.
        o_cdi {np.array} -- Array of the output cdi_index layer.
        o_val_count {np.array} -- Array of the output value_count layer.
        index {np.array} -- Coordinates of the nearest point in function of the selected quadran.
        m_size {int} -- Size of the interpolated window.
        mask {np.array} -- Geo mask.

    Returns:
        np.array x 4 -- Array of elevation, interpolation_flag, cdi_index, value_count layers.
    """
    size = m_size - 1
    rowSize = o_elev.shape[0]
    colSize = o_elev.shape[1]
    rowSizeExt = rowSize + 2 * m_size
    colSizeExt = colSize + 2 * m_size

    n_cdis = int(np.max(o_cdi) + 1)

    # Number of quadran
    nbr = 4

    # Extend grids with interpolated window size
    o_elev_ext = np.full(shape=(rowSizeExt, colSizeExt), fill_value=np.nan)
    o_elev_ext[m_size:-m_size, m_size:-m_size] = o_elev
    o_elev = o_elev_ext
    i_elev_ext = np.full(shape=(rowSizeExt, colSizeExt), fill_value=np.nan)
    i_elev_ext[m_size:-m_size, m_size:-m_size] = i_elev
    i_elev = i_elev_ext
    o_interp_ext = np.full(shape=(rowSizeExt, colSizeExt), fill_value=np.nan)
    o_interp_ext[m_size:-m_size, m_size:-m_size] = o_interp
    o_interp = o_interp_ext
    o_cdi_ext = np.full(shape=(rowSizeExt, colSizeExt), fill_value=np.nan)
    o_cdi_ext[m_size:-m_size, m_size:-m_size] = o_cdi
    o_cdi = o_cdi_ext
    o_val_count_ext = np.full(shape=(rowSizeExt, colSizeExt), fill_value=np.nan)
    o_val_count_ext[m_size:-m_size, m_size:-m_size] = o_val_count
    o_val_count = o_val_count_ext
    mask_ext = np.full(shape=(rowSizeExt, colSizeExt), fill_value=np.nan)
    mask_ext[m_size:-m_size, m_size:-m_size] = mask
    mask = mask_ext

    for row in prange(m_size, m_size + rowSize):
        for col in prange(m_size, m_size + colSize):
            if np.isnan(i_elev[row, col]) and mask[row, col]:
                # Bouchage de trou par interpolation bilinéaire.
                # Interpolation step 1
                x = np.full(nbr, np.nan)
                y = np.full(nbr, np.nan)
                z = np.full(nbr, np.nan)

                cdi_frequencies = np.zeros(n_cdis, dtype=np.int32)
                cdis = np.full(nbr, -1)

                for q in prange(nbr):
                    ok = 0
                    for i in prange(index[q].shape[0]):
                        r = index[q][i][0]
                        c = index[q][i][1]

                        # Generic
                        if not ok:
                            element = None
                            if q == 0:
                                r += 1
                                element = (row + r, col + c)
                            elif q == 1:
                                c += 1
                                element = (row - (size - r), col + c)
                            elif q == 2:
                                r -= 1
                                element = (row - (size - r), col - (size - c))
                            elif q == 3:
                                c -= 1
                                element = (row + r, col - (size - c))

                            if element is not None:
                                el_row, el_col = element
                                if not np.isnan(i_elev[el_row, el_col]):
                                    x[q] = el_col
                                    y[q] = el_row
                                    z[q] = i_elev[el_row, el_col]
                                    cdis[q] = o_cdi[el_row, el_col]
                                    if cdis[q] >= 0:
                                        cdi_frequencies[int(cdis[q])] += 1
                                    ok = 1

                # Interpolation step 2
                value = np.nan

                # Calcul of coordinates of the intersection between the rectangle ABCD and axes.
                if x[1] != x[0] and x[0] != 0:
                    aab = (y[1] - y[0]) / (x[1] - x[0])
                    bab = (y[1] - x[1] / x[0] * y[0]) / (1 - x[1] / x[0])
                    if aab != 0:
                        xab = (row - bab) / aab
                    else:
                        xab = x[1]
                else:
                    xab = x[1]
                ybc = np.nan
                if x[2] != x[1] and x[1] != 0:
                    abc = (y[2] - y[1]) / (x[2] - x[1])
                    bbc = (y[2] - x[2] / x[1] * y[1]) / (1 - x[2] / x[1])
                    ybc = abc * col + bbc

                if x[2] != x[3] and x[2] != 0:
                    acd = (y[3] - y[2]) / (x[3] - x[2])
                    bcd = (y[3] - x[3] / x[2] * y[2]) / (1 - x[3] / x[2])
                    if acd != 0:
                        xcd = (row - bcd) / acd
                    else:
                        xcd = x[2]
                else:
                    xcd = x[2]

                if x[0] != x[3] and x[3] != 0:
                    ada = (y[0] - y[3]) / (x[0] - x[3])
                    bda = (y[0] - x[0] / x[3] * y[3]) / (1 - x[0] / x[3])
                    yda = ada * col + bda
                else:
                    yda = np.nan

                # Calcul of the value interpolated.
                zab = np.nan
                if y[0] != y[1]:
                    zab = z[0] + (z[1] - z[0]) * (row - y[0]) / (y[1] - y[0])
                zbc = np.nan
                if x[2] != x[1]:
                    zbc = z[1] + (z[2] - z[1]) * (col - x[1]) / (x[2] - x[1])
                zcd = np.nan
                if y[3] != y[2]:
                    zcd = z[2] + (z[3] - z[2]) * (row - y[2]) / (y[3] - y[2])
                zda = np.nan
                if x[0] != x[3]:
                    zda = z[3] + (z[0] - z[3]) * (col - x[3]) / (x[0] - x[3])

                if xcd - xab == 0 or yda - ybc == 0:
                    zabcd = np.nan
                    zbcda = np.nan
                else:
                    zabcd = zab + (zcd - zab) * (col - xab) / (xcd - xab)
                    zbcda = zbc + (zda - zbc) * (row - ybc) / (yda - ybc)

                value = (zabcd + zbcda) / 2

                # Calcul of the dominant cdi.
                cdi = argmax(cdi_frequencies)

                if not np.isnan(value):
                    o_elev[row, col] = value
                    o_cdi[row, col] = cdi
                    o_interp[row, col] = 1
                    o_val_count[row, col] = 0
                else:
                    o_val_count[row, col] = NO_VALUE_COUNT
                    o_cdi[row, col] = NO_CDI

            else:
                # No whole or outside the geographic zone. Elevation remains the same.
                o_elev[row, col] = i_elev[row, col]

    # returns back to original sizes
    o_elev = o_elev[m_size:-m_size, m_size:-m_size]
    o_interp = o_interp[m_size:-m_size, m_size:-m_size]
    o_cdi = o_cdi[m_size:-m_size, m_size:-m_size]
    o_val_count = o_val_count[m_size:-m_size, m_size:-m_size]

    return o_elev, o_interp, o_cdi, o_val_count
