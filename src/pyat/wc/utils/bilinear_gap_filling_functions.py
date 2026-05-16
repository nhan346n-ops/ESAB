#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from numba import prange, njit


@njit(
    parallel=False,
    cache=True,
)
def find_distance(size):
    """Find the distance between cells.

    Arguments:
        size {int} -- Size of the search window.

    Returns:
        np.array -- List of 4 arrays.
    """
    result = np.zeros((np.int32(4), np.int32(size), np.int32(size)), dtype=np.int32)
    temp = np.zeros((np.int32(size), np.int32(size)), dtype=np.int32)
    index = np.zeros(temp.shape)
    distance = np.zeros((np.int32(size + 1), np.int32(size + 1)), dtype=np.int32)
    max_distance = np.int32(2*(size + 1)**2)

    for row in range(size + 1):
        for col in range(size + 1):
            distance[row, col] = (row ** 2 + col ** 2)
    temp = distance[1:, :size]

    for count in range(size ** 2):
        e = np.nonzero(temp == np.min(temp))
        e = (e[0][0], e[1][0])
        temp[e] = max_distance
        index[e] = count

    result[0] = index
    result[1] = np.rot90(result[0])  # on tourne d'un quart de tour
    result[2] = np.rot90(result[1])  # on tourne d'un demi de tour
    result[3] = np.rot90(result[2])  # on tourne de trois quarts de tour
    return result


@njit(
    parallel=False,
    cache=True,
)
def find_coord(index):
    """Continuation of the find_distance method. Transform the numerotation into
    coordinates(x, y).

    Arguments:
        index {np.array} -- Result of the find_distance method.

    Returns:
        np.array -- List of array of coordinates.
    """
    coord = np.zeros((index.shape[0], index.shape[1] ** 2, 2), dtype=np.int32)
    count = 0
    for q in range(index.shape[0]):
        for count in range(index[q].size):
            a = np.nonzero(index[q] == count)
            coord[q, count, 0] = np.int32(a[0][0])
            coord[q, count, 1] = np.int32(a[1][0])

    return coord


@njit(
    parallel=True,
    cache=True,
)
def interpolation(o_value, i_value, coord, m_size, mask):
    """For each mask value which respect condition, do the interpolation. Set the result
    to the corresponding layer.
    First step of the interpolation. The algorithme looks for in each quadran, the nearest cell
    with a correct value.
    Second step of the interpolation. The algorithm use the logic of
    bilinear interpolation.

    Arguments:
        o_value {np.array} -- Array of the output layer.
        i_value {np.array} -- Array of the input layer.
        coord {np.array} -- Coordinates of the nearest point in function of the selected quadran.
        m_size {int} -- Size of the interpolated window.
        mask {np.array} -- Geo mask.

    Returns:
        np.array x 1 -- Array of values.
    """
    size = m_size - 1
    rowSize = o_value.shape[0] - 2 * m_size
    colSize = o_value.shape[1] - 2 * m_size

    # Number of quadran
    nbr = 4

    for row in prange(m_size, m_size + rowSize):
        for col in prange(m_size, m_size + colSize):
            if np.isnan(i_value[row, col]) and mask[row, col]:
                # Fill gaps by bilinear interpolation
                # Interpolation step 1 : find closest point by quadran
                x = np.full(nbr, np.nan)
                y = np.full(nbr, np.nan)
                z = np.full(nbr, np.nan)

                for q in prange(nbr):
                    for i in range(coord.shape[1]):
                        r = coord[q][i][0]
                        c = coord[q][i][1]
                        # Generic
                        if q == 0:
                            r += 1
                            element = (row + r, col + c)
                        elif q == 1:
                            c += 1
                            element = (row - (size - r), col + c)
                        elif q == 2:
                            r -= 1
                            element = (row - (size - r), col - (size - c))
                        else: #if q == 3:
                            c -= 1
                            element = (row + r, col - (size - c))

                        if not np.isnan(i_value[element[0], element[1]]):
                            x[q] = element[1]
                            y[q] = element[0]
                            z[q] = i_value[element[0], element[1]]
                            break

                # Interpolation step 2
                # Check presence of element in each quadrant not too far each other
                if not np.any(np.isnan(z)) and (np.max(x) - np.min(x) <= m_size) and (np.max(y) - np.min(y) <= m_size):
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

                    if x[2] != x[1] and x[1] != 0:
                        abc = (y[2] - y[1]) / (x[2] - x[1])
                        bbc = (y[2] - x[2] / x[1] * y[1]) / (1 - x[2] / x[1])
                        ybc = abc * col + bbc
                    else:
                        ybc = y[1]

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
                        yda = y[0]

                    # Calcul of the value interpolated.
                    zab = z[0]
                    if y[0] != y[1]:
                        zab = z[0] + (z[1] - z[0]) * (row - y[0]) / (y[1] - y[0])
                    zbc = z[1]
                    if x[2] != x[1]:
                        zbc = z[1] + (z[2] - z[1]) * (col - x[1]) / (x[2] - x[1])
                    zcd = z[2]
                    if y[3] != y[2]:
                        zcd = z[2] + (z[3] - z[2]) * (row - y[2]) / (y[3] - y[2])
                    zda = z[3]
                    if x[0] != x[3]:
                        zda = z[3] + (z[0] - z[3]) * (col - x[3]) / (x[0] - x[3])

                    if xcd - xab == 0:
                        zabcd = zab
                    else:
                        zabcd = zab + (zcd - zab) * (col - xab) / (xcd - xab)

                    if yda - ybc == 0:
                        zbcda = zbc
                    else:
                        zbcda = zbc + (zda - zbc) * (row - ybc) / (yda - ybc)

                    o_value[row, col] = (zabcd + zbcda) / 2
            else:
                # No whole or outside the geographic zone. Elevation remains the same.
                o_value[row, col] = i_value[row, col]

    return o_value
