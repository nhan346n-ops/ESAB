#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from numba import prange, njit


def find_distance(size):
    """Find the distance between cells.

    Arguments:
        size {int} -- Size of the search window.

    Returns:
        np.array
    """
    center = int(np.floor(size/2))
    distance = np.zeros((size, size))
    index = np.zeros(distance.shape)

    for row in range(size):
        for col in range(size):
            distance[row, col] = (row - center) ** 2 + (col - center) ** 2

    for count in range(size ** 2):
        e = np.where(distance == np.min(distance))
        if len(e[0]) > 1:
            e = (e[0][1], e[1][1])
        distance[e] = 99999
        index[e] = count

    return index


def find_coord(index):
    """Continuation of the find_distance method. Transform the numerotation into
    coordinates(x, y).

    Arguments:
        index {np.array} -- Result of the find_distance method.

    Returns:
        np.array -- array of coordinates.
    """
    coord = np.zeros((index.shape[0] ** 2, 2), dtype=int)
    center = np.where(index == 0)
    for count in range(1,index.size):
        a = np.where(index == count)
        coord[count, 0] = int(a[0] - center[0])
        coord[count, 1] = int(a[1] - center[1])

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
    rowSize = o_value.shape[0] - 2 * m_size
    colSize = o_value.shape[1] - 2 * m_size

    for row in prange(m_size, m_size + rowSize):
        for col in prange(m_size, m_size + colSize):
            if np.isnan(i_value[row, col]) and mask[row, col]:
                # Fill with nearest value
                for i in range(coord.shape[0]):
                    r = coord[i][0]
                    c = coord[i][1]
                    # Generic
                    element = (row + r, col + c)
                    if not np.isnan(i_value[element[0], element[1]]):
                        o_value[row, col] = i_value[element[0], element[1]]
                        break
            else:
                # No whole or outside the geographic zone. Elevation remains the same.
                o_value[row, col] = i_value[row, col]

    return o_value
