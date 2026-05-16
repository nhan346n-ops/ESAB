#! /usr/bin/env python3
# coding: utf-8


from pyat.utils.numpy_utils import (
    compute_standard_deviation_second_pass,
    compute_standard_deviation_first_pass,
    compute_statistics,
)

import numpy as np


def test_computing_standard_deviation():
    """
    Convert a LonLat tiff to a LonLat DTM.
    Raster is spanning the 180th meridian
    """
    in_values = np.array([4, 14, 7, 5, 9999.99, 11999.99, 999, np.nan], dtype=np.float32)
    in_values_double = np.float64(in_values)
    x_array = np.array([0, 0, 0, 0, 1, 1, 2, 3], dtype=int)
    y_array = np.copy(x_array)
    in_mean_array = np.full((4, 4), np.nan, dtype=np.float64)
    in_mean_array[0, 0] = np.mean(in_values_double[:4])
    in_mean_array[1, 1] = np.mean(in_values_double[4:6])
    in_mean_array[2, 2] = in_values_double[6]
    tmp_square_array = np.zeros(in_mean_array.shape, dtype=np.float64)
    out_stdev_array = np.zeros(in_mean_array.shape, dtype=np.float64)

    # Prepare the computing (here, out_stdev_array is filled with sum(value²))
    compute_standard_deviation_first_pass(in_values, x_array, y_array, tmp_square_array)
    assert tmp_square_array[0, 0] == np.square(in_values_double[:4]).sum()
    assert tmp_square_array[1, 1] == np.square(in_values_double[4:6]).sum()
    assert tmp_square_array[2, 2] == np.square(in_values_double[6])
    assert tmp_square_array[3, 3] == 0

    # Finalize the computing
    in_count_array = np.zeros(in_mean_array.shape, dtype=int)
    in_count_array[0, 0] = 4
    in_count_array[1, 1] = 2
    in_count_array[2, 2] = 1
    compute_standard_deviation_second_pass(in_count_array, in_mean_array, tmp_square_array, out_stdev_array)
    assert out_stdev_array[0, 0] == np.std([4, 14, 7, 5])
    assert out_stdev_array[1, 1] == np.std([9999.99, 11999.99])
    assert out_stdev_array[2, 2] == 0  # Only one value
    assert np.isnan(out_stdev_array[3, 3])  # Only Nan value


def test_compute_statistics_min_max():
    """
    Check compute_statistics for min/max computation
    """
    in_values = np.array([4, 14, 7, 5, 999, np.nan], dtype=float)
    x_array = np.array([0, 0, 0, 0, 1, 2], dtype=int)
    y_array = np.copy(x_array)
    out_min_array = np.full((3, 3), np.nan, dtype=float)
    out_max_array = np.full((3, 3), np.nan, dtype=float)

    # Finalize the computing
    compute_statistics(
        in_array=in_values, x_array=x_array, y_array=y_array, out_min_array=out_min_array, out_max_array=out_max_array
    )

    assert out_min_array[0, 0] == np.min([4, 14, 7, 5])
    assert out_min_array[1, 1] == 999
    assert np.isnan(out_min_array[2, 2])
    assert out_max_array[0, 0] == np.max([4, 14, 7, 5])
    assert out_max_array[1, 1] == 999
    assert np.isnan(out_max_array[2, 2])


def test_compute_statistics_mean():
    """
    Check compute_statistics for mean computation
    """
    in_values = np.array([4, 14, 7, 5, 9999, 10999.01, 999, np.nan], dtype=np.float32)
    x_array = np.array([0, 0, 0, 0, 1, 1, 2, 3], dtype=int)
    y_array = np.copy(x_array)
    out_mean_array = np.full((4, 4), np.nan, dtype=np.float32)
    out_count_array = np.full((4, 4), -1, dtype=int)

    # Finalize the computing
    compute_statistics(
        in_array=in_values,
        x_array=x_array,
        y_array=y_array,
        out_mean_array=out_mean_array,
        out_count_array=out_count_array,
    )

    assert out_count_array[0, 0] == 4
    assert out_count_array[1, 1] == 2
    assert out_count_array[2, 2] == 1
    assert out_count_array[3, 3] <= 0

    assert out_mean_array[0, 0] == np.mean([4, 14, 7, 5])
    assert out_mean_array[1, 1] == np.mean(np.float64([9999, np.float32(10999.01)]))
    assert out_mean_array[2, 2] == 999
    assert np.isnan(out_mean_array[3, 3])


def test_compute_statistics_filtered():
    """
    Check compute_statistics for filtered computation
    """
    in_values = np.array([4, 14, 7, 5, 999, np.nan], dtype=float)
    x_array = np.array([0, 0, 0, 0, 1, 2], dtype=int)
    y_array = np.copy(x_array)
    out_filtered_array = np.full((3, 3), -1, dtype=int)
    out_count_array = np.full((3, 3), -1, dtype=int)

    # Finalize the computing
    compute_statistics(
        in_array=in_values,
        x_array=x_array,
        y_array=y_array,
        out_filtered_array=out_filtered_array,
        out_count_array=out_count_array,
    )

    assert out_count_array[0, 0] == 4
    assert out_count_array[1, 1] == 1
    assert out_count_array[2, 2] <= 0
    assert out_filtered_array[0, 0] <= 0
    assert out_filtered_array[1, 1] <= 0
    assert out_filtered_array[2, 2] == 1
