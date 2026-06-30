import numpy as np


def floatsecond_tonano_array(value: np.array):
    """we try to minimize rounding issues by first upgrading float32 time stamps to float64
    with this we expect to minimize differences and rounding issues
    :param value the input values as an array of float in second
    :return a new allocated array to values in nanosecond
    """
    return (value.astype(np.float64) * int(1e9)).astype(np.uint64, copy=False)
