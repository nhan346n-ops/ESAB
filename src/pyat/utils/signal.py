"""Module for signal (backscatter) conversion"""

from numbers import Number
from typing import Union, Iterable

import numpy as np
from numpy import ndarray


def amplitude_to_db(value: Union[Number, ndarray, Iterable], *args, **kwargs) -> np.ndarray:
    """Convert from amplitude to dB the formula is dB=20log10(amplitude)"""
    log_value = np.log10(value, *args, **kwargs)
    return np.multiply(log_value, 20, *args, **kwargs)


def energy_to_db(value: Union[Number, ndarray, Iterable], *args, **kwargs) -> np.ndarray:
    """Convert from energy to dB the formula is dB=10log10(energy)"""
    log_value = np.log10(value, *args, **kwargs)
    return np.multiply(log_value, 10, *args, **kwargs)


def db_to_amplitude(value: Union[Number, ndarray, Iterable]) -> np.ndarray:
    """Convert from dB to amplitude the formula is amplitude=10^(dB/20)"""
    return 10 ** (value / 20)


def db_to_energy(value: Union[Number, ndarray, Iterable]) -> np.ndarray:
    """Convert from dB to energy the formula is energy=10^(dB/10)"""
    return 10 ** (value / 10)


def db_to_db_mean_amplitude(value_db: Union[Number, ndarray, Iterable], axis=None, where=np._NoValue) -> np.ndarray:
    """
    Compute the arithmetic mean of amplitude along the specified axis, ignoring NaNs.
    """
    return amplitude_to_db(np.nanmean(db_to_amplitude(value_db[:]), axis=axis, where=where))


def db_to_db_mean_energy(value_db: Union[Number, ndarray, Iterable], axis=None, where=np._NoValue) -> np.ndarray:
    """
    Compute the arithmetic mean of energy along the specified axis, ignoring NaNs.
    """
    return energy_to_db(np.nanmean(db_to_energy(value_db[:]), axis=axis, where=where))
