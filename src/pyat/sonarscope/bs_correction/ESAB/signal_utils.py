"""信号转换工具函数（从 pyat.utils.signal 独立提取，无其他依赖）"""

from numbers import Number
from typing import Union, Iterable

import numpy as np
from numpy import ndarray


def amplitude_to_db(value: Union[Number, ndarray, Iterable], *args, **kwargs) -> np.ndarray:
    """振幅转 dB: dB=20log10(amplitude)"""
    log_value = np.log10(value, *args, **kwargs)
    return np.multiply(log_value, 20, *args, **kwargs)


def energy_to_db(value: Union[Number, ndarray, Iterable], *args, **kwargs) -> np.ndarray:
    """能量转 dB: dB=10log10(energy)"""
    log_value = np.log10(value, *args, **kwargs)
    return np.multiply(log_value, 10, *args, **kwargs)


def db_to_amplitude(value: Union[Number, ndarray, Iterable]) -> np.ndarray:
    """dB 转振幅: amplitude=10^(dB/20)"""
    return 10 ** (value / 20)


def db_to_energy(value: Union[Number, ndarray, Iterable]) -> np.ndarray:
    """dB 转能量: energy=10^(dB/10)"""
    return 10 ** (value / 10)


def db_to_db_mean_amplitude(value_db: Union[Number, ndarray, Iterable], axis=None) -> np.ndarray:
    """计算振幅的算术均值（忽略 NaN）后转 dB"""
    return amplitude_to_db(np.nanmean(db_to_amplitude(np.asarray(value_db)), axis=axis))


def db_to_db_mean_energy(value_db: Union[Number, ndarray, Iterable], axis=None) -> np.ndarray:
    """计算能量的算术均值（忽略 NaN）后转 dB"""
    return energy_to_db(np.nanmean(db_to_energy(np.asarray(value_db)), axis=axis))
