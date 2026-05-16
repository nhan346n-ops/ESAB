#! /usr/bin/env python3
# coding: utf-8
from typing import NamedTuple

import numpy as np
import numpy.ma as ma
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from sklearn.ensemble import IsolationForest

import pyat.utils.pyat_logger as log

__logger = log.logging.getLogger("AutomaticFiltering")


class MappedFilesFilteringArg(NamedTuple):
    """
    Class representing all arguments for configuring the process

    :param inAbscissaFile: Binary input file containing the X values (projected latitudes)
    :param inOrdinateFile: Binary input file containing the Y values (projected longitudes)
    :param inDepthFile: Binary input file containing the Z values (depths)
    :param inValidityFile: Binary input file containing the validity of the soundings (1 == valid, 0 = invalid).
    :param outValidityFile: Copy of validity file modified with the result of the filtering. Values are 0 or 1 (original valid and invalid), 2 = validated, 3 = invalidated).

    :param contamination: The amount of contamination of the data set, i.e. the proportion of outliers in the data set (default is 0.5, range is ]0, 0.5])
    """

    inAbscissaFile: str
    inOrdinateFile: str
    inDepthFile: str
    inValidityFile: str
    outValidityFile: str

    contamination: float = 0.05
    monitor: ProgressMonitor = DefaultMonitor


def process(**kwargs) -> None:
    """
    Entry point for the automatic filtering process configured with dict of arguments
    Function accepting all arguments of the process as a dict. Possible arguments are listed in "MappedFilesFilteringArg" class
    """
    process_with_MappedFilesFilteringArg(MappedFilesFilteringArg(**kwargs))


def process_with_MappedFilesFilteringArg(args: MappedFilesFilteringArg) -> None:
    """
    Entry point for the automatic filtering process configured with MappedFilesFilteringArg
    Browsing input XSF files and applying the automatic filtering
    """
    __logger.info("Starting automatic filtering of raw files")

    __logger.info("Loading data from files")
    x = np.fromfile(args.inAbscissaFile, dtype=float)
    y = np.fromfile(args.inOrdinateFile, dtype=float)
    z = np.fromfile(args.inDepthFile, dtype=float)
    validity = np.fromfile(args.inValidityFile, dtype=np.uint8)

    new_validity = process_with_ndarray(x, y, z, validity, args.contamination)

    # Save new validity
    __logger.info("Saving new validities")
    new_validity.tofile(args.outValidityFile)


def process_with_ndarray(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, validity: np.ndarray, contamination: float
) -> np.ndarray:
    """
    Entry point for the automatic filtering process configured with ndarray (from XSF)
    """
    __logger.info("Starting automatic filtering of raw files")

    if contamination <= 0 or contamination > 0.5:
        raise ValueError("Contamination must be in the range ]0, 0.5]")

    masked_xyz = _mask_invalid_sounds(x, y, z, validity)
    flatten_masked_xyz = masked_xyz.reshape(-1, 3)
    flatten_validity = validity.reshape(-1)
    flatten_validity = _apply_filter(flatten_masked_xyz, flatten_validity, contamination)

    return flatten_validity.reshape(validity.shape)


def _mask_invalid_sounds(x: np.ndarray, y: np.ndarray, z: np.ndarray, validity: np.ndarray) -> np.ndarray:
    """
    Main function
    Browsing input XSF files and applying the automatic filtering
    """
    __logger.info("Masking invalid sounds")
    masked_x = ma.masked_where(validity != 1, x)
    masked_y = ma.masked_where(validity != 1, y)
    masked_z = ma.masked_where(validity != 1, z)
    return np.dstack((masked_x, masked_y, masked_z))


def _apply_filter(masked_xyz: np.ndarray, validity: np.ndarray, contamination: float) -> np.ndarray:
    __logger.info("Applying filter")
    clf = IsolationForest(contamination=contamination)
    labels = clf.fit_predict(masked_xyz)

    __logger.info(f"{np.sum(labels != 1)} sounds filtered")
    return np.where(labels == 1, validity, 2)
