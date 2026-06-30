#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import pyat.xyz.xyz_constants as XyzConstants
from pyat.xyz.xyz_file import XyzFile
import pyat.utils.numpy_utils as NumpyUtils


class XyzDriver:
    """Class utility for xyz file."""

    def computeMinMax(self, xyz_arr: np.array, xyz_file: XyzFile):
        """Compute the min and the max of a given array for each columns.

        Arguments:
            xyz_arr {np.array} -- array of xyz file (result of pandas.read_csv())
            xyz_file {XyzFile} -- object Xyzfile
        """
        emoMinMax = np.full((2, 2), np.nan)
        NumpyUtils.minMaxOnFloat(xyz_arr, emoMinMax)
        for colIndex in range(emoMinMax.shape[1]):
            if not np.isnan(emoMinMax[0, colIndex]) and not np.isnan(emoMinMax[1, colIndex]):
                column = XyzFile.ColumnNames[colIndex]
                xyz_file.minmax[column] = [
                    np.nanmin([xyz_file.min(column), emoMinMax[0, colIndex]]),
                    np.nanmax([xyz_file.max(column), emoMinMax[1, colIndex]]),
                ]

    def check_file(self, file_path: str) -> XyzFile:
        """
        Verify that filePath is a suitable Xyz file

        Raised exception :  OSError when file does not exist, is not readable or is not a suitable Xyz file
        """
        result = XyzFile(file_path=file_path)

        return result

    def parse(self, xyz_file: XyzFile, chunksize: int = 10 ** 6) -> np.array:
        """Parse the XYZ file with Pandas and return a TextFileReader

        Arguments:
            xyz_file {XyzFile} -- class XyzFile of a file xyz file.

        Keyword Arguments:
            chunksize {int} -- limit of chunk (default: {10**6})

        Returns:
            np.array -- array of each lines and columns
        """
        names = XyzFile.ColumnNames
        dtype = XyzFile.ColumnDescriptions
        return pd.read_csv(
            xyz_file.file_path, chunksize=chunksize, delimiter=";", header=None, names=names, dtype=dtype, usecols=names
        )

    def read_extent(self, xyz_file: XyzFile) -> None:
        """Read the extent from the xyz_file and write in the object.

        Arguments:
            xyz_file {XyzFile} -- Xyz file object.
        """
        text_file_reader = self.parse(xyz_file)

        for chunk in text_file_reader:
            xyz_file.line_count += chunk.shape[0]
            npChunk = chunk.to_numpy(copy=False)
            self.computeMinMax(npChunk, xyz_file)

        spatial_resolution = self.compute_spatial_resolution(npChunk)

        xyz_file.spatial_resolutionX = spatial_resolution
        xyz_file.spatial_resolutionY = spatial_resolution
        xyz_file.west = xyz_file.min(XyzConstants.COL_LON) - 0.5 * xyz_file.spatial_resolutionX
        xyz_file.east = xyz_file.max(XyzConstants.COL_LON) + 0.5 * xyz_file.spatial_resolutionX
        xyz_file.south = xyz_file.min(XyzConstants.COL_LAT) - 0.5 * xyz_file.spatial_resolutionY
        xyz_file.north = xyz_file.max(XyzConstants.COL_LAT) + 0.5 * xyz_file.spatial_resolutionY

    def compute_spatial_resolution(self, array: np.array) -> float:
        """Compute spatial resolution. Find the first diff in the 2 first columns and take
        the minimum value for the spatial_resolution.

        Arguments:
            array {np.array} -- result of panda.read_csv()

        Returns:
            float -- spatial_resolution for the futur dtm.
        """
        spatial_reso = {XyzConstants.COL_LON: 0, XyzConstants.COL_LAT: 0}

        for i, col in enumerate(spatial_reso.keys()):
            y = 0
            while array[y, i] == array[y + 1, i]:
                y += 1

            spatial_reso[col] = abs(array[y + 1, i] - array[y, i])

        return min(spatial_reso.values())
