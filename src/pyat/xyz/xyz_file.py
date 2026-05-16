#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import pyat.common.geo_file as gf
import pyat.xyz.xyz_constants as XyzConstants


class XyzFile(gf.GeoFile):
    """
    XYZ file's properties.
    """

    ColumnDescriptions = {XyzConstants.COL_LON: float, XyzConstants.COL_LAT: float, XyzConstants.COL_DEPTH: float}
    # Column's order in a emo file
    ColumnNames = list(ColumnDescriptions.keys())

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.minmax = {key: [np.nan, np.nan] for (key, value) in XyzFile.ColumnDescriptions.items()}
        self.line_count = 0
        self.spatial_resolutionX = np.nan
        self.spatial_resolutionY = np.nan

    def min(self, columnName):
        return self.minmax[columnName][0]

    def max(self, columnName):
        return self.minmax[columnName][1]
