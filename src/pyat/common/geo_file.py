from abc import ABC

import numpy
from osgeo import osr

osr.UseExceptions()

SR_WGS_84 = osr.SpatialReference()
SR_WGS_84.ImportFromEPSG(4326)
SR_PSEUDO_MERCATOR = osr.SpatialReference()
SR_PSEUDO_MERCATOR.ImportFromEPSG(3857)


class GeoFile(ABC):
    """
    File's properties.
        - filePath : path to the emo file
        - extent : array holding the bounding box (west , east, south, north)
    """

    def __init__(self, file_path: str):
        self._file_path = file_path
        self.west = numpy.nan
        self.east = numpy.nan
        self.south = numpy.nan
        self.north = numpy.nan
        self.spatial_reference = SR_WGS_84

    @property
    def file_path(self):
        return self._file_path

    @property
    def west(self):
        return self._west

    @west.setter
    def west(self, west: float):
        self._west = west

    @property
    def east(self):
        return self._east

    @east.setter
    def east(self, east: float):
        self._east = east

    @property
    def south(self):
        return self._south

    @south.setter
    def south(self, south: float):
        self._south = south

    @property
    def north(self):
        return self._north

    @north.setter
    def north(self, north: float):
        self._north = north

    @property
    def spatial_reference(self):
        return self._spatial_reference

    @spatial_reference.setter
    def spatial_reference(self, spatial_reference: osr.SpatialReference):
        self._spatial_reference = spatial_reference
