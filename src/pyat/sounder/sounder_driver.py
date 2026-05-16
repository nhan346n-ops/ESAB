#! /usr/bin/env python3
# coding: utf-8

import logging
from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import numpy as np
from osgeo import osr

from pyat.common.geo_file import SR_WGS_84, GeoFile


class SounderFile(GeoFile):
    """
    Sounder file's properties
    """

    @property
    def swath_count(self):
        return self._swath_count

    @swath_count.setter
    def swath_count(self, swath_count: float):
        self._swath_count = swath_count

    @property
    def beam_count(self):
        return self._beam_count

    @beam_count.setter
    def beam_count(self, beam_count: float):
        self._beam_count = beam_count

    @property
    def antenna_count(self):
        return self._antenna_count

    @antenna_count.setter
    def antenna_count(self, antenna_count: float):
        self._antenna_count = antenna_count

    def __init__(self, filePath: str, spatial_reference: osr.SpatialReference = SR_WGS_84):
        super().__init__(filePath)
        self.spatial_reference = spatial_reference
        self.west = self.east = self.south = self.north = np.nan


class SounderDriver(ABC):
    @property
    def sounder_file(self) -> SounderFile:
        return self._sounder_file

    def __init__(self, file_path: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Managing {file_path} with a {self.__class__.__name__}")
        self._sounder_file = SounderFile(file_path)

    @abstractmethod
    def open(self, mode: str = "r") -> None:
        """
        Open the file and return the resulting Dataset
        """

    @abstractmethod
    def close(self) -> None:
        """Close the dataset if opened"""

    def get_file_path(self) -> str:
        return self.sounder_file.file_path

    @abstractmethod
    def read_validity_flags(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of validity flags
        """

    @abstractmethod
    def read_fcs_depths(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of depths. Shape is (to_swath - from_swath, beam_count)
        Depths are projected in Coordinates system transformations FCS (Fixed Coordinate System)
        """

    @abstractmethod
    def read_scs_depths(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of depths. Shape is (to_swath - from_swath, beam_count)
        Depths are projected in Coordinates system transformations SCS (Surface Coordinate System)
        """

    @abstractmethod
    def read_reflectivities(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of Reflectivity values of all antennas
        """

    @abstractmethod
    def read_across_distances(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of across distance. Shape is (to_swath - from_swath, beam_count)
        """

    @abstractmethod
    def read_across_angles(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of across angles. Shape is (to_swath - from_swath, beam_count)
        """

    @abstractmethod
    def read_platform_longitudes(self) -> np.ndarray:
        """
        return the numpy array of platform longitudes. Shape is (swath_count)
        """

    @abstractmethod
    def read_platform_latitudes(self) -> np.ndarray:
        """
        return the numpy array of platform latitudes. Shape is (swath_count)
        """

    @abstractmethod
    def read_platform_headings(self) -> np.ndarray:
        """
        return the numpy array of platform headings. Shape is (swath_count)
        """

    @abstractmethod
    def read_ping_times(self) -> np.ndarray:
        """
        return the numpy array (DateTime64) of ping times. Shape is (swath_count)
        """

    @abstractmethod
    def read_platform_vertical_offsets(self) -> np.ndarray:
        """
        return the numpy array of computed read_platform vertical offsets. Shape is (swath_count)
        """

    @abstractmethod
    def iter_beam_positions(
        self, swath_count_by_iter: int, first_swath: int = 0
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        return an Iterable of the numpy arrays of beam's longitude and latitude
        """

    @abstractmethod
    def read_detection_longitude(self) -> np.ndarray | None:
        """
        return the numpy array of longitude of the detection.
        """

    @abstractmethod
    def read_detection_latitude(self) -> np.ndarray | None:
        """
        return the numpy array of latitude of the detection.
        """

    @abstractmethod
    def read_detection_quality_factor(self) -> np.ndarray | None:
        """
        return the numpy array of the estimated standard deviation as % of the detected depth.
        """

    @abstractmethod
    def read_detection_tx_beam(self) -> np.ndarray | None:
        """
        return the numpy array of the detection transmit beam index.
        """

    @abstractmethod
    def read_detection_type(self) -> np.ndarray | None:
        """
        return the numpy array of the type of detection.
        """

    @abstractmethod
    def read_multiping_sequence(self) -> np.ndarray | None:
        """
        return the numpy array of the multiping sequence identifier.
        """

    @abstractmethod
    def read_multiping_center_frequency(self) -> np.ndarray | None:
        """
        return the numpy array of the center frequency in transmitted pulse.
        """

    @abstractmethod
    def read_detection_ping_frequency(self) -> np.ndarray | None:
        """
        return the numpy array of the detection ping frequencies.
        """
