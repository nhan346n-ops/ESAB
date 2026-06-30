from abc import abstractmethod
from typing import runtime_checkable, Protocol, Optional

import numpy as np


@runtime_checkable
class AbstractNavigation(Protocol):
    """
    Interface for navigation data.
    """

    @abstractmethod
    def get_name(self) -> Optional[str]: ...

    @abstractmethod
    def get_times(self) -> np.ndarray: ...

    @abstractmethod
    def get_latitudes(self) -> np.ndarray: ...

    @abstractmethod
    def get_longitudes(self) -> np.ndarray: ...

    # Optional methods

    def get_headings(self) -> Optional[np.ndarray]:
        return None

    def get_altitudes(self) -> Optional[np.ndarray]:
        """
        Returns height above reference ellipsoid in meters.
        """
        return None

    def get_vertical_offsets(self) -> Optional[np.ndarray]:
        """
        Returns height above sea surface in meters.
        """
        return None

    def get_speeds(self) -> Optional[np.ndarray]:
        return None

    def get_courses_over_ground(self) -> Optional[np.ndarray]:
        return None

    def get_sensor_quality_indicators(self) -> Optional[np.ndarray]:
        """
        Returns Sensor Quality Indicators, as defined as the GPS positionning type of GPGGA records (NMEA 0183)
        """
        return None
