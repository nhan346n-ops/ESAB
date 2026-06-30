from typing import Optional

import numpy as np

from pyat.navigation.abstract_navigation import AbstractNavigation


def copy_from(other_nav: AbstractNavigation):
    return NavigationData(
        name=other_nav.get_name(),
        times=other_nav.get_times(),
        latitudes=other_nav.get_latitudes(),
        longitudes=other_nav.get_longitudes(),
        headings=other_nav.get_headings(),
        altitudes=other_nav.get_altitudes(),
        vertical_offsets=other_nav.get_vertical_offsets(),
        speeds=other_nav.get_speeds(),
        courses_over_ground=other_nav.get_courses_over_ground(),
        sensor_quality_indicators=other_nav.get_sensor_quality_indicators(),
    )


class NavigationData(AbstractNavigation):
    """
    Implementation of AbstractNavigation based on numpy arrays.
    """

    def __init__(
        self,
        times: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        name: Optional[str] = None,
        headings: Optional[np.ndarray] = None,
        altitudes: Optional[np.ndarray] = None,
        vertical_offsets: Optional[np.ndarray] = None,
        speeds: Optional[np.ndarray] = None,
        courses_over_ground: Optional[np.ndarray] = None,
        sensor_quality_indicators: Optional[np.ndarray] = None,
    ):
        self.times = times
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.name = name
        self.headings = headings
        self.altitudes = altitudes
        self.vertical_offsets = vertical_offsets
        self.speeds = speeds
        self.courses_over_ground = courses_over_ground
        self.sensor_quality_indicators = sensor_quality_indicators

    @classmethod
    def copy_from(cls, other_nav: AbstractNavigation):
        """
        Creates a new NavigationData from other AbstractNavigation (useful to store data in arrays).
        """
        return cls(
            name=other_nav.get_name(),
            times=other_nav.get_times(),
            latitudes=other_nav.get_latitudes(),
            longitudes=other_nav.get_longitudes(),
            headings=other_nav.get_headings(),
            altitudes=other_nav.get_altitudes(),
            vertical_offsets=other_nav.get_vertical_offsets(),
            speeds=other_nav.get_speeds(),
            courses_over_ground=other_nav.get_courses_over_ground(),
            sensor_quality_indicators=other_nav.get_sensor_quality_indicators(),
        )

    def get_name(self) -> Optional[str]:
        return self.name

    def get_times(self) -> np.ndarray:
        return self.times

    def get_latitudes(self) -> np.ndarray:
        return self.latitudes

    def get_longitudes(self) -> np.ndarray:
        return self.longitudes

    def get_headings(self) -> Optional[np.ndarray]:
        return self.headings

    def get_altitudes(self) -> Optional[np.ndarray]:
        return self.altitudes

    def get_vertical_offsets(self) -> Optional[np.ndarray]:
        return self.vertical_offsets

    def get_speeds(self) -> Optional[np.ndarray]:
        return self.speeds

    def get_courses_over_ground(self) -> Optional[np.ndarray]:
        return self.courses_over_ground

    def get_sensor_quality_indicators(self) -> Optional[np.ndarray]:
        return self.sensor_quality_indicators
