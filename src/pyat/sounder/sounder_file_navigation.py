import os
from os import PathLike
from typing import Optional

import numpy as np

from pyat.navigation.abstract_navigation import AbstractNavigation
from pyat.sounder import sounder_driver_factory


class SounderFileNavigation(AbstractNavigation):
    """
    Implementation of AbstractNavigation for sounder files (.mbg, .xsf.nc).
    """

    def __init__(self, file_path: PathLike | str):
        self.file_path = file_path
        self.sounder_driver = sounder_driver_factory.get_sounder_driver(self.file_path)
        self.sounder_driver.open()

    def get_name(self) -> Optional[str]:
        return os.path.basename(self.file_path)

    def close(self):
        self.sounder_driver.close()

    def get_times(self) -> np.ndarray:
        return self.sounder_driver.read_ping_times()

    def get_latitudes(self) -> np.ndarray:
        return self.sounder_driver.read_platform_latitudes()

    def get_longitudes(self) -> np.ndarray:
        return self.sounder_driver.read_platform_longitudes()

    def get_headings(self) -> Optional[np.ndarray]:
        return self.sounder_driver.read_platform_headings()

    def get_altitudes(self) -> Optional[np.ndarray]:
        return None

    def get_vertical_offsets(self) -> Optional[np.ndarray]:
        return self.sounder_driver.read_platform_vertical_offsets()

    def get_speeds(self) -> Optional[np.ndarray]:
        return None

    def get_courses_over_ground(self) -> Optional[np.ndarray]:
        return None
