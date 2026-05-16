#! /usr/bin/env python3
# coding: utf-8


from typing import NamedTuple

import numpy as np
import pandas as pd

# Dataframe current columns
TIME_INDEX = "time_index"
RANGE = "range"
RANGE_INDEX = "range_index"
EASTWARD_VELOCITY = "eastward_velocity"
NORTHWARD_VELOCITY = "northward_velocity"
DOWNWARD_VELOCITY = "downward_velocity"


class AdcpData(NamedTuple):
    """
    Class representing all data of an ADCP file
    """

    # Path to the file
    file_path: str
    # Navigation data
    time: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    # Current data
    current_data: pd.DataFrame

    def apply_range_filter(self, range_min: float, range_max: float) -> "AdcpData":
        """
        Apply a range filter to the AdcpData object.
        """
        filtered_data = self.current_data
        if range_min is not None:
            filtered_data = filtered_data[filtered_data[RANGE] >= range_min]
        if range_max is not None:
            filtered_data = filtered_data[filtered_data[RANGE] <= range_max]
        return self._with_current(filtered_data)

    def apply_range_sampling(self, range_sampling: int) -> "AdcpData":
        """
        Apply a range sampling to the AdcpData object.
        Keep only "range_sampling" elements of each time index.
        """
        if range_sampling is None or range_sampling <= 0:
            return self

        filtered_data = self.current_data[self.current_data[RANGE_INDEX] % range_sampling == 0]

        return self._with_current(filtered_data)

    def apply_time_filter(self, time_index_min: float, time_index_max: float) -> "AdcpData":
        """
        Apply a time filter to the AdcpData object.
        """
        filtered_data = self.current_data
        if time_index_min is not None:
            filtered_data = filtered_data[filtered_data[TIME_INDEX] >= time_index_min]
        if time_index_max is not None:
            filtered_data = filtered_data[filtered_data[TIME_INDEX] <= time_index_max]
        return self._with_current(filtered_data)

    def apply_time_sampling(self, time_sampling: int) -> "AdcpData":
        """
        Apply a time sampling to the AdcpData object.
        """
        if time_sampling is None or time_sampling <= 0:
            return self

        filtered_data = self.current_data[self.current_data[TIME_INDEX] % time_sampling == 0]

        return self._with_current(filtered_data)

    def reduce(self, max_count: int) -> "AdcpData":
        """
        Apply a limit to the AdcpData object.
        Sample the current data based on the max count
        """
        if self.current_data.index.size <= max_count or max_count <= 0:
            return self

        filtered_data = self.current_data.sample(n=max_count, random_state=1)
        return self._with_current(filtered_data)

    def _with_current(self, new_current_data: pd.DataFrame) -> "AdcpData":
        """
        Create a new AdcpData object with the current data replaced.
        """
        return AdcpData(
            file_path=self.file_path,
            time=self.time,
            latitude=self.latitude,
            longitude=self.longitude,
            current_data=new_current_data,
        )
