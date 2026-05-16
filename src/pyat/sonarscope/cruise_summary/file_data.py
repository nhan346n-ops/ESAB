from typing import Dict

import numpy as np
import sonar_netcdf.sonar_groups as sg

from pyat.sonarscope.model.signal.ping_detection_signal import PingDetectionSignal
from pyat.sonarscope.model.signal.ping_signal import PingSignal
from pyat.sonarscope.model.sonar_metadata import SonarFileMetaData
from pyat.sonarscope.model.sound_velocity_profile import SVPProfiles
from pyat.utils.netcdf import get_default_fillvalue


class VariableMetadata:
    """Simple class containing min and max values associated with a variable"""

    def __init__(self, min_value, max_value, fill_value=np.nan):
        self.min_value = min_value
        self.max_value = max_value

        if not np.isnan(self.max_value) and self.max_value == fill_value:
            self.max_value = np.nan
        if not np.isnan(self.min_value) and self.min_value == fill_value:
            self.min_value = np.nan


class FileMetadata:
    """Container for metadata associated to a file"""

    def __init__(self):
        self.ping_count = 0
        # storage for metadata for all variables
        self.minmax_values: Dict[VariableMetadata] = {}


class FileDataStore:
    """A file container which store per file data used by cruise summary"""

    def __init__(
        self,
        file: str,
        ping_timed_dataset: PingSignal,
        global_attributes: SonarFileMetaData,
        sound_speed_profile: SVPProfiles,
        ping_detection_dataset: PingDetectionSignal,
    ):
        self.file = file
        self.ping_time_dataset = ping_timed_dataset
        self.svp_dataset = sound_speed_profile
        self.statistics = self._compute_stats()
        self.global_attributes = global_attributes
        self.ping_detection_dataset = ping_detection_dataset
        self.decimated_dataset = None
        self.storage = {}  # dictionary storage for ancillary data values

    def _compute_stats(self):
        """compute some global information about dataset (number of ping, date/time and so on...)"""
        meta = FileMetadata()
        meta.ping_count = len(self.ping_time_dataset.variables[sg.BeamGroup1Grp.PING_TIME_VNAME])

        # compute metadata for all variables, we just compute the min and max values
        for v in self.ping_time_dataset.variables:
            values = self.ping_time_dataset.variables[v].values
            fill_value = get_default_fillvalue(values.dtype, raise_exception=False)
            if fill_value is None:  # no fill value is defined
                fill_value = np.nan  # always return False when using == test
            if values.size > 0:
                meta.minmax_values[v] = VariableMetadata(
                    np.nanmin(self.ping_time_dataset.variables[v].values),
                    np.nanmax(self.ping_time_dataset.variables[v].values),
                    fill_value=fill_value,
                )
            else:
                meta.minmax_values[v] = VariableMetadata(np.nan, np.nan)
        return meta

    def decimate(self, reduction_factor=1):
        """decimate dataset in order to render without to many point the cruise trajectory"""

        expected_ping_count = reduction_factor * self.statistics.ping_count
        if expected_ping_count >= self.statistics.ping_count:
            # we do nothing
            self.decimated_dataset = self.ping_time_dataset.xr_dataset
        else:
            new_ping_count = int(self.statistics.ping_count * reduction_factor)
            self.decimated_dataset = self.ping_time_dataset.xr_dataset.coarsen(
                {"ping_time": new_ping_count}, boundary="trim"
            ).mean()
