from typing import Any, Dict, Set

import numpy as np
from sonar_netcdf.sonar_groups import BeamGroup1Grp as b

from pyat.sonarscope.cruise_summary.file_data import FileDataStore


class VariableMetadata:
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value


class GlobalMetadata:
    """Container for metadata associated to a file"""

    def __init__(self):
        self.ping_count = 0
        self.file_count = 0
        # storage for metadata for all variables
        self.variable_metadata: Dict[str, VariableMetadata] = {}
        self.attributes: Dict[str, Set[Any]] = {}  # a dictionnary of set containing the merge list of global attributes

    def add_file(self, ping_count, attributes: Dict[str, Any]):
        self.file_count += 1
        self.ping_count += ping_count
        for name, value in attributes.items():
            if name not in self.attributes:
                self.attributes[name] = set()
            self.attributes[name].add(value)

    def add_data(self, name, min_value, max_value):
        if name not in self.variable_metadata:
            self.variable_metadata[name] = VariableMetadata(min_value, max_value)
        else:
            variable_meta = self.variable_metadata[name]
            variable_meta.min_value = np.nanmin([min_value, variable_meta.min_value])
            variable_meta.max_value = np.nanmax([max_value, variable_meta.max_value])


class GlobalDataModel:
    def __init__(self, file_data: Dict[str, FileDataStore]):
        self.file_data = file_data
        self.metadata = GlobalMetadata()
        self.compute_global_metadata()
        self.decimate()

    def compute_global_metadata(self):
        """Compute global attributes"""

        for f, v in self.file_data.items():

            self.metadata.add_file(ping_count=v.statistics.ping_count, attributes=v.global_attributes.metadata)

            for variable_name, value_minmax in v.statistics.minmax_values.items():
                self.metadata.add_data(
                    name=variable_name, min_value=value_minmax.min_value, max_value=value_minmax.max_value
                )

    def decimate(self):
        """Decimate datasets to be able to have loadable plots in browsers"""

        total_ping_count = self.metadata.ping_count
        if total_ping_count == 0:
            return

        reduction_factor = 2**15 / total_ping_count  # 2**15 = 33768 points
        reduction_factor = max(1, reduction_factor)  # only reduce dataset if there is too many pings

        for f, v in self.file_data.items():
            v.decimate(reduction_factor=reduction_factor)

    def get_min_date(self):
        return self.metadata.variable_metadata[b.PING_TIME_VNAME].min_value

    def get_max_date(self):
        return self.metadata.variable_metadata[b.PING_TIME_VNAME].max_value

    def get_min_latitude(self):
        return self.metadata.variable_metadata[b.PLATFORM_LATITUDE_VNAME].min_value

    def get_max_latitude(self):
        return self.metadata.variable_metadata[b.PLATFORM_LATITUDE_VNAME].max_value

    def get_min_longitude(self):
        return self.metadata.variable_metadata[b.PLATFORM_LONGITUDE_VNAME].min_value

    def get_max_longitude(self):
        return self.metadata.variable_metadata[b.PLATFORM_LONGITUDE_VNAME].max_value
