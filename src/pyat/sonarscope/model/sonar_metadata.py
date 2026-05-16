""" contains """
from pyat.sonarscope.model.constants import SONAR_GROUP_NAME
from pyat.sonarscope.common.xarray_utils import get_nc_attribute
from pyat.xsf.xsf_driver import XsfDriver


class SonarFileMetaData:
    def __init__(self, xsf_dataset: XsfDriver):
        self.nc_dataset = xsf_dataset  # the netcdf dataset
        self.metadata = {}

    def read(self) -> dict:
        """read sonar attributes and return a dictionary of values read"""
        # We assume that the dataset was opened previously
        sonar_group = self.nc_dataset[SONAR_GROUP_NAME]
        desc = {}
        self.metadata = get_nc_attribute(nc_variable_or_group=sonar_group)
        return desc
