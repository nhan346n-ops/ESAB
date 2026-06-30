"""Manage SVP profile models"""

import sonar_netcdf.sonar_groups as sg
import xarray as xr


class SVPProfiles:
    def __init__(self, file_path: str):
        self.file_path = file_path  # the source file
        self.xr_dataset = None

    def read_svp(self) -> xr.Dataset:
        """Read per entirely the sound speed profiles in files and store values in a xarray"""
        with xr.open_dataset(
            self.file_path,
            group=sg.SoundSpeedProfileGrp.get_group_path(),
            decode_times=True,
        ) as ds:
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            self.xr_dataset = ds
        return ds
