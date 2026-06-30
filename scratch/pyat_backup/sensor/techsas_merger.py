import datetime as dt
import os
from typing import Dict, List, Optional

import numpy as np
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from pytechsas.sensor.sensor_constant import DATE_CREATED
from pytechsas.sensor.techsas_constant import (
    CREATION_TIME,
    FIRST_FRAME_DATE,
    HISTORY,
    LAST_FRAME_DATE,
)
from pytechsas.sensor.techsas_file import add_history, open_nc_file, read_times
from sonar_netcdf.utils import nc_merger as nc_m

from pyat.xsf.netcdf_merger_bridge import NcMergerBridge


class TechsasMerger(NcMergerBridge):

    def __init__(
        self,
        i_paths: List[str],
        o_paths: List[str],
        cut_file: Optional[str] = None,
        geo_mask_file: Optional[str] = None,
        reverse_geo_mask: bool = False,
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None,
        timelines: Optional[List[nc_m.Timeline]] = None,
        overwrite: bool = False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        super().__init__(
            i_paths,
            o_paths,
            cut_file=cut_file,
            geo_mask_file=geo_mask_file,
            reverse_geo_mask=reverse_geo_mask,
            start_date=start_date,
            end_date=end_date,
            timelines=timelines,
            overwrite=overwrite,
            monitor=monitor,
        )

    def __call__(self) -> Dict:
        self.check_before()
        result = super().__call__()
        self.post_process(result["outfile"])
        return result

    def check_before(self) -> None:
        """
        Checks if input parameters are valid before processing.
        """
        # Check extensions
        extensions = [os.path.splitext(i_path)[1] for i_path in self.merger.i_paths]
        if len(set(extensions)) != 1:
            raise ValueError(f"Input files have different extensions: {set(extensions)}")

        # Check global attributes
        exclude_attrs = [HISTORY, FIRST_FRAME_DATE, LAST_FRAME_DATE, CREATION_TIME, DATE_CREATED]
        ref_global_attributes = None
        for i_path in self.merger.i_paths:
            file_name = os.path.basename(i_path)
            with open_nc_file(i_path) as dataset:
                if ref_global_attributes is None:
                    ref_global_attributes = {
                        attr: dataset.getncattr(attr) for attr in dataset.ncattrs() if attr not in exclude_attrs
                    }
                else:
                    other_global_attrs = {
                        attr: dataset.getncattr(attr) for attr in dataset.ncattrs() if attr not in exclude_attrs
                    }
                    # Union of keys from both dictionaries
                    for key in set(ref_global_attributes) | set(other_global_attrs):
                        if key not in other_global_attrs:
                            self.logger.warning(f"Attribute '{key}' is missing in file {file_name}")
                        elif key not in ref_global_attributes:
                            self.logger.warning(f"Attribute '{key}' is in file {file_name} but not in previous files.")
                        elif ref_global_attributes[key] != other_global_attrs[key]:
                            self.logger.warning(
                                f"Attribute '{key}' has different values: {ref_global_attributes[key]} != {other_global_attrs[key]}"
                            )

    def post_process(self, resulting_files: Dict):
        """
        Post-processing : recompute global attributes.
        """
        for file_path in resulting_files:
            self.logger.info(f"Post process file : {os.path.basename(file_path)}...")

            with open_nc_file(file_path, mode="r+") as dataset:
                # Update history
                add_history(
                    new_ds=dataset,
                    module_name=self.__class__.__name__,
                    history_info="Merged with techsas_merger (PyAT).",
                )
                # self.logger.info(f"{HISTORY} : {dataset.getncattr(HISTORY)}")
