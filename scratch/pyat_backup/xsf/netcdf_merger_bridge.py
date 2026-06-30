#! /usr/bin/env python3
# coding: utf-8

import datetime as dt
import os
from typing import Dict, List, Optional

from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from sonar_netcdf.utils import nc_merger as nc_m

import pyat.utils.argument_utils as arg_util
import pyat.utils.cut_file_utils as cut_util
import pyat.utils.pyat_logger as log


class NcMergerBridge:
    """
    Callable used by pyat/app to launch a Cut/Merge process on Netcdf files.
    This is a bridge to sonar_netcdf generic merger class (NCMerger).
    This class aim to adapt arguments comming from Globe and pass them to NCMerger.

    In specific cases, a sub-type of NCMerger can be provided with nc_merger_class argument.
    For XSF or any other sonar file it is recommended to specify SNMerger of sonar_netcdf instead.
    """

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
        nc_merger_class=nc_m.NCMerger,
    ):
        """
        Constructor
        """
        self.logger = log.logging.getLogger(self.__class__.__name__)
        self.monitor = monitor

        # Parsing parameters
        i_paths = arg_util.parse_list_of_files("i_paths", i_paths)
        o_paths = arg_util.parse_list_of_files("o_paths", o_paths, False)
        start_date = arg_util.parse_datetime(start_date)
        end_date = arg_util.parse_datetime(end_date)

        # Computing timelines
        computed_timelines: List[nc_m.Timeline] = []
        if cut_file is not None:
            self.logger.info("Using cut_file argument to determine the cutting time intervals")
            computed_timelines = cut_util.parse_cut_file(cut_file, self.logger)
            if len(computed_timelines) == 0:
                self.logger.error(f"No cut line found in cut file. Merge abort")
                return
            # checking cut file timelines and output file path coherence
            if len(computed_timelines) != len(o_paths):
                self.logger.error(
                    f"Cut line number ({len(computed_timelines)}) in cut file differs from "
                    f"number of output file paths ({len(o_paths)}). "
                    f"Possible cause is a malformated .cut file, check it out.  Merge abort"
                )
                return

        elif geo_mask_file is not None:
            self.logger.info(f"Compute cut lines from geographic mask (reverse = {reverse_geo_mask}).")
            computed_timelines = cut_util.create_cut_lines_from_files(
                i_paths=i_paths, o_paths=o_paths, i_geo_mask_path=geo_mask_file, reverse_geo_mask=reverse_geo_mask
            )
            if len(computed_timelines) == 0:
                self.logger.warning("Geographic mask does not cut input files.")
                return

            # Build output filenames from computed timelines.
            o_dir = os.path.dirname(o_paths[0])
            o_paths = [os.path.join(o_dir, timeline.name) for timeline in computed_timelines]

        elif start_date is not None and end_date is not None:
            self.logger.info(f"Apply custom time interval : from {start_date} to {end_date}.")
            computed_timelines.append(nc_m.Timeline("Single", start_date, end_date))
        elif timelines:
            self.logger.info("Using timelines argument to determine the cutting time intervals")
            computed_timelines = timelines
        else:
            self.logger.info("No cutting time interval specified. Merging files...")

        if len(computed_timelines) > 1:
            self.logger.info(f"Apply {len(computed_timelines)} cut lines : ")
            for timeline in computed_timelines:
                self.logger.info(f"{timeline.name} :  from {timeline.start} to {timeline.stop}")

        self.merger = nc_merger_class(i_paths, o_paths, computed_timelines, overwrite)

    def __call__(self) -> Dict:
        """Runs cut/merge."""
        resulting_files = self.merger.merge() if hasattr(self, "merger") else []
        return {"outfile": [str(file_path) for file_path in resulting_files]}
