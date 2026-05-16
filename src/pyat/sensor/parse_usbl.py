#! /usr/bin/env python3
# coding: utf-8

import datetime
import os
from typing import List
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.utils.process_utils as process_util
import pyat.utils.pyat_logger as log


class UsblParser:
    """
    Utility class to filter and parse USBL log files to CSV readable by GLOBE.
    """

    def __init__(
        self, i_paths: List[str], o_paths: List[str], overwrite: bool = False, monitor: ProgressMonitor = DefaultMonitor
    ):
        self.logger = log.logging.getLogger(UsblParser.__name__)
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.overwrite = overwrite
        self.monitor = monitor

    def __call__(self):
        begin = datetime.datetime.now()
        self.monitor.set_work_remaining(len(self.i_paths))
        file_in_error = []
        for i_path, o_path in zip(self.i_paths, self.o_paths):
            sub_monitor = self.monitor.split(1)
            try:
                if os.path.exists(o_path):
                    if self.overwrite:
                        os.remove(o_path)
                    else:
                        self.logger.info(f"{o_path} already exists and overwrite is not set")
                        continue
                base_path = os.path.dirname(o_path)
                if not os.path.exists(base_path):
                    os.makedirs(base_path)
                self.logger.info(f"Starting to convert '{os.path.basename(i_path)}' to '{os.path.basename(o_path)}'...")

                # Read and parse PTSAG frames.
                with open(i_path, "r", encoding="utf-8") as i_file:
                    lines = [line.replace(",", ";") for line in i_file.readlines() if line.startswith("$PTSAG")]
                self.logger.info(f"$PTAG frames : {len(lines)}")

                # Write output file.
                with open(o_path, "w", encoding="utf-8") as o_file:
                    self.logger.info(f"Save in '{os.path.basename(o_path)}'...")
                    o_file.writelines(lines)

            except Exception as e:
                file_in_error.append(i_path)
                self.logger.error(f"An exception was thrown : {str(e)}", exc_info=True, stack_info=True)
            finally:
                sub_monitor.done()
        self.monitor.done()
        process_util.log_result(self.logger, begin, file_in_error)
