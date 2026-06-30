#! /usr/bin/env python3
# coding: utf-8

import datetime
import os
from typing import List

from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.utils.cut_file_utils import create_cut_file_from_ncfile_set


class XsfToCut:

    def __init__(
        self,
        i_paths: List[str],
        o_path: str,
        overwrite: bool = False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):

        self.logger = log.logging.getLogger(self.__class__.__name__)
        self.monitor = monitor

        # Parsing parameters
        self.i_paths = arg_util.parse_list_of_files("i_paths", i_paths, True)
        self.o_path = o_path
        self.overwrite = overwrite

    def __call__(self) -> None:
        """Run method"""
        self.monitor.set_work_remaining(len(self.i_paths))
        begin = datetime.datetime.now()
        file_in_error = []

        try:
            self.logger.info(f"Starting to create cut file from {self.i_paths}")
            self.logger.info(f"\tto {self.o_path}")

            if not self.overwrite and os.path.exists(self.o_path):
                self.logger.warning(f"File {self.o_path} already exists and overwrite is not allowed.")
            else:
                create_cut_file_from_ncfile_set(self.i_paths, self.o_path)

        except Exception as e:
            self.logger.error(f"An exception was thrown : {str(e)}", exc_info=True, stack_info=True)

        self.monitor.done()
        process_util.log_result(self.logger, begin, file_in_error)
