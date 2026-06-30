#! /usr/bin/env python3
# coding: utf-8

import datetime
import os
from typing import List

from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.utils.application_utils as app_util
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.pyat_logger as log
from pyat.mbg.mbg_to_csv import export_vertical_depth_nmea
from pyat.sounder import sounder_driver_factory


class Mbg2Nmea:
    def __init__(
        self, i_paths: List[str], o_paths: List[str], overwrite: bool = False, monitor: ProgressMonitor = DefaultMonitor
    ):
        """
        Initialize a new DiffMnt process
        :param target_file:
        :param reference_file:
        :return:
        """
        self.logger = log.logging.getLogger(Mbg2Nmea.__name__)
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.overwrite = overwrite
        self.monitor = monitor

    def __call__(self):
        begin = datetime.datetime.now()
        self.monitor.set_work_remaining(len(self.i_paths))
        file_in_error = []
        for mbg_file, nmea_file in zip(self.i_paths, self.o_paths):
            sub_monitor = self.monitor.split(1)

            try:
                if os.path.exists(nmea_file):
                    if self.overwrite:
                        os.remove(nmea_file)
                    else:
                        self.logger.info(f"{nmea_file} already exists and overwrite is not set")
                        continue

                base_path = os.path.dirname(nmea_file)
                if not os.path.exists(base_path):
                    os.makedirs(base_path)
                self.logger.info(f"Starting to convert {mbg_file} to {nmea_file}")
                with sounder_driver_factory.open_sounder(mbg_file) as mbg_driver:

                    export_vertical_depth_nmea(input_mbg=mbg_driver, output_file=nmea_file)
                self.logger.info(
                    f"End of conversion for {nmea_file} :time elapsed {datetime.datetime.now() - datetime.datetime.now()}"
                )

            except Exception as e:
                file_in_error.append(mbg_file)
                self.logger.error(f"An exception was thrown : {str(e)}", exc_info=True, stack_info=True)
            finally:
                sub_monitor.done()
        self.monitor.done()
        process_util.log_result(self.logger, begin, file_in_error)


if __name__ == "__main__":
    app_util.launch_application(app_util.get_json_configuration_file(__file__), Mbg2Nmea)
