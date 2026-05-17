#! /usr/bin/env python3
# coding: utf-8

import datetime
from typing import Dict, List

import numpy as np
import pygws.service.execution_context as exec_ctx
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.utils.process_utils as process_util
import pyat.utils.pyat_logger as log
from pyat.sounder import sounder_driver_factory


class PolarEchogramsArgumentsEvaluator:
    """
    Estimator for sample_resolution and height from input files stats
    """

    def __init__(
        self,
        i_paths: List[str],
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor.
        """
        self.i_paths = i_paths
        self.monitor = monitor
        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __call__(self) -> Dict | None:
        """Run method."""
        begin = datetime.datetime.now()
        file_in_error: List[str] = []

        self.logger.info("Evalutating...")

        sample_resolution, height = self.evaluate()

        process_util.log_result(self.logger, begin, file_in_error)
        return self._report_result(sample_resolution, height)

    def evaluate(self):
        """
        return sample_resolution and height based on max sample_interval and sound_speed
        """
        _max_depth = float("-inf")
        _max_sample_interval = float("-inf")
        _max_sound_speed = float("-inf")

        for i_path in self.i_paths:
            with sounder_driver_factory.open_sounder(i_path) as xsf_driver:
                # DEPTH
                detection_z = xsf_driver.read_vertical_distances(0, xsf_driver.sounder_file.swath_count)
                valid = xsf_driver.read_validity_flags(0, xsf_driver.sounder_file.swath_count)
                detection_z[valid is False] = np.nan
                # max_vertical_distance
                max_depth = np.nanmax(detection_z)
                if max_depth > _max_depth:
                    _max_depth = max_depth
                # sample_interval
                sample_interval = xsf_driver["Sonar"]["Beam_group1"]["sample_interval"][:]
                min_sample_interval = np.nanmin(sample_interval)
                if _max_sample_interval == float("-inf") or min_sample_interval < _max_sample_interval:
                    _max_sample_interval = min_sample_interval
                # sound_speed
                sound_speed = xsf_driver["Sonar"]["Beam_group1"]["sound_speed_at_transducer"][:]
                max_sound_speed = np.nanmax(sound_speed)
                if max_sound_speed > _max_sound_speed:
                    _max_sound_speed = max_sound_speed

        # deltas
        sample_res = _max_sample_interval * _max_sound_speed / 2
        round_sample_res = round(sample_res, 2)
        height = round(max_depth / sample_res)

        self.logger.info(f"sample resolution: {round_sample_res} height: {height}")
        return float(sample_res), float(height)

    def _report_result(self, sample_resolution: float, height: float) -> Dict | None:
        """
        Prepare the result for the report JSON file
        """
        result = {
            "sample_resolution": sample_resolution,
            "height": height,
        }

        # Using rsocket (if present) to send the result
        rsocket_msg_emitter = exec_ctx.get_rsocket_msg_emitter()
        if rsocket_msg_emitter is not None:
            rsocket_msg_emitter.emit_map_of_double(result)
            return None

        return {"result": result}
