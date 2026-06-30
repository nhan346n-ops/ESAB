#! /usr/bin/env python3
# coding: utf-8

import datetime
from typing import Dict, List

import numpy as np
import pygws.service.execution_context as exec_ctx
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.utils.process_utils as process_util
import pyat.utils.pyat_logger as log
from pyat.function.evaluate_sounder_spatial_resolution import (
    SpatialResolutionEvaluator,
)
from pyat.sounder import sounder_driver_factory


class LongitudinalSectionArgumentsEvaluator:
    """
    Estimator for delta_elevation, delta_across and delta_along from input files stats
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

        delta_elevation, delta_across, delta_along = self._evaluate()

        process_util.log_result(self.logger, begin, file_in_error)
        return self._report_result(delta_elevation, delta_across, delta_along)

    def _evaluate(self):
        """ """
        _min_across = float("inf")
        _max_across = float("-inf")
        _min_depth = float("inf")
        _max_depth = float("-inf")
        _max_sample_interval = float("-inf")
        _max_sound_speed = float("-inf")

        for i_path in self.i_paths:
            with sounder_driver_factory.open_sounder(i_path) as xsf_driver:
                # ACROSS
                across = xsf_driver.read_across_distances(0, xsf_driver.sounder_file.swath_count)
                valid = xsf_driver.read_validity_flags(0, xsf_driver.sounder_file.swath_count)
                across[valid is False] = np.nan
                # min_across
                min_across = np.nanmin(across)
                if min_across < _min_across:
                    _min_across = min_across
                # max_across
                max_across = np.nanmax(across)
                if max_across > _max_across:
                    _max_across = max_across
                # DEPTH
                detection_z = xsf_driver.read_vertical_distances(0, xsf_driver.sounder_file.swath_count)
                # max_vertical_distance
                max_depth = np.nanmax(detection_z)
                if max_depth > _max_depth:
                    _max_depth = max_depth
                # sample_interval
                sample_interval = xsf_driver["Sonar"]["Beam_group1"]["sample_interval"][:]
                max_sample_interval = np.nanmax(sample_interval)
                if max_sample_interval > _max_sample_interval:
                    _max_sample_interval = max_sample_interval
                # sound_speed
                sound_speed = xsf_driver["Sonar"]["Beam_group1"]["sound_speed_at_transducer"][:]
                max_sound_speed = np.nanmax(sound_speed)
                if max_sound_speed > _max_sound_speed:
                    _max_sound_speed = max_sound_speed

        # deltas
        delta_elevation = round(_max_sample_interval * _max_sound_speed / 2, 2)

        # experimental value based on Carla feedback
        delta_across = round(np.abs(_max_depth) / 100, 2)

        spatial_evaluator = SpatialResolutionEvaluator(self.i_paths)
        delta_along, _ = spatial_evaluator.evaluate()

        self.logger.info(f"delta elevation: {delta_elevation}\ndelta across: {delta_across}")
        return float(delta_elevation), float(delta_across), float(delta_along)

    def _report_result(self, delta_elevation: float, delta_across: float, delta_along: float) -> Dict | None:
        """
        Prepare the result for the report JSON file
        """
        result = {
            "delta_elevation": delta_elevation,
            "delta_across": delta_across,
            "delta_along": delta_along,
        }

        # Using rsocket (if present) to send the result
        rsocket_msg_emitter = exec_ctx.get_rsocket_msg_emitter()
        if rsocket_msg_emitter is not None:
            rsocket_msg_emitter.emit_map_of_double(result)
            return None

        return {"result": result}
