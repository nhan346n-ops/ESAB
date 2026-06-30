#! /usr/bin/env python3
# coding: utf-8
from typing import Dict, List, Tuple

import numpy as np
import pygws.service.execution_context as exec_ctx
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from pyproj import Geod

import pyat.sounder.sounder_driver_factory as sounder_driver_factory
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.sounder.sounder_driver import SounderDriver


class SpatialResolutionEvaluator:
    def __init__(
        self,
        i_paths: List[str],
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor.
        :param : i_paths : path of the sounding files to analyse
        :param : result : path of the resulting json file
        """
        i_paths = arg_util.parse_list_of_files("i_paths", i_paths)
        self.i_path = i_paths[0]

        # Prefer to use RSocket monitor if available
        self.monitor = monitor
        if exec_ctx.get_root_progress_monitor() is not None:
            self.monitor = exec_ctx.get_root_progress_monitor()

        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __evaluate_beams(self, beam_iter, geod: Geod) -> float:
        """
        Iterate over the iterator for extracting longitudes and latitudes.
        Evaluate the mean distance between:
          - 2 consecutive beams of the same swath
          - the beams of 2 consecutive swaths
        """
        lons, lats = next(beam_iter)
        mean_distance_between_swath = self.evaluate_mean_distance_between_swath(lons, lats, geod)
        mean_distance_between_beam = self.evaluate_mean_distance_between_beam(lons, lats, geod)
        return max(mean_distance_between_swath, mean_distance_between_beam)

    def evaluate_mean_distance_between_swath(self, lons: np.ndarray, lats: np.ndarray, geod: Geod) -> float:
        """
        returns the mean of all differences between 2 consecutive swath
        """
        _, _, distance = geod.inv(lons[0:-1], lats[0:-1], lons[1:], lats[1:])
        distance[distance == 0] = np.nan  # useful to ignore 0 when using np.nanmean
        return np.nanmean(distance)

    def evaluate_mean_distance_between_beam(self, lons: np.ndarray, lats: np.ndarray, geod: Geod) -> float:
        """
        returns the mean of all differences between 2 consecutive beam of same swath
        """
        _, _, distance = geod.inv(lons[:, 0:-1], lats[:, 0:-1], lons[:, 1:], lats[:, 1:])
        distance[distance == 0] = np.nan  # useful to ignore 0 when using np.nanmean
        return np.nanmean(distance)

    def _evaluate_resolution_meter(self, i_sounder_driver: SounderDriver, geod: Geod) -> float:
        """
        Browse some swaths to evaluate the spatial resolution in meter
        """
        swath_count = i_sounder_driver.sounder_file.swath_count
        if swath_count > 20:
            res_meter = np.nanmean(
                [
                    # Analysing the 10th first swaths
                    self.__evaluate_beams(i_sounder_driver.iter_beam_positions(10), geod),
                    # Analysing the 10th last swaths
                    self.__evaluate_beams(i_sounder_driver.iter_beam_positions(10, swath_count - 10), geod),
                ]
            )
        else:
            # Analysing all swaths
            res_meter = self.__evaluate_beams(i_sounder_driver.iter_beam_positions(20), geod)

        if np.isfinite(res_meter):
            self.logger.info(f"Evaluation of the spatial resolution in meter is {res_meter}")
        else:
            self.logger.error(f"Evaluation of the spatial resolution in meter is {res_meter}")

        return res_meter

    def _round_resolution_meter(self, res_meter: float) -> float:
        """
        Round the resolution in meter according to its precision
        """
        if res_meter > 100.0:
            # Precision 10m
            return round(res_meter, -1)

        if res_meter > 10.0:
            # Precision 1m
            return round(res_meter)

        # Precision 0.01m
        return round(res_meter, 2)

    def _evaluate_resolution_degree(self, i_sounder_driver: SounderDriver, res_meter: float, geod: Geod) -> float:
        """
        Use the spatial resolution in meter as distance from one point of the navigation to evaluate the resolution in degree
        """
        nav_point = int(i_sounder_driver.sounder_file.swath_count / 2)
        lons = i_sounder_driver.read_platform_longitudes()
        lats = i_sounder_driver.read_platform_latitudes()

        lon = lons.flat[nav_point]
        lon2, _, _ = geod.fwd(lon, lats.flat[nav_point], 90.0, res_meter)
        res_degree = abs(lon - lon2)

        if np.isfinite(res_degree):
            self.logger.info(f"Evaluation of the spatial resolution in degree {res_degree}")
        else:
            self.logger.error(f"Evaluation of the spatial resolution in degree {res_degree}")

        return res_degree

    def evaluate(self) -> Tuple[float, float]:
        """
        Process the evaluation of the spatial resolution.
        Return the result as a tuple of float. [0] for meter value. [1] for the degree one
        """
        self.logger.info(f"Starting spatial resolution evaluation of {self.i_path}")
        self.monitor.begin_task("Evalutating the spatial resolution", 100)

        geod = Geod(ellps="WGS84")
        res_meter = 2.0
        with sounder_driver_factory.open_sounder(self.i_path) as i_sounder_driver:
            self.monitor.worked(10)
            res_meter = self._evaluate_resolution_meter(i_sounder_driver, geod)
            self.monitor.worked(30)
            res_meter = self._round_resolution_meter(res_meter)
            self.monitor.worked(30)
            res_degree = self._evaluate_resolution_degree(i_sounder_driver, res_meter, geod)
            self.monitor.worked(30)

        return res_meter, res_degree

    def __call__(self) -> Dict | None:
        """Run method."""
        res_meter, res_degree = self.evaluate()
        return self._report_result(res_meter, res_degree)

    def _report_result(self, res_meter: float, res_degree: float) -> Dict | None:
        """
        Serialize the result in JSON format
        """
        result = {
            "meter": res_meter,
            "degree": res_degree,
        }

        # Using rsocket (if present) to send the result
        rsocket_msg_emitter = exec_ctx.get_rsocket_msg_emitter()
        if rsocket_msg_emitter is not None:
            rsocket_msg_emitter.emit_map_of_double(result)
            return None

        return {"result": result}
