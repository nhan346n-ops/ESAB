#! /usr/bin/env python3
# coding: utf-8
from typing import Dict

import pygws.service.execution_context as exec_ctx
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.utils.pyat_logger as log
from pyat.sonarscope.bs_correction.mean_bs_model import (
    BackscatterCurve,
    MeanBSModel,
)


class BSReferenceLevelEvaluator:
    def __init__(
        self,
        mean_model_file: str,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor.
        :param : mean_model_file : path of the bsar file to analyse
        """
        self.mean_model_file = mean_model_file
        self.monitor = monitor
        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __call__(self) -> Dict | None:
        """Run method."""
        self.logger.info(f"Starting reference level evaluation of {self.mean_model_file}")
        mean_model = MeanBSModel.read_from_netcdf(self.mean_model_file)
        # get an angular independent response model for the surveyed area
        avg = 0.0
        count = 0
        for mode, (curve_by_incidence, _) in mean_model.model.items():
            weights = curve_by_incidence.ds[BackscatterCurve.VALUE_COUNT].fillna(0)
            weighted = curve_by_incidence.ds[BackscatterCurve.MEAN_BS].weighted(weights)
            avg += weighted.sum()
            count += weights.sum()

        avg = float(avg / count)

        return self._report_result(mean_bs=avg)

    def _report_result(self, mean_bs: float) -> Dict | None:
        """
        Serialize the result in JSON format
        """
        result = {
            "reference_level": mean_bs,
        }

        # Using rsocket (if present) to send the result
        rsocket_msg_emitter = exec_ctx.get_rsocket_msg_emitter()
        if rsocket_msg_emitter is not None:
            rsocket_msg_emitter.emit_map_of_double(result)
            return None

        return {"result": result}
