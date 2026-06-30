#! /usr/bin/env python3
# coding: utf-8
from typing import Dict, List

import numpy as np
import pygws.service.execution_context as exec_ctx
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from pyproj import Transformer, crs

import pyat.sounder.sounder_driver_factory as sounder_driver_factory
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log


class GeoboxEvaluator:
    def __init__(
        self,
        i_paths: List[str],
        target_spatial_reference: str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor.
        :param : i_paths : path of the sounding files to analyse
        """
        self.i_paths = arg_util.parse_list_of_files("i_paths", i_paths)
        self.crs = crs.CRS.from_proj4(target_spatial_reference)

        # Prefer to use RSocket monitor if available
        self.monitor = monitor
        if exec_ctx.get_root_progress_monitor() is not None:
            self.monitor = exec_ctx.get_root_progress_monitor()

        self.logger = log.logging.getLogger(self.__class__.__name__)

    def evaluate(self, i_sounder_driver) -> arg_util.Geobox:
        """
        Process the evaluation of the geobox.
        """
        latitudes = i_sounder_driver.read_detection_latitude()
        longitudes = i_sounder_driver.read_detection_longitude()

        # Prepare a CRS transformer to convert LatLon to specified projection
        xs, ys = longitudes, latitudes
        if self.crs.is_projected:
            transform = Transformer.from_crs(
                crs.CRS.from_epsg(4326),
                self.crs,
                always_xy=True,
            )
            xs, ys = transform.transform(longitudes, latitudes, radians=False)

        swath_count = latitudes.shape[0]
        validities = i_sounder_driver.read_validity_flags(0, swath_count)
        ys[~validities] = np.nan
        xs[~validities] = np.nan
        result = arg_util.Geobox(
            upper=np.nanmax(ys),
            lower=np.nanmin(ys),
            right=np.nanmax(xs),
            left=np.nanmin(xs),
        )
        return result

    def _report_result(self, geobox: arg_util.Geobox) -> Dict | None:
        """
        Serialize the result in JSON format
        """
        result = {"top": geobox.upper, "bottom": geobox.lower, "left": geobox.left, "right": geobox.right}

        # Using rsocket (if present) to send the result
        rsocket_msg_emitter = exec_ctx.get_rsocket_msg_emitter()
        if rsocket_msg_emitter is not None:
            rsocket_msg_emitter.emit_map_of_double(result)
            return None

        return {"result": result}

    def __call__(self) -> Dict | None:
        """Run method."""
        result: arg_util.Geobox = None
        self.monitor.begin_task("Evalutating the geobox", 100 * len(self.i_paths))

        for i_path in self.i_paths:
            self.logger.info(f"Starting geobox evaluation of {i_path}")
            with sounder_driver_factory.open_sounder(i_path) as i_sounder_driver:
                self.monitor.worked(10)
                geobox = self.evaluate(i_sounder_driver)
                self.monitor.worked(80)
                self.logger.info(f"Evaluated geobox : {str(geobox)}")
                if result is None:
                    result = geobox
                else:
                    result.extend(geobox.upper, geobox.lower, geobox.left, geobox.right)
                self.monitor.worked(10)
        self.monitor.done()

        return self._report_result(result)
