# pylint:disable=no-member
import os
from typing import Dict, List, Optional

import numpy as np
import sonarnative
from osgeo import osr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from sonarnative import MemEchos, SpatializerHolder

import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.sounder import sounder_driver_factory
from pyat.utils.signal import db_to_energy
from pyat.wc import wc_constants
from pyat.wc.horizontal_section_gridder import HorizontalSectionGridder
from pyat.wc.utils.filter import apply_filters
from pyat.wc.utils.statistics import get_biggest_echoes_count, get_xsf_statistics
from pyat.wc.wc_constants import (
    contains_compensated_layer,
    contains_elevation_layer,
    contains_raw_layer,
)
from pyat.xsf.xsf_driver import WATERLINE_TO_CHART_DATUM


class HorizontalSection:
    """
    Horizontal section of backscatters
    """

    def __init__(
        self,
        i_paths: List[str],
        o_paths: List[str],
        monitor: ProgressMonitor = DefaultMonitor,
        delta_elevation: float = 0,
        grid_count: int = 0,
        vertical_offset: float = 0,
        vertical_reference: str = None,
        coord: Optional[Dict] = None,
        target_resolution: float = 1.0 / 3600.0,
        filters: str = None,
        layers: List[str] = None,
        normalization_offset: float = 0,
        overwrite: bool = False,
    ):
        """
        Constructor.
        """
        self.logger = log.logging.getLogger(self.__class__.__name__)
        self.i_paths: List[str] = i_paths
        self.o_paths: List[str] = o_paths
        self.monitor = monitor
        self.delta_elevation: float = arg_util.parse_float("delta_elevation", delta_elevation)
        self.json_filters = filters
        self.layers = arg_util.parse_list_of_str(layers)
        self.normalization_offset: float = arg_util.parse_float("normalization_offset", normalization_offset)
        self.overwrite = overwrite
        self.grid_count: int = arg_util.parse_int("grid_count", grid_count)
        self.vertical_offset: float = arg_util.parse_float("vertical_offset", vertical_offset)
        self.vertical_reference: str = vertical_reference
        self.spatial_resolution: float = arg_util.parse_float("target_resolution", target_resolution)
        self.coord = coord
        self.bottom = "sea_floor" == self.vertical_reference
        if self.bottom:
            self.layers.append(wc_constants.ELEVATION)
        if coord is not None:
            self.geobox = arg_util.parse_geobox("coord", coord)
            self.geobox.spatial_reference = osr.SpatialReference()
            self.geobox.spatial_reference.ImportFromProj4("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

        # Min/Max values read in xsf files
        self.min_elevation, self.max_elevation, self.max_vertical_distance = (np.nan, np.nan, np.nan)

    def __call__(self):
        if len(self.layers) == 0:
            self.logger.error(f"No layer to export. Please check at least one layer")
            return
        if len(self.i_paths) > 1 and len(self.o_paths) == 1:
            # Case of a merge
            self._convert(self.i_paths, self.o_paths[0], self.monitor)
        else:
            # Case of a single conversion
            self.monitor.set_work_remaining(len(self.i_paths))
            for in_xsf, out_g3d in zip(self.i_paths, self.o_paths):
                self._convert([in_xsf], out_g3d, self.monitor.split(1))

    def _convert(self, in_xsfs: List[str], out_g3d: str, monitor: ProgressMonitor):
        """
        Initiate a gridder
        Invoke the spatialization process on each xsf and fill the gridder with the echoes
        Finalize the gridder
        Write the resulting G3D
        """
        if os.path.exists(out_g3d) and not self.overwrite:
            self.logger.error(
                f"File already exists and overwrite not allowed (allow overwrite with option : '-o --overwrite)"
            )
            return

        gridder = self._prepare_gridder(in_xsfs)
        self._grid_xsf(in_xsfs, gridder, monitor)
        self.logger.info(f"Finalizing")
        if gridder.values_count() == 0:
            self.logger.error(f"No output data")
            return

        gridder.finalize()
        self.logger.info(f"Writing file: {out_g3d}")
        gridder.generate_g3d_file(out_g3d)

    def _prepare_gridder(self, in_xsfs: List[str]) -> HorizontalSectionGridder:
        """
        Create and initialize a HorizontalSectionGridder
        """
        self._compute_grid_features(in_xsfs)

        # compute elevation bounds
        if self.delta_elevation == 0:
            if self.grid_count == 0:
                self.grid_count = 1
            if self.bottom:
                self.delta_elevation = self.max_vertical_distance / self.grid_count
                self.min_elevation = 0
                self.max_elevation = self.max_vertical_distance
            else:
                self.delta_elevation = (self.max_elevation - self.min_elevation) / self.grid_count
        elif self.bottom:  # self.delta_elevation != 0.0
            min_elevation_index = 0
            if self.grid_count == 0:
                max_elevation_index = np.floor(self.max_vertical_distance / self.delta_elevation)
                self.grid_count = int(max_elevation_index - min_elevation_index + 1)
            else:
                max_elevation_index = min_elevation_index + self.grid_count - 1
            self.min_elevation = 0
            self.max_elevation = (max_elevation_index + 1) * self.delta_elevation
        else:  # self.delta_elevation != 0.0
            max_elevation_index = np.ceil(self.max_elevation / self.delta_elevation)
            if self.grid_count == 0:
                min_elevation_index = np.ceil(self.min_elevation / self.delta_elevation)
                self.grid_count = int(max_elevation_index - min_elevation_index + 1)
            else:
                min_elevation_index = max_elevation_index - self.grid_count + 1
            self.min_elevation = (min_elevation_index - 1) * self.delta_elevation
            self.max_elevation = max_elevation_index * self.delta_elevation

        # apply vertical offset
        self.min_elevation += self.vertical_offset
        self.max_elevation += self.vertical_offset

        gridder = HorizontalSectionGridder(
            geobox=self.geobox,
            spatial_resolution=self.spatial_resolution,
            min_elevation=self.min_elevation,
            max_elevation=self.max_elevation,
            delta_elevation=self.delta_elevation,
            layers=self.layers,
        )

        gridder.initialize_grid()
        return gridder

    def _grid_xsf_elevation(self, in_xsfs: List[str], gridder: HorizontalSectionGridder, monitor: ProgressMonitor):
        """
        Read detections elevations on each xsf and fill the gridder with elevation data
        """
        monitor.begin_task(f"Gridding elevation of {len(in_xsfs)} file(s)", 1)
        for i_xsf in in_xsfs:
            with sounder_driver_factory.open_sounder(i_xsf) as xsf_driver:
                latitude = xsf_driver.read_detection_latitude()
                longitude = xsf_driver.read_detection_longitude()
                swath_count = latitude.shape[0]
                validities = xsf_driver.read_validity_flags(0, swath_count)
                elevations = -xsf_driver.read_fcs_depths(0, swath_count)
                elevations[~validities] = np.nan
                gridder.fill_elevations(lon=longitude, lat=latitude, elev=elevations)

        gridder.interpolate_elevations()

    def _grid_xsf(self, in_xsfs: List[str], gridder: HorizontalSectionGridder, monitor: ProgressMonitor):
        """
        Invoke the spatialization process on each xsf and fill the gridder with the echoes
        """
        if contains_elevation_layer(self.layers):
            self._grid_xsf_elevation(in_xsfs, gridder, monitor)

        num_fill_grid = 0
        if contains_raw_layer(self.layers):
            num_fill_grid += len(in_xsfs)
        if contains_compensated_layer(self.layers):
            num_fill_grid += len(in_xsfs)
        monitor.begin_task(f"Gridding {len(in_xsfs)} file(s)", num_fill_grid)
        echoes_count = get_biggest_echoes_count(in_xsfs, 1)
        mem_echos = sonarnative.MemEchos(echoes_count)

        for i_xsf in in_xsfs:
            try:
                self.logger.info(f"read file {i_xsf}")
                # retrieve xsf waterline
                with sounder_driver_factory.open_sounder(i_xsf) as xsf_driver:
                    waterline_to_chart_datum = xsf_driver[WATERLINE_TO_CHART_DATUM][:]
                # memory reservation
                spatializer = sonarnative.open_spatializer(i_xsf, -1, True)
                # setup filters
                apply_filters(self.json_filters, spatializer)

                if contains_raw_layer(self.layers):
                    self.logger.info(f"compute raw layers")
                    self._fill_grid(
                        gridder=gridder,
                        monitor=monitor.split(1),
                        spatializer=spatializer,
                        mem_echos=mem_echos,
                        waterline=waterline_to_chart_datum,
                        compensated=False,
                    )

                if contains_compensated_layer(self.layers):
                    self.logger.info(f"compute compensated layers")
                    # setup image processing
                    native_param = sonarnative.RangeNormalizationParameter(True, self.normalization_offset)
                    sonarnative.apply_range_normalization_signal_processing(spatializer, native_param)

                    self._fill_grid(
                        gridder=gridder,
                        monitor=monitor.split(1),
                        spatializer=spatializer,
                        mem_echos=mem_echos,
                        waterline=waterline_to_chart_datum,
                        compensated=True,
                    )
            finally:
                sonarnative.close_spatializer(spatializer)

    @staticmethod
    def _fill_grid(
        gridder: HorizontalSectionGridder,
        monitor: ProgressMonitor,
        spatializer: SpatializerHolder,
        mem_echos: MemEchos,
        waterline: np.ndarray,
        compensated: bool = False,
    ):
        monitor.begin_task("Compute", spatializer.get_swath_count())
        for sp_swath in range(spatializer.get_swath_count()):
            monitor.worked(1)

            # spatialization
            sonarnative.spatialize_in_memory(spatializer, sp_swath, 1, mem_echos)

            if mem_echos.size == 0:
                continue
            # shift elevations relative to waterline
            elevation = mem_echos.elevation + waterline[sp_swath]

            # transform reflectivity in db to natural energy
            echos = db_to_energy(value=mem_echos.echo)

            # values from sonarnative are sent in the gridder
            gridder.fill_grid(
                sound_lon=mem_echos.longitude,
                sound_lat=mem_echos.latitude,
                sound_elev=elevation,
                sound_backscatter=echos,
                compensated=compensated,
            )
        monitor.done()

    def _compute_grid_features(self, in_xsfs: List[str]):
        """
        Compute the grids features to allow the gridder initialization :
         - Gap in meter between to grids
         - Number of columns in each grid
         - elevation min and max
        """
        self.swath_count = 0
        for i_xsf in in_xsfs:
            with sounder_driver_factory.open_sounder(i_xsf) as xsf_driver:
                _, _, xsf_min_elevation, xsf_max_elevation, xsf_max_vertical_distance = get_xsf_statistics(xsf_driver)
                self.min_elevation = np.nanmin([self.min_elevation, xsf_min_elevation])
                self.max_elevation = np.nanmax([self.max_elevation, xsf_max_elevation])
                self.max_vertical_distance = np.nanmax([self.max_vertical_distance, xsf_max_vertical_distance])
                self.swath_count += int(xsf_driver.sounder_file.swath_count)
