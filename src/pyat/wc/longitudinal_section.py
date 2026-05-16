# pylint:disable=no-member
import math
import os
from typing import List

import numpy as np
import scipy
import sonarnative
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from sonarnative import MemEchos, SpatializerHolder

import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.utils.coords import compute_detection_position, compute_distance
from pyat.utils.numpy_utils import interp1d_nan
from pyat.utils.signal import db_to_energy
from pyat.wc.longitudinal_section_gridder import LongitudinalSectionGridder
from pyat.wc.utils.filter import apply_filters
from pyat.wc.utils.statistics import get_biggest_echoes_count, get_xsf_statistics
from pyat.wc.wc_constants import contains_compensated_layer, contains_raw_layer
from pyat.xsf.xsf_driver import (
    TRANSDUCER_OFFSET_X,
    WATERLINE_TO_CHART_DATUM,
    open_xsf,
)


class LongitudinalSection:
    """
    longitudinal section of backscatters
    """

    def __init__(
        self,
        i_paths: List[str],
        o_paths: List[str],
        monitor: ProgressMonitor = DefaultMonitor,
        delta_across: float = 0,
        delta_elevation: float = 0,
        delta_along: float = 0,
        grid_count: int = 0,
        interpolate: bool = False,
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
        self.delta_along: float = arg_util.parse_float("delta_along", delta_along)
        self.delta_across: float = arg_util.parse_float("delta_across", delta_across)
        self.interpolate = interpolate
        self.json_filters = filters
        self.layers = arg_util.parse_list_of_str(layers)
        self.normalization_offset = arg_util.parse_float("normalization_offset", normalization_offset)
        self.overwrite = overwrite
        self.grid_count: int = arg_util.parse_int("grid_count", grid_count)

        # Nb of swath per grid
        self.swath_count = 0
        # Nb of column per grid
        self.col_count = 0
        # Nb of row per grid
        self.row_count = 0

        # Min/Max values read in xsf files
        self.min_across, self.max_across, self.min_elevation, self.max_elevation = (np.nan, np.nan, np.nan, np.nan)

        # Nb of expected grids
        if self.delta_across <= 0 and self.grid_count <= 0:
            raise ValueError("arguments delta_across or grid_count must take one positive value")

        if self.delta_elevation <= 0:
            raise ValueError("arguments delta_elevation must take one positive value")

    def __call__(self):
        if len(self.layers) == 0:
            self.logger.error("No layer to export. Please check at least one layer")
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
                "File already exists and overwrite not allowed (allow overwrite with option : '-o --overwrite)"
            )
            return

        gridder = self._prepare_gridder(in_xsfs)
        monitor.set_work_remaining(2)
        self._grid_xsf(in_xsfs, gridder, monitor.split(1))
        self.logger.info("Finalizing")
        if gridder.values_count() == 0:
            self.logger.error("No output data")
            return

        gridder.finalize(monitor=monitor.split(1), interpolate=self.interpolate)
        self.logger.info(f"Writing file: {out_g3d}")
        gridder.generate_g3d_file(out_g3d)

    def _prepare_gridder(self, in_xsfs: List[str]) -> LongitudinalSectionGridder:
        """
        Create and initialize a LongitudinalSectionGridder
        """
        self._compute_grid_features(in_xsfs)

        # compute col_count
        if self.delta_along == 0:
            self.col_count = self.swath_count
            self.delta_along = self.nav_distances[-1] / self.col_count
        else:
            self.col_count = int(np.ceil(self.nav_distances[-1] / self.delta_along))

        # compute row_count
        if self.delta_elevation > 0:
            self.row_count = int(math.ceil((self.max_elevation - self.min_elevation) / self.delta_elevation))
            self.min_elevation = self.max_elevation - self.row_count * self.delta_elevation

        # compute grid_count
        if self.delta_across != 0.0:
            if self.grid_count == 0:
                min_across_index = np.floor(self.min_across / self.delta_across + 0.5)
                max_across_index = np.ceil(self.max_across / self.delta_across - 0.5)
                self.grid_count = int(max_across_index - min_across_index + 1)
            else:
                min_across_index = -np.floor(self.grid_count / 2)
                max_across_index = min_across_index + self.grid_count - 1
            self.min_across = (min_across_index - 0.5) * self.delta_across
            self.max_across = (max_across_index + 0.5) * self.delta_across

        self.delta_across = (self.max_across - self.min_across) / self.grid_count

        self.logger.info(f"grid_count: {self.grid_count} delta_across: {self.delta_across}")
        self.logger.info(f"row_count: {self.row_count} delta_elevation: {self.delta_elevation}")
        self.logger.info(f"col_count: {self.col_count} delta_along: {self.delta_along}")

        gridder = LongitudinalSectionGridder(
            x_count=self.col_count,
            y_count=self.grid_count,
            z_count=self.row_count,
            min_elevation=self.min_elevation,
            max_elevation=self.max_elevation,
            delta_elevation=self.delta_elevation,
            layers=self.layers,
        )

        gridder = self._initialize_gridder(gridder)
        return gridder

    def _grid_xsf(self, in_xsfs: List[str], gridder: LongitudinalSectionGridder, monitor: ProgressMonitor):
        """
        Invoke the spatialization process on each xsf and fill the gridder with the echoes
        """
        num_fill_grid = 0
        if contains_raw_layer(self.layers):
            num_fill_grid += len(in_xsfs)
        if contains_compensated_layer(self.layers):
            num_fill_grid += len(in_xsfs)
        monitor.begin_task(f"Gridding {len(in_xsfs)} file(s)", num_fill_grid)
        echoes_count = get_biggest_echoes_count(in_xsfs, 1)
        mem_echos = sonarnative.MemEchos(echoes_count)
        swath_origin = 0
        for i_xsf in in_xsfs:
            # memory reservation
            try:
                self.logger.info(f"read file {i_xsf}")
                # retrieve xsf waterline
                with open_xsf(i_xsf) as xsf_driver:
                    waterline_to_chart_datum = xsf_driver[WATERLINE_TO_CHART_DATUM][:]

                spatializer = sonarnative.open_spatializer(i_xsf, -1, True)
                # setup filters
                apply_filters(self.json_filters, spatializer)

                if contains_raw_layer(self.layers):
                    self.logger.info("compute raw layers")
                    self._fill_grid(
                        gridder=gridder,
                        monitor=monitor.split(1),
                        spatializer=spatializer,
                        mem_echos=mem_echos,
                        waterline=waterline_to_chart_datum,
                        swath_origin=swath_origin,
                        compensated=False,
                    )

                if contains_compensated_layer(self.layers):
                    self.logger.info("compute compensated layers")
                    # setup image processing
                    native_param = sonarnative.RangeNormalizationParameter(True, self.normalization_offset)
                    sonarnative.apply_range_normalization_signal_processing(spatializer, native_param)

                    self._fill_grid(
                        gridder=gridder,
                        monitor=monitor.split(1),
                        spatializer=spatializer,
                        mem_echos=mem_echos,
                        waterline=waterline_to_chart_datum,
                        swath_origin=swath_origin,
                        compensated=True,
                    )
            finally:
                swath_origin += spatializer.get_swath_count()
                sonarnative.close_spatializer(spatializer)
        monitor.done()

    def _fill_grid(
        self,
        gridder: LongitudinalSectionGridder,
        monitor: ProgressMonitor,
        spatializer: SpatializerHolder,
        mem_echos: MemEchos,
        waterline: np.ndarray,
        swath_origin: int,
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
            across = mem_echos.across

            # compute across indices
            y_idx = np.around((across - self.min_across) / self.delta_across - 0.5).astype(int)
            # estimate along index for current swath
            estimated_x_idx = int(self.nav_distances[swath_origin + sp_swath] / self.delta_along)

            # values from sonarnative are sent in the gridder
            gridder.fill_grid(
                sound_lon=mem_echos.longitude,
                sound_lat=mem_echos.latitude,
                sound_elev=elevation,
                sound_backscatter=echos,
                y_idx=y_idx,
                init_x_idx=estimated_x_idx,
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
        self.nav_latitudes = np.ndarray(0)
        self.nav_longitudes = np.ndarray(0)
        self.nav_headings = np.ndarray(0)
        self.swath_count = 0
        self.beam_count = 0
        for i_xsf in in_xsfs:
            with open_xsf(i_xsf) as xsf_driver:
                xsf_min_across, xsf_max_across, xsf_min_elevation, xsf_max_elevation, _ = get_xsf_statistics(xsf_driver)
                self.min_across = np.nanmin([self.min_across, xsf_min_across])
                self.max_across = np.nanmax([self.max_across, xsf_max_across])
                self.min_elevation = np.nanmin([self.min_elevation, xsf_min_elevation])
                self.max_elevation = np.nanmax([self.max_elevation, xsf_max_elevation])
                self.nav_latitudes = np.append(self.nav_latitudes, xsf_driver.read_platform_latitudes())
                self.nav_longitudes = np.append(self.nav_longitudes, xsf_driver.read_platform_longitudes())
                self.nav_headings = np.append(self.nav_headings, xsf_driver.read_platform_headings())
                self.swath_count += int(xsf_driver.sounder_file.swath_count)
                self.beam_count = np.nanmax([self.beam_count, xsf_driver.sounder_file.beam_count])
                rx_transducer_index = xsf_driver.get_rx_transducers()
                # retrive along position of the first rx transducer
                if rx_transducer_index is None or len(rx_transducer_index) == 0:
                    self.rx_along_offset = 0.0
                    self.logger.error(f"no transducer found in {i_xsf}")
                else:
                    self.rx_along_offset = xsf_driver[TRANSDUCER_OFFSET_X][rx_transducer_index[0]]

        self.nav_longitudes = interp1d_nan(self.nav_longitudes)
        self.nav_latitudes = interp1d_nan(self.nav_latitudes)
        self.nav_headings = interp1d_nan(self.nav_headings)
        self.nav_distances = compute_distance(self.nav_longitudes, self.nav_latitudes)
        self.nav_distances = np.cumsum(self.nav_distances)

    def _initialize_gridder(self, gridder: LongitudinalSectionGridder):
        """
        Initialize a SectionGridder
        """
        grid_gap_across = np.linspace(
            self.min_across + self.delta_across / 2, self.max_across - self.delta_across / 2, gridder.y_count
        )
        grid_gap_along = np.linspace(0, (self.col_count - 1) * self.delta_along, gridder.x_count)
        # latitudes
        f_lat_cos = scipy.interpolate.interp1d(self.nav_distances, np.cos(np.radians(self.nav_latitudes)))
        f_lat_sin = scipy.interpolate.interp1d(self.nav_distances, np.sin(np.radians(self.nav_latitudes)))
        grid_latitudes = np.degrees(np.arctan2(f_lat_sin(grid_gap_along), f_lat_cos(grid_gap_along)))
        # longitudes
        f_lon_cos = scipy.interpolate.interp1d(self.nav_distances, np.cos(np.radians(self.nav_longitudes)))
        f_lon_sin = scipy.interpolate.interp1d(self.nav_distances, np.sin(np.radians(self.nav_longitudes)))
        grid_longitudes = np.degrees(np.arctan2(f_lon_sin(grid_gap_along), f_lon_cos(grid_gap_along)))
        # headings
        f_head_cos = scipy.interpolate.interp1d(self.nav_distances, np.cos(np.radians(self.nav_headings)))
        f_head_sin = scipy.interpolate.interp1d(self.nav_distances, np.sin(np.radians(self.nav_headings)))
        grid_headings = np.degrees(np.arctan2(f_head_sin(grid_gap_along), f_head_cos(grid_gap_along)))

        # use tx along position as reference for the gridder
        grid_along = np.full_like(grid_gap_across, self.rx_along_offset)
        for x_idx in range(gridder.x_count):
            # with sended values, compute the positions of the reference grids
            col_lons, col_lats = compute_detection_position(
                grid_along, grid_gap_across, grid_longitudes[x_idx], grid_latitudes[x_idx], grid_headings[x_idx]
            )
            for y_idx, (lon, lat) in enumerate(zip(col_lons, col_lats)):
                gridder.add(lon=lon, lat=lat, x_idx=x_idx, y_idx=y_idx)

        gridder.initialize_grid()
        return gridder
