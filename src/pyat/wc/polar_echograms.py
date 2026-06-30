# pylint:disable=no-member
import math
import os
from typing import List

import numpy as np
import sonarnative
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from skspatial.objects import Plane, Point, Points
from sonar_netcdf.utils import nc_merger
from sonarnative import MemEchos, SpatializerHolder

import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.function.evaluate_polar_echograms_arguments import PolarEchogramsArgumentsEvaluator
from pyat.sounder import sounder_driver_factory
from pyat.utils.coords import compute_detection_position
from pyat.utils.numpy_utils import interp1d_nan
from pyat.utils.signal import db_to_energy
from pyat.wc.polar_echograms_gridder import PolarEchogramsGridder
from pyat.wc.utils.filter import apply_filters
from pyat.wc.utils.statistics import get_biggest_echoes_count, get_xsf_statistics
from pyat.wc.wc_constants import contains_compensated_layer, contains_raw_layer
from pyat.xsf.xsf_driver import PING_TIME, TX_TRANSDUCER_DEPTH, WATERLINE_TO_CHART_DATUM

POLAR_MAX_INTERPOLATION_LIMIT = 64
POLAR_MAX_HEIGHT = 4096


class PolarEchograms:
    """
    polar echograms rasterisation
    """

    def __init__(
        self,
        i_paths: List[str],
        o_paths: List[str],
        monitor: ProgressMonitor = DefaultMonitor,
        sample_resolution: float = 0,
        height: float = 0,
        interpolate: bool = True,
        filters: str | None = None,
        layers: List[str] | None = None,
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
        self.sample_resolution: float = arg_util.parse_float("sample_resolution", sample_resolution)
        self.height: float = arg_util.parse_float("height", height)
        self.interpolate = interpolate
        self.json_filters = filters
        self.layers = arg_util.parse_list_of_str(layers)
        self.normalization_offset: float = arg_util.parse_float("normalization_offset", normalization_offset)
        self.overwrite = overwrite

        # Nb of column per grid
        self.max_col_count = 0
        # Nb of row per grid
        self.max_row_count = 0

        # Min/Max values read in xsf files
        self.min_across, self.max_across = (np.nan, np.nan)

        # expected resolution
        if self.sample_resolution <= 0:
            self.sample_resolution, _ = PolarEchogramsArgumentsEvaluator(
                i_paths=self.i_paths, monitor=self.monitor
            ).evaluate()

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
        # Get start and stop time and time sort file list
        sorted_file_list = nc_merger.time_sort_files(in_xsfs)
        in_xsfs = [file for file, _, _ in sorted_file_list]

        gridder = self._prepare_gridder(in_xsfs, out_g3d)
        try:
            self._grid_xsf(in_xsfs, gridder, monitor.split(1))
        finally:
            self.logger.info(f"Writing file: {out_g3d}")
            gridder.flush_dataset()

    def _prepare_gridder(self, in_xsfs: List[str], out_g3d: str) -> PolarEchogramsGridder:
        """
        Create and initialize a PolarEchogramsGridder
        """
        self._compute_grid_features(in_xsfs)

        # compute grid_count
        self.grid_count = self.swath_count

        # compute indicative col_count and row_count
        # add 10% on max depth
        self.max_row_count = int(math.ceil(1.1 * self.max_vertical_distance / self.sample_resolution))
        self.max_col_count = int(math.ceil(1.1 * (self.max_across - self.min_across) / self.sample_resolution))
        self.max_abs_across = max(np.fabs(self.max_across), np.fabs(self.min_across))
        self.delta_elevation = self.sample_resolution
        self.delta_across = self.sample_resolution

        self.logger.info(f"grid_count: {self.grid_count}")
        self.logger.info(f"indicative row_count: {self.max_row_count} delta_elevation: {self.delta_elevation}")
        self.logger.info(f"indicative col_count: {self.max_col_count} delta_across: {self.delta_across}")

        # initialize a gridder with default size buffer. Buffer will be axpanded later if necessary
        gridder = PolarEchogramsGridder(
            x_count=self.grid_count,
            y_count=self.max_col_count,
            z_count=self.max_row_count,
            layers=self.layers,
            output_path=out_g3d,
        )

        gridder = self._initialize_gridder(gridder)
        return gridder

    def _grid_xsf(self, in_xsfs: List[str], gridder: PolarEchogramsGridder, monitor: ProgressMonitor):
        """
        Invoke the spatialization process on each xsf and fill the gridder with the echoes
        """
        num_fill_layers = len(in_xsfs)
        monitor.begin_task(f"Gridding {len(in_xsfs)} file(s)", num_fill_layers)
        echoes_count = get_biggest_echoes_count(in_xsfs, 1)
        mem_echos = sonarnative.MemEchos(echoes_count)
        swath_origin = 0
        for i_xsf in in_xsfs:
            submonitor = monitor.split(1)
            # memory reservation
            try:
                self.logger.info(f"read file {i_xsf}")
                # retrieve xsf waterline
                with sounder_driver_factory.open_sounder(i_xsf) as xsf_driver:
                    waterline_to_chart_datum = xsf_driver[WATERLINE_TO_CHART_DATUM][:]
                    transducer_depth = xsf_driver[TX_TRANSDUCER_DEPTH][:]
                    ping_time = xsf_driver[PING_TIME][:]

                spatializer = sonarnative.open_spatializer(i_xsf, -1, True)
                # setup filters
                apply_filters(self.json_filters, spatializer)

                submonitor.begin_task("Compute", spatializer.get_swath_count())
                for sp_swath in range(spatializer.get_swath_count()):
                    submonitor.worked(1)
                    # reinit gridder for swath
                    gridder.reset_grid()

                    if contains_raw_layer(self.layers):
                        native_param = sonarnative.RangeNormalizationParameter(False, self.normalization_offset)
                        sonarnative.apply_range_normalization_signal_processing(spatializer, native_param)
                        self._fill_grid(
                            gridder=gridder,
                            spatializer=spatializer,
                            mem_echos=mem_echos,
                            waterline=waterline_to_chart_datum,
                            transducer_depth=transducer_depth,
                            swath_origin=swath_origin,
                            swath_index=sp_swath,
                            ping_time=ping_time,
                            compensated=False,
                        )

                    if contains_compensated_layer(self.layers):
                        # setup image processing
                        native_param = sonarnative.RangeNormalizationParameter(True, self.normalization_offset)
                        sonarnative.apply_range_normalization_signal_processing(spatializer, native_param)
                        self._fill_grid(
                            gridder=gridder,
                            spatializer=spatializer,
                            mem_echos=mem_echos,
                            waterline=waterline_to_chart_datum,
                            transducer_depth=transducer_depth,
                            swath_origin=swath_origin,
                            swath_index=sp_swath,
                            ping_time=ping_time,
                            compensated=True,
                        )

                    gridder.finalize(interpolate=self.interpolate)
                    gridder.add_g3d_grid(grid_idx=swath_origin + sp_swath)
                    # flush dataset every 100 swaths to avoid too much memory usage
                    # flushing every swath can cause performance issue due to too many I/O operations,
                    # especially if swath count is high. So we flush every 100 swaths as a compromise.
                    if (sp_swath + 1) % 100 == 0:
                        gridder.flush_dataset()
                swath_origin += spatializer.get_swath_count()
            finally:
                gridder.flush_dataset()
                sonarnative.close_spatializer(spatializer)
            submonitor.done()
        monitor.done()

    def _fill_grid(
        self,
        gridder: PolarEchogramsGridder,
        spatializer: SpatializerHolder,
        mem_echos: MemEchos,
        waterline: np.ndarray,
        transducer_depth: np.ndarray,
        ping_time: np.ndarray,
        swath_origin: int,
        swath_index: int,
        compensated: bool = False,
    ):

        # spatialization
        sonarnative.spatialize_in_memory(spatializer, swath_index, 1, mem_echos)

        if mem_echos.size == 0:
            return

        # memechos elevations are relative to surface
        elevation = mem_echos.elevation

        # transform reflectivity in db to natural energy
        echos = db_to_energy(value=mem_echos.echo)
        across = mem_echos.across
        along = mem_echos.along
        opening = mem_echos.non_overlapping_beam_opening_angle

        # compute swath index
        x_idx = swath_index + swath_origin

        # setup grid (only one time per swath)
        if gridder.y_count == 0 and gridder.z_count == 0:
            min_across = np.nanmin(across)
            max_across = np.nanmax(across)

            min_elevation = np.nanmin(elevation)
            max_elevation = -transducer_depth[swath_index]

            # check validity
            min_across = np.nanmax([min_across, -2 * self.max_abs_across])
            max_across = np.nanmin([max_across, 2 * self.max_abs_across])
            min_elevation = np.nanmax([min_elevation, max_elevation - POLAR_MAX_HEIGHT * self.delta_elevation])

            # compute grid corners
            # estimate plane best fitting 3D points with 100 points from echoes
            data = np.column_stack((across, elevation, along))
            choices = np.random.choice(mem_echos.size, 100)
            points = Points(data[choices])
            plane = Plane.best_fit(points)

            grid_gap_across = np.array([min_across, max_across])
            grid_gap_elevation = np.array([min_elevation, max_elevation])

            top_left = Point([grid_gap_across[0], grid_gap_elevation[1], 0]) - plane.point
            bottom_right = Point([grid_gap_across[1], grid_gap_elevation[0], 0]) - plane.point

            result_points = plane.to_points(
                lims_x=(top_left[0], bottom_right[0]), lims_y=(top_left[1], bottom_right[1])
            )
            grid_lons, grid_lats = compute_detection_position(
                along=result_points[:, 2],
                across=result_points[:, 0],
                nav_longitude=self.nav_longitudes[x_idx],
                nav_latitude=self.nav_latitudes[x_idx],
                heading=self.nav_headings[x_idx],
            )

            # setup grid corners shifted relative to waterline
            gridder.set_top_left(
                lon=grid_lons[0], lat=grid_lats[0], elevation=waterline[swath_index] + grid_gap_elevation[1]
            )
            gridder.set_top_right(
                lon=grid_lons[1], lat=grid_lats[1], elevation=waterline[swath_index] + grid_gap_elevation[1]
            )
            gridder.set_bottom_left(
                lon=grid_lons[2], lat=grid_lats[2], elevation=waterline[swath_index] + grid_gap_elevation[0]
            )
            gridder.set_bottom_right(
                lon=grid_lons[3], lat=grid_lats[3], elevation=waterline[swath_index] + grid_gap_elevation[0]
            )

            # add optional attributes
            gridder.set_min_max_across(min_across=min_across, max_across=max_across)
            gridder.set_time(
                time=np.datetime_as_string(ping_time[swath_index].astype("datetime64[ns]"), timezone="UTC")
            )

            # estimate across spacing
            opening = np.nanmax(opening)
            across_dist = np.max([np.fabs([min_across, max_across])])
            across_spacing = (
                np.tan(np.deg2rad(opening / 2))
                * 2
                * (np.square(across_dist) + np.square(max_elevation - min_elevation))
                / across_dist
            )
            across_limit = np.ceil(1.1 * across_spacing / self.delta_across + 1).astype(int)
            gridder.set_interpolate_limit(min(across_limit, POLAR_MAX_INTERPOLATION_LIMIT))

            max_y_idx = np.around((max_across - min_across) / self.delta_across).astype(int)
            max_z_idx = np.around((-transducer_depth[swath_index] - min_elevation) / self.delta_elevation).astype(int)

            # set effective size of current grid
            gridder.set_size(y_count=max_y_idx + 1, z_count=max_z_idx + 1)

        # compute across indices
        y_idx = np.around((across - gridder.min_across) / self.delta_across).astype(int)
        # compute elevation indices (relative to transducer)
        z_idx = np.around((-transducer_depth[swath_index] - elevation) / self.delta_elevation).astype(int)

        # values from sonarnative are sent in the gridder
        gridder.fill_grid(sound_backscatter=echos, y_idx=y_idx, z_idx=z_idx, compensated=compensated)

    def _compute_grid_features(self, in_xsfs: List[str]):
        """
        Compute the grids features to allow the gridder initialization
        """
        self.nav_latitudes = np.ndarray(0)
        self.nav_longitudes = np.ndarray(0)
        self.nav_headings = np.ndarray(0)
        self.max_vertical_distance = 0
        self.swath_count = 0
        for i_xsf in in_xsfs:
            with sounder_driver_factory.open_sounder(i_xsf) as xsf_driver:
                xsf_min_across, xsf_max_across, _, _, xsf_max_vertical_distance = get_xsf_statistics(xsf_driver)
                self.min_across = np.nanmin([self.min_across, xsf_min_across])
                self.max_across = np.nanmax([self.max_across, xsf_max_across])
                self.max_vertical_distance = np.nanmax([self.max_vertical_distance, xsf_max_vertical_distance])
                self.swath_count += int(xsf_driver.sounder_file.swath_count)
                self.nav_latitudes = np.append(self.nav_latitudes, xsf_driver.read_platform_latitudes())
                self.nav_longitudes = np.append(self.nav_longitudes, xsf_driver.read_platform_longitudes())
                self.nav_headings = np.append(self.nav_headings, xsf_driver.read_platform_headings())

        self.nav_longitudes = interp1d_nan(self.nav_longitudes)
        self.nav_latitudes = interp1d_nan(self.nav_latitudes)
        self.nav_headings = interp1d_nan(self.nav_headings)

    def _initialize_gridder(self, gridder: PolarEchogramsGridder):
        """
        Initialize a PolarEchogramsGridder
        """

        gridder.initialize_grid()
        gridder.initialize_g3d_file()

        return gridder
