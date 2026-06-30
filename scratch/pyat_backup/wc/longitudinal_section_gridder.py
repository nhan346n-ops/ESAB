# pylint:disable=no-member
import tempfile
from typing import List

import numba as nb
import numpy as np
import pandas as pd
from netCDF4 import Group
from pygws.service.progress_monitor import ProgressMonitor

import pyat.utils.pyat_logger as log
from pyat.utils.nc_encoding import open_nc_file
from pyat.utils.netcdf_utils import DEFAULT_COMPRESSION_LIB
from pyat.utils.signal import energy_to_db
from pyat.wc import wc_constants


class LongitudinalSectionGridder:
    def __init__(
        self,
        x_count: int,
        y_count: int,
        z_count: int,
        min_elevation: float,
        max_elevation: float,
        delta_elevation: float,
        layers: List[str],
    ):
        """
        Args:
            x_count: number of longitudinal elements
            y_count: number of lateral elements
            z_count: number of vertical elements
        """
        self.x_count = x_count
        self.y_count = y_count
        self.z_count = z_count
        self.max_elevation = max_elevation
        self.min_elevation = min_elevation
        self.delta_elevation = delta_elevation
        self.layers = layers

        # Position of horizontal cells
        self.col_head_lon = np.full((y_count, x_count), np.nan)
        self.col_head_lat = np.full((y_count, x_count), np.nan)

        self.logger = log.logging.getLogger(self.__class__.__name__)

        # raw data
        self.map_file_nb_values = None
        self.map_file_backscatter_mean = None
        self.map_file_backscatter_max = None
        self.temp_map_file_nb_value = None
        self.temp_map_file_echo_mean = None
        self.temp_map_file_echo_max = None

        # comp data
        self.map_file_nb_values_comp = None
        self.map_file_backscatter_comp_mean = None
        self.map_file_backscatter_comp_max = None
        self.temp_map_file_nb_value_comp = None
        self.temp_map_file_echo_comp_mean = None
        self.temp_map_file_echo_comp_max = None

        self.grid_count = y_count
        self.row_count = z_count
        self.col_count = x_count

    # pylint: disable=consider-using-with
    def initialize_grid(self):
        self.logger.info(f"Preparing {self.grid_count} grids, size {self.col_count}x{self.row_count} cells each")

        if wc_constants.contains_raw_layer(self.layers):
            # Optimize access to data with numpy array
            self.temp_map_file_nb_value = tempfile.NamedTemporaryFile(
                suffix=wc_constants.BACKSCATTER_VALUE_COUNT + ".memmap"
            )
            self.map_file_nb_values = np.memmap(
                self.temp_map_file_nb_value,
                shape=(self.grid_count, self.row_count, self.col_count),
                dtype=int,
                mode="w+",
            )
            self.map_file_nb_values.fill(0)

            self.temp_map_file_echo_mean = tempfile.NamedTemporaryFile(suffix=wc_constants.BACKSCATTER_MEAN + ".memmap")
            self.map_file_backscatter_mean = np.memmap(
                self.temp_map_file_echo_mean,
                shape=(self.grid_count, self.row_count, self.col_count),
                dtype=np.float32,
                mode="w+",
            )
            self.map_file_backscatter_mean.fill(0)

            # Optimize access to data with numpy array
            self.temp_map_file_echo_max = tempfile.NamedTemporaryFile(suffix=wc_constants.BACKSCATTER_MAX + ".memmap")
            self.map_file_backscatter_max = np.memmap(
                self.temp_map_file_echo_max,
                shape=(self.grid_count, self.row_count, self.col_count),
                dtype=np.float32,
                mode="w+",
            )
            self.map_file_backscatter_max.fill(-np.inf)

        if wc_constants.contains_compensated_layer(self.layers):
            self.temp_map_file_nb_value_comp = tempfile.NamedTemporaryFile(
                suffix=wc_constants.BACKSCATTER_COMP_VALUE_COUNT + "memmap"
            )
            self.map_file_nb_values_comp = np.memmap(
                self.temp_map_file_nb_value_comp,
                shape=(self.grid_count, self.row_count, self.col_count),
                dtype=int,
                mode="w+",
            )
            self.map_file_nb_values_comp.fill(0)

            # Optimize access to data with numpy array
            self.temp_map_file_echo_comp_mean = tempfile.NamedTemporaryFile(
                suffix=wc_constants.BACKSCATTER_COMP_MEAN + ".memmap"
            )
            self.map_file_backscatter_comp_mean = np.memmap(
                self.temp_map_file_echo_comp_mean,
                shape=(self.grid_count, self.row_count, self.col_count),
                dtype=np.float32,
                mode="w+",
            )
            self.map_file_backscatter_comp_mean.fill(0)

            # Optimize access to data with numpy array
            self.temp_map_file_echo_comp_max = tempfile.NamedTemporaryFile(
                suffix=wc_constants.BACKSCATTER_COMP_MAX + ".memmap"
            )
            self.map_file_backscatter_comp_max = np.memmap(
                self.temp_map_file_echo_comp_max,
                shape=(self.grid_count, self.row_count, self.col_count),
                dtype=np.float32,
                mode="w+",
            )
            self.map_file_backscatter_comp_max.fill(-np.inf)

    def add(self, lon: float, lat: float, x_idx: int, y_idx: int) -> None:
        """
        add point in the numpy arrays.
        This points are the head of the columns.
        Args:
            lon: longitudes of the reference grids
            lat: latitudes of the reference grids
            x_idx : longitudinal index
            y_idx : lateral index
        """
        self.col_head_lon[y_idx, x_idx] = lon
        self.col_head_lat[y_idx, x_idx] = lat

    def values_count(self):
        count = 0
        if self.map_file_nb_values is not None:
            count = np.nansum(self.map_file_nb_values)
        if self.map_file_nb_values_comp is not None:
            count = np.nansum(self.map_file_nb_values_comp, initial=count)
        return count

    def finalize(self, monitor: ProgressMonitor, interpolate: bool = False):
        monitor.begin_task("Finalizing", len(self.layers))
        if wc_constants.BACKSCATTER_MEAN in self.layers:
            self.map_file_backscatter_mean[self.map_file_nb_values == 0] = np.nan
            # # post interpolation :
            if interpolate:
                self.logger.info(f"Interpolate {wc_constants.BACKSCATTER_MEAN}")
                self._interpolate(self.map_file_backscatter_mean, monitor=monitor.split(1))
            # return values in db
            # reflectivity mean
            energy_to_db(value=self.map_file_backscatter_mean, out=self.map_file_backscatter_mean)

        if wc_constants.BACKSCATTER_MAX in self.layers:
            self.map_file_backscatter_max[self.map_file_nb_values == 0] = np.nan
            # # post interpolation :
            if interpolate:
                self.logger.info(f"Interpolate {wc_constants.BACKSCATTER_MAX}")
                self._interpolate(self.map_file_backscatter_max, monitor=monitor.split(1))
            # reflectivity max
            energy_to_db(value=self.map_file_backscatter_max, out=self.map_file_backscatter_max)

        if wc_constants.BACKSCATTER_COMP_MEAN in self.layers:
            self.map_file_backscatter_comp_mean[self.map_file_nb_values_comp == 0] = np.nan
            # # post interpolation :
            if interpolate:
                self.logger.info(f"Interpolate {wc_constants.BACKSCATTER_COMP_MEAN}")
                self._interpolate(self.map_file_backscatter_comp_mean, monitor=monitor.split(1))
            # return values in db
            # reflectivity mean
            energy_to_db(value=self.map_file_backscatter_comp_mean, out=self.map_file_backscatter_comp_mean)

        if wc_constants.BACKSCATTER_COMP_MAX in self.layers:
            self.map_file_backscatter_comp_max[self.map_file_nb_values_comp == 0] = np.nan
            # # post interpolation :
            if interpolate:
                self.logger.info(f"Interpolate {wc_constants.BACKSCATTER_COMP_MAX}")
                self._interpolate(self.map_file_backscatter_comp_max, monitor=monitor.split(1))
            # reflectivity max
            energy_to_db(value=self.map_file_backscatter_comp_max, out=self.map_file_backscatter_comp_max)
        monitor.done()

    @staticmethod
    def _interpolate(map_file_backscatter: np.memmap, monitor: ProgressMonitor):
        """
        Final interpolation specific step before writing output g3d file
        """
        nb_grid = map_file_backscatter.shape[0]
        nb_row = map_file_backscatter.shape[1]
        monitor.begin_task("Interpolating", nb_grid)
        for grid_idx in range(nb_grid):
            for row_idx in range(nb_row):
                map_file_backscatter[grid_idx][row_idx] = pd.Series(
                    map_file_backscatter[grid_idx][row_idx]
                ).interpolate(limit=2, limit_direction="both", limit_area="inside")
            monitor.worked(1)
        monitor.done()

    def generate_g3d_file(self, path_g3d: str):
        with open_nc_file(path_g3d, mode="w", nc_format="NETCDF4") as dataset:
            dataset.dataset_type = "FlyTexture"
            dataset.history = "Created by PyAT with LongitudinalSection"

            dataset.createDimension("datalayer_count", len(self.layers))
            datalayer_variable_name = dataset.createVariable("datalayer_variable_name", str, ("datalayer_count",))
            for index, layer in enumerate(self.layers):
                datalayer_variable_name[index] = layer

            height = self.z_count
            length = position = self.x_count
            vector = 2
            if wc_constants.contains_raw_layer(self.layers):
                nb_grid = self.map_file_backscatter_mean.shape[0]
            else:
                nb_grid = self.map_file_backscatter_comp_mean.shape[0]

            for grid_idx in range(nb_grid):
                grp = dataset.createGroup(f"{grid_idx + 1}".zfill(3))
                grp.createDimension("height", height)
                grp.createDimension("length", length)
                grp.createDimension("vector", vector)
                grp.createDimension("position", position)

                elevations = grp.createVariable(
                    "elevation", "f4", ("vector", "position"), compression=DEFAULT_COMPRESSION_LIB
                )
                elevations.units = "meters"
                elevations.long_name = "elevation"
                elevations.standard_name = "elevation"
                elevations[0, :] = self.max_elevation - self.delta_elevation / 2
                elevations[1, :] = self.min_elevation + self.delta_elevation / 2

                longitude = grp.createVariable(
                    "longitude", "f8", ("vector", "position"), compression=DEFAULT_COMPRESSION_LIB
                )
                longitude.units = "degrees_east"
                longitude.long_name = "longitude"
                longitude.standard_name = "longitude"
                longitudes = self.col_head_lon[grid_idx::nb_grid]
                longitude[0, :] = longitudes[:]
                longitude[1, :] = longitudes[:]

                latitude = grp.createVariable(
                    "latitude", "f8", ("vector", "position"), compression=DEFAULT_COMPRESSION_LIB
                )
                latitude.units = "degrees_north"
                latitude.long_name = "latitude"
                latitude.standard_name = "latitude"
                latitudes = self.col_head_lat[grid_idx::nb_grid]
                latitude[0, :] = latitudes[:]
                latitude[1, :] = latitudes[:]

                if wc_constants.BACKSCATTER_MEAN in self.layers:
                    self.add_backscatter_variable(
                        grp=grp, layer_name=wc_constants.BACKSCATTER_MEAN, data=self.map_file_backscatter_mean[grid_idx]
                    )

                if wc_constants.BACKSCATTER_MAX in self.layers:
                    self.add_backscatter_variable(
                        grp=grp, layer_name=wc_constants.BACKSCATTER_MAX, data=self.map_file_backscatter_max[grid_idx]
                    )

                if wc_constants.BACKSCATTER_COMP_MEAN in self.layers:
                    self.add_backscatter_variable(
                        grp=grp,
                        layer_name=wc_constants.BACKSCATTER_COMP_MEAN,
                        data=self.map_file_backscatter_comp_mean[grid_idx],
                    )

                if wc_constants.BACKSCATTER_COMP_MAX in self.layers:
                    self.add_backscatter_variable(
                        grp=grp,
                        layer_name=wc_constants.BACKSCATTER_COMP_MAX,
                        data=self.map_file_backscatter_comp_max[grid_idx],
                    )

    @staticmethod
    def add_backscatter_variable(grp: Group, layer_name: str, data):
        backscatter = grp.createVariable(layer_name, "f4", ("height", "length"), compression=DEFAULT_COMPRESSION_LIB)
        backscatter.units = "dB"
        backscatter.long_name = layer_name
        backscatter.standard_name = layer_name
        backscatter[:] = data[::-1, :]

    def fill_grid(
        self,
        sound_lon: np.ndarray,
        sound_lat: np.ndarray,
        sound_elev: np.ndarray,
        sound_backscatter: np.ndarray,
        y_idx: np.ndarray,
        init_x_idx: int,
        compensated: bool,
    ):
        """
        Param :
           - sound_lon / sound_lat / sound_elev : position of the sounds
           - sound_backscatter : value of the sound
           - y_idx : lateral index of each sound
           - init_x_idx : approximative x index in grids
        """
        # Range of index of column where searching the nearest cell of each sound
        start_x_index = 0
        stop_x_index = self.x_count
        x_idx = np.zeros_like(sound_lon, dtype=int)

        _find_closest_ref_points(
            self.col_head_lon,
            self.col_head_lat,
            start_x_index,
            stop_x_index,
            init_x_idx,
            sound_lon,
            sound_lat,
            y_idx,
            x_idx,
        )
        z_idx = np.round((self.max_elevation - sound_elev) / self.delta_elevation - 0.5).astype(int)

        self._fill_grids_xyz(
            x_idxs=x_idx, y_idxs=y_idx, z_idxs=z_idx, backscatters=sound_backscatter, compensated=compensated
        )

    def _fill_grids_xyz(
        self, x_idxs: np.ndarray, y_idxs: np.ndarray, z_idxs: np.ndarray, backscatters: np.ndarray, compensated: bool
    ):
        """
        Remap spatial references point indices to grid reference point indices
        """
        if compensated:
            _fill_grids(
                grid_idxs=y_idxs,
                row_idxs=z_idxs,
                col_idxs=x_idxs,
                backscatters=backscatters,
                o_mean_array=self.map_file_backscatter_comp_mean,
                o_max_array=self.map_file_backscatter_comp_max,
                o_count_array=self.map_file_nb_values_comp,
            )
        else:
            _fill_grids(
                grid_idxs=y_idxs,
                row_idxs=z_idxs,
                col_idxs=x_idxs,
                backscatters=backscatters,
                o_mean_array=self.map_file_backscatter_mean,
                o_max_array=self.map_file_backscatter_max,
                o_count_array=self.map_file_nb_values,
            )


@nb.njit(cache=True, fastmath=True, parallel=True)
def _find_closest_ref_points(
    col_head_lon: np.ndarray,
    col_head_lat: np.ndarray,
    start_x_index: int,
    stop_x_index: int,
    init_x_index: int,
    sound_lon: np.ndarray,
    sound_lat: np.ndarray,
    y_idx: np.ndarray,
    o_x_idx: np.ndarray,
):
    """
    Function aiming to find the closest column head point for each sound (in sound_lon/sound_lat)
    Parameters :
     - col_head_lon / col_head_lat : column head positions
     - start_x_index / stop_x_index : range of x index where seaching the nearest column head
     - init_x_index : initial x index for optimal search
     - sound_lon / sound_lat : sound positions
     - y_idx: lateral index of each sound
    Resulting array is o_x_idx

    The algorithm search for a local minimum around init_x_index. First forward, then backward.
    """
    y_size = col_head_lon.shape[0]
    for i in nb.prange(len(sound_lon)):
        min_dist = np.inf
        o_x_idx[i] = -1
        # Sanity checks
        if y_idx[i] < 0 or y_idx[i] >= y_size:
            continue
        if np.isnan(sound_lon[i]) or np.isnan(sound_lat[i]):
            continue

        x_incr = 1
        x_idx = init_x_index
        stop = False
        output_x = o_x_idx[i]
        while stop is False:
            # check column range
            if x_idx < start_x_index or x_idx >= stop_x_index:
                stop = True
            else:
                dist = (col_head_lon[y_idx[i], x_idx] - sound_lon[i]) ** 2 + (
                    col_head_lat[y_idx[i], x_idx] - sound_lat[i]
                ) ** 2
                # check if a minimum is reached
                if dist < min_dist:
                    output_x = x_idx
                    min_dist = dist
                else:
                    stop = True
            # check stop condition
            if stop and x_incr == 1:
                x_incr = -1
                x_idx = init_x_index
                stop = False
            # next step
            x_idx += x_incr

        # end for loop
        o_x_idx[i] = output_x


@nb.njit(cache=True, fastmath=True, parallel=True)
def _find_closest_ref_points_dummy(
    col_head_lon: np.ndarray,
    col_head_lat: np.ndarray,
    start_x_index: int,
    stop_x_index: int,
    sound_lon: np.ndarray,
    sound_lat: np.ndarray,
    y_idx: np.ndarray,
    o_x_idx: np.ndarray,
):
    """
    Function aiming to find the closest column head point for each sound (in sound_lon/sound_lat)
    Parameters :
     - col_head_lon / col_head_lat : column head positions
     - start_x_index / stop_x_index : range of x index where seaching the nearest column head
     - sound_lon / sound_lat : sound positions
     - y_idx: lateral index of each sound
    Resulting array is o_x_idx
    """
    y_size = col_head_lon.shape[0]
    for i in nb.prange(len(sound_lon)):
        min_dist = np.inf
        o_x_idx[i] = -1
        # Sanity checks
        if y_idx[i] < 0 or y_idx[i] >= y_size:
            continue
        if np.isnan(sound_lon[i]) or np.isnan(sound_lat[i]):
            continue

        output_x = o_x_idx[i]
        for x_idx in np.arange(start_x_index, stop_x_index):
            dist = (col_head_lon[y_idx[i], x_idx] - sound_lon[i]) ** 2 + (
                col_head_lat[y_idx[i], x_idx] - sound_lat[i]
            ) ** 2
            # check if a minimum is reached
            if dist < min_dist:
                output_x = x_idx
                min_dist = dist

        # end for loop
        o_x_idx[i] = output_x


@nb.njit(cache=True, fastmath=True)
def _fill_grids(
    grid_idxs: np.ndarray,
    row_idxs: np.ndarray,
    col_idxs: np.ndarray,
    backscatters: np.ndarray,
    o_mean_array: np.ndarray,
    o_max_array: np.ndarray,
    o_count_array: np.ndarray,
):
    """
    Function aiming to find the closest grid reference point (index in ref_lons/ref_lats) for each sounder point (in longitudes/latitudes)
    """
    grid_max_idx, row_max_idx, col_max_idx = o_mean_array.shape
    for grid_idx, row_idx, col_idx, backscatter in zip(grid_idxs, row_idxs, col_idxs, backscatters):
        # Sanity checks
        if grid_idx < 0 or grid_idx >= grid_max_idx:
            continue
        if row_idx < 0 or row_idx >= row_max_idx:
            continue
        if col_idx < 0 or col_idx >= col_max_idx:
            continue

        prev_count = o_count_array[grid_idx][row_idx][col_idx]
        prev_mean = o_mean_array[grid_idx][row_idx][col_idx]

        o_mean_array[grid_idx][row_idx][col_idx] = (prev_count * prev_mean + backscatter) / (prev_count + 1)
        o_count_array[grid_idx][row_idx][col_idx] += 1
        o_max_array[grid_idx][row_idx][col_idx] = max(backscatter, o_max_array[grid_idx][row_idx][col_idx])
