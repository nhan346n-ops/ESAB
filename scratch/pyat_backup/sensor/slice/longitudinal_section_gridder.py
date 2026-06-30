# pylint:disable=no-member
import logging
from dataclasses import dataclass
from typing import Dict, Generator, List

import numba as nb
import numpy as np
from netCDF4 import Group
from pygws.service.progress_monitor import ProgressMonitor

from pyat.sensor.slice.g3d_writer import Coords, Variables, write_vertical_textures
from pyat.sensor.slice.interpolation import interpolate
from pyat.utils.coords import haversine_distance
from pyat.utils.nc_encoding import open_nc_file
from pyat.utils.netcdf_utils import DEFAULT_COMPRESSION_LIB

logger = logging.getLogger(__name__)


class LongitudinalSectionGridder:
    def __init__(
        self,
        x_count: int,
        z_count: int,
        min_elevation: float,
        max_elevation: float,
        delta_elevation: float,
        layers: List[str],
    ):
        """
        Gridder for computing vertical images under a georeferenced path.

        Args:
            x_count: number of horizontal elements along the path
            z_count: number of vertical elements
            min_elevation: minimum elevation value
            max_elevation: maximum elevation value
            delta_elevation: vertical cell size
            layers: list of layer names to compute
        """
        self.x_count = x_count
        self.z_count = z_count
        self.max_elevation = max_elevation
        self.min_elevation = min_elevation
        self.delta_elevation = delta_elevation
        self.layers = layers

        # Position of horizontal cells along the path
        self.col_head_lon = np.full(x_count, np.nan)
        self.col_head_lat = np.full(x_count, np.nan)

        # Grid data for each layer (in-memory arrays)
        self.grid_values: Dict[str, np.ndarray] = {}
        self.grid_nb_values: Dict[str, np.ndarray] = {}

        self.row_count = z_count
        self.col_count = x_count

    def initialize_grid(self):
        """Initialize the grid arrays in memory."""
        logger.info(f"Preparing {len(self.layers)} grids, size {self.col_count}x{self.row_count} cells each")

        for layer in self.layers:
            # Create in-memory array for value counts
            self.grid_nb_values[layer] = np.zeros((self.row_count, self.col_count), dtype=int)

            # Create in-memory array for mean values
            self.grid_values[layer] = np.zeros((self.row_count, self.col_count), dtype=np.float32)

    def add(self, lon: float, lat: float, x_idx: int) -> None:
        """
        Add a reference point along the path.

        Args:
            lon: longitude of the reference point
            lat: latitude of the reference point
            x_idx: horizontal index along the path
        """
        self.col_head_lon[x_idx] = lon
        self.col_head_lat[x_idx] = lat

    def finalize(
        self,
        monitor: ProgressMonitor,
        interpolation_method: str | None = None,
    ):
        """
        Finalize the grids by setting NaN where no data and optionally interpolating.

        Args:
            monitor: progress monitor for tracking completion
            interpolate_missing: whether to interpolate missing values
            interpolation_method: interpolation method to use
        """
        monitor.begin_task("Finalizing grids", len(self.layers) * 2)

        for layer in self.layers:
            # Set NaN where no values were accumulated
            self.grid_values[layer][self.grid_nb_values[layer] == 0] = np.nan
            monitor.worked(1)

            # Optional interpolation
            if interpolation_method:
                logger.info(f"Interpolating layer {layer} using method {interpolation_method}")
                self.grid_values[f"{layer}_i"] = interpolate(grid=self.grid_values[layer], method=interpolation_method)
            monitor.worked(1)

        monitor.done()

    def generate_g3d_file(self, path_g3d: str):
        """
        Generate a G3D NetCDF file with the grid data.

        Args:
            path_g3d: path to the output G3D file
        """
        coords = Coords(
            latitudes=self.col_head_lat[:],
            longitudes=self.col_head_lon[:],
            min_elevation=self.max_elevation - self.delta_elevation / 2,
            max_elevation=self.min_elevation + self.delta_elevation / 2,
        )
        variables = Variables(self.layers)
        for layer, values in self.grid_values.items():
            variables[layer] = values

        write_vertical_textures(coords, variables, path_g3d)

    @staticmethod
    def add_layer_variable(grp: Group, layer_name: str, data: np.ndarray):
        """
        Add a layer variable to the NetCDF group.

        Args:
            grp: NetCDF group
            layer_name: name of the layer
            data: data array for the layer
        """
        layer_var = grp.createVariable(layer_name, "f4", ("height", "length"), compression=DEFAULT_COMPRESSION_LIB)
        layer_var.long_name = layer_name
        layer_var.standard_name = layer_name
        # Reverse elevation axis (from top to bottom)
        layer_var[:] = data[::-1, :]

    def fill_grid(
        self,
        layer: str,
        longitudes: np.ndarray,
        latitudes: np.ndarray,
        elevations: np.ndarray,
        values: np.ndarray,
        max_distance: float,
    ) -> bool:
        """
        Fill the grid with measurements for a specific layer.

        Args:
            layer: layer name to fill
            longitudes: longitude of each sensor data
            latitudes: latitude of each sensor data
            elevations: elevation of each sensor data
            values: value of each sensor data
            max_distance: maximum distance to consider a match
        """
        if layer not in self.layers:
            raise ValueError(f"Layer {layer} not in configured layers: {self.layers}")

        # Find the closest horizontal position for each sensor data
        x_idx = np.full_like(longitudes, dtype=int, fill_value=-1)

        _find_closest_ref_points(
            reference_longitudes=self.col_head_lon,
            reference_latitudes=self.col_head_lat,
            longitudes=longitudes,
            latitudes=latitudes,
            o_x_idx=x_idx,
            max_distance=max_distance,
        )
        if np.all(x_idx == -1):
            return False  # No valid points found

        # Calculate vertical index
        z_idx = np.round((self.max_elevation - elevations) / self.delta_elevation - 0.5).astype(int)

        # Fill the grid for the specified layer
        self._fill_grid_xz(x_idxs=x_idx, z_idxs=z_idx, values=values, layer=layer)

        return True

    def _fill_grid_xz(self, x_idxs: np.ndarray, z_idxs: np.ndarray, values: np.ndarray, layer: str):
        """
        Fill the grid with values at specified indices for a specific layer.

        Args:
            x_idxs: horizontal indices
            z_idxs: vertical indices
            values: values to fill
            layer: layer name to fill
        """
        _fill_grid(
            row_idxs=z_idxs,
            col_idxs=x_idxs,
            values=values,
            o_mean_array=self.grid_values[layer],
            o_count_array=self.grid_nb_values[layer],
        )


@nb.njit(cache=True, fastmath=True, parallel=True)
def _find_closest_ref_points(
    reference_longitudes: np.ndarray,
    reference_latitudes: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    o_x_idx: np.ndarray,
    max_distance: float,
):
    """
    Find the closest column head point for each measurement.

    The algorithm searches for a local minimum around init_x_index,
    first forward, then backward.

    Args:
        reference_longitudes: longitude of column head positions
        reference_latitudes: latitude of column head positions
        longitudes: longitude of measurements
        latitudes: latitude of measurements
        o_x_idx: output array for x indices
        max_distance: maximum distance to consider a match
    """
    for i in nb.prange(len(longitudes)):
        min_dist = np.inf
        o_x_idx[i] = -1

        # Sanity checks
        if np.isnan(longitudes[i]) or np.isnan(latitudes[i]):
            continue

        for x_idx, ref_lon in enumerate(reference_longitudes):
            dist = haversine_distance(
                lat1=reference_latitudes[x_idx], lon1=ref_lon, lat2=latitudes[i], lon2=longitudes[i]
            )
            # Check if a minimum is reached
            if dist < max_distance and dist < min_dist:
                o_x_idx[i] = x_idx
                min_dist = dist


@nb.njit(cache=True, fastmath=True)
def _fill_grid(
    row_idxs: np.ndarray,
    col_idxs: np.ndarray,
    values: np.ndarray,
    o_mean_array: np.ndarray,
    o_count_array: np.ndarray,
):
    """
    Fill the grid with values, computing running mean.

    Args:
        row_idxs: vertical indices
        col_idxs: horizontal indices
        values: values to fill
        o_mean_array: output array for mean values
        o_count_array: output array for value counts
    """
    row_max_idx, col_max_idx = o_mean_array.shape

    for row_idx, col_idx, value in zip(row_idxs, col_idxs, values):
        # Sanity checks
        if row_idx < 0 or row_idx >= row_max_idx:
            continue
        if col_idx < 0 or col_idx >= col_max_idx:
            continue
        if np.isnan(value):
            continue

        prev_count = o_count_array[row_idx][col_idx]
        prev_mean = o_mean_array[row_idx][col_idx]

        # Update running mean
        o_mean_array[row_idx][col_idx] = (prev_count * prev_mean + value) / (prev_count + 1)
        o_count_array[row_idx][col_idx] += 1
