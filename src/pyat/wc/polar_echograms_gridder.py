# pylint:disable=no-member
from typing import List

import cv2
import numba as nb
import numpy as np
from netCDF4 import Group

from pyat.utils.netcdf import get_default_fillvalue
import pyat.utils.pyat_logger as log
import pyat.wc.utils.bilinear_gap_filling_functions as gff
from pyat.utils.nc_encoding import open_nc_file
from pyat.utils.netcdf_utils import DEFAULT_COMPRESSION_LIB
from pyat.utils.signal import energy_to_db
from pyat.wc import wc_constants


class PolarEchogramsGridder:
    def __init__(self, x_count: int, y_count: int, z_count: int, layers: List[str], output_path: str):
        """
        Args:
            x_count: number of longitudinal elements
            y_count: number of lateral elements
            z_count: number of vertical elements
        """
        self.x_count = x_count
        self.y_count = y_count
        self.z_count = z_count
        self.layers = layers
        self.output_path = output_path
        self.dataset = None

        # Position of echograms corners
        self.top_left_lon = np.nan
        self.top_left_lat = np.nan
        self.top_left_elev = np.nan
        self.top_right_lon = np.nan
        self.top_right_lat = np.nan
        self.top_right_elev = np.nan
        self.bottom_left_lon = np.nan
        self.bottom_left_lat = np.nan
        self.bottom_left_elev = np.nan
        self.bottom_right_lon = np.nan
        self.bottom_right_lat = np.nan
        self.bottom_right_elev = np.nan

        self.logger = log.logging.getLogger(self.__class__.__name__)

        # raw data
        self.buffer_nb_values = None
        self.buffer_backscatter_mean = None
        self.buffer_backscatter_max = None

        # comp data
        self.buffer_nb_values_comp = None
        self.buffer_backscatter_comp_mean = None
        self.buffer_backscatter_comp_max = None

        # buffer size
        self.grid_count = x_count
        self.row_count = z_count
        self.col_count = y_count
        self.border_count = 0

    # pylint: disable=consider-using-with
    def initialize_grid(self):
        self.mask = np.full(shape=(self.row_count, self.col_count), dtype=np.uint8, fill_value=0)
        self.inmask = np.full(shape=(self.row_count, self.col_count), dtype=np.uint8, fill_value=0)
        self.temp_buffer = np.full(shape=(self.row_count, self.col_count), dtype=np.float32, fill_value=np.nan)
        if wc_constants.contains_raw_layer(self.layers):
            # Optimize access to data with numpy array
            self.buffer_nb_values = np.full(shape=(self.row_count, self.col_count), dtype=np.int32, fill_value=0)

            self.buffer_backscatter_mean = np.full(
                shape=(self.row_count, self.col_count), dtype=np.float32, fill_value=0
            )
            self.buffer_backscatter_max = np.full(
                shape=(self.row_count, self.col_count), dtype=np.float32, fill_value=-np.inf
            )

        if wc_constants.contains_compensated_layer(self.layers):
            # Optimize access to data with numpy array
            self.buffer_nb_values_comp = np.full(shape=(self.row_count, self.col_count), dtype=np.int32, fill_value=0)
            self.buffer_backscatter_comp_mean = np.full(
                shape=(self.row_count, self.col_count), dtype=np.float32, fill_value=0
            )
            self.buffer_backscatter_comp_max = np.full(
                shape=(self.row_count, self.col_count), dtype=np.float32, fill_value=-np.inf
            )

    def reset_grid(self):
        # buffer size
        self.y_count = 0
        self.z_count = 0
        self.border_count = 0

        if self.buffer_nb_values is not None:
            self.buffer_nb_values.fill(0)

        if self.buffer_backscatter_mean is not None:
            self.buffer_backscatter_mean.fill(0)

        if self.buffer_backscatter_max is not None:
            self.buffer_backscatter_max.fill(-np.inf)

        if self.buffer_nb_values_comp is not None:
            self.buffer_nb_values_comp.fill(0)

        if self.buffer_backscatter_comp_mean is not None:
            self.buffer_backscatter_comp_mean.fill(0)

        if self.buffer_backscatter_comp_max is not None:
            self.buffer_backscatter_comp_max.fill(-np.inf)

    def set_size(self, y_count: int, z_count: int) -> None:
        """
        num of col and rows of current grid
        Args:
            y_count: num of columns
            z_count: num of rows
        """

        if y_count + 2 * self.border_count > self.col_count or z_count + 2 * self.border_count > self.row_count:
            self.col_count = y_count + 2 * self.border_count
            self.row_count = z_count + 2 * self.border_count
            self.initialize_grid()

        self.y_count = y_count
        self.z_count = z_count

    def set_top_left(self, lon: float, lat: float, elevation: float) -> None:
        """
        This point is a corner of the grid.
        Args:
            lon: longitude of the corner
            lat: latitude of the corner
            elevation : elevation of the corner
        """
        self.top_left_lon = lon
        self.top_left_lat = lat
        self.top_left_elev = elevation

    def set_top_right(self, lon: float, lat: float, elevation: float) -> None:
        """
        This point is a corner of the grid.
        Args:
            lon: longitude of the corner
            lat: latitude of the corner
            elevation : elevation of the corner
        """
        self.top_right_lon = lon
        self.top_right_lat = lat
        self.top_right_elev = elevation

    def set_bottom_left(self, lon: float, lat: float, elevation: float) -> None:
        """
        This point is a corner of the grid.
        Args:
            lon: longitude of the corner
            lat: latitude of the corner
            elevation : elevation of the corner
        """
        self.bottom_left_lon = lon
        self.bottom_left_lat = lat
        self.bottom_left_elev = elevation

    def set_bottom_right(self, lon: float, lat: float, elevation: float) -> None:
        """
        This point is a corner of the grid.
        Args:
            lon: longitude of the corner
            lat: latitude of the corner
            elevation : elevation of the corner
        """
        self.bottom_right_lon = lon
        self.bottom_right_lat = lat
        self.bottom_right_elev = elevation

    def set_interpolate_limit(self, limit: int) -> None:
        """
        interpolation limit for fill gap algorithm.
        Args:
            limit: max number of empty pixels to fill between valid data
        """
        self.border_count = max(limit, 0)

    def set_min_max_across(self, min_across: float, max_across: float) -> None:
        """
        min/max across of the grid.
        Args:
            min: min across distance
            max: max across distance
        """
        self.min_across = min_across
        self.max_across = max_across

    def set_time(self, time: str) -> None:
        """
        ping time of the grid.
        Args:
            time: time representation as string
        """
        self.time = time

    def values_count(self):
        count = 0
        if self.buffer_nb_values is not None:
            count = np.nansum(self.buffer_nb_values)
        if self.buffer_nb_values_comp is not None:
            count = np.nansum(self.buffer_nb_values_comp, initial=count)
        return count

    def finalize(self, interpolate: bool = False):
        if self.y_count == 0 or self.z_count == 0:
            return

        # prepare mask
        if wc_constants.BACKSCATTER_MEAN in self.layers:
            self.finalize_singlelayer(
                value_buffer=self.buffer_backscatter_mean,
                value_count=self.buffer_nb_values,
                interpolate=interpolate,
            )

        if wc_constants.BACKSCATTER_MAX in self.layers:
            self.finalize_singlelayer(
                value_buffer=self.buffer_backscatter_max,
                value_count=self.buffer_nb_values,
                interpolate=interpolate,
            )

        if wc_constants.BACKSCATTER_COMP_MEAN in self.layers:
            self.finalize_singlelayer(
                value_buffer=self.buffer_backscatter_comp_mean,
                value_count=self.buffer_nb_values_comp,
                interpolate=interpolate,
            )

        if wc_constants.BACKSCATTER_COMP_MAX in self.layers:
            self.finalize_singlelayer(
                value_buffer=self.buffer_backscatter_comp_max,
                value_count=self.buffer_nb_values_comp,
                interpolate=interpolate,
            )

    def finalize_singlelayer(self, value_buffer: np.ndarray, value_count: np.ndarray, interpolate: bool):
        value_buffer[value_count == 0] = np.nan
        # # post interpolation :
        if interpolate:
            self.interpolate(
                in_buffer=value_buffer,
                in_count=value_count,
                out_buffer=self.temp_buffer,
            )
        else:
            self.temp_buffer[:] = value_buffer[:]
        # return values in db
        # reflectivity mean
        energy_to_db(
            value=self.temp_buffer,
            out=value_buffer,
        )

    def interpolate(self, in_buffer: np.ndarray, in_count: np.ndarray, out_buffer: np.ndarray):
        # reset output buffer
        out_buffer.fill(np.nan)
        # prepare mask with valid data
        self.inmask[:] = np.where(in_count == 0, 0, 1)
        # apply closing to expand mask to cells to be filled
        cv2.morphologyEx(
            src=self.inmask,
            op=cv2.MORPH_CLOSE,
            dst=self.mask,
            kernel=np.ones((self.border_count, self.border_count)),
            iterations=1,
        )
        # apply interpolation on masked data
        self._interpolate(
            out_buffer=out_buffer,
            in_buffer=in_buffer,
            limit=self.border_count,
            mask=self.mask,
        )

    @staticmethod
    def _interpolate(out_buffer: np.ndarray, in_buffer: np.ndarray, limit: int, mask: np.ndarray):
        """
        Final interpolation specific step before writing output g3d file
        """
        # In function of the size of the mask, create matrix distance.
        index = gff.find_distance(limit)
        # Then transform it into coordinates.
        coord = gff.find_coord(index)
        # Elevation interpolation
        out_buffer = gff.interpolation(out_buffer, in_buffer, coord, limit, mask)

    def initialize_g3d_file(self):
        with open_nc_file(self.output_path, mode="w", nc_format="NETCDF4") as dataset:
            dataset.dataset_type = "FlyTexture"
            dataset.history = "Created by PyAT with PolarEchograms"

            dataset.createDimension("datalayer_count", len(self.layers))
            datalayer_variable_name = dataset.createVariable("datalayer_variable_name", str, ("datalayer_count",))
            for index, layer in enumerate(self.layers):
                datalayer_variable_name[index] = layer

    def add_g3d_grid(self, grid_idx: int):
        if self.z_count == 0 or self.y_count == 0:
            return
        height = self.z_count
        length = self.y_count
        vector = position = 2
        grpname = f"{grid_idx + 1}".zfill(3)

        if self.dataset is None:
            self.dataset = open_nc_file(self.output_path, mode="a", nc_format="NETCDF4")
        grp = self.dataset.createGroup(grpname)
        grp.createDimension("height", height)
        grp.createDimension("length", length)
        grp.createDimension("vector", vector)
        grp.createDimension("position", position)
        # additional attributes
        grp.long_name = f"Ping {grpname}"
        grp.time = self.time
        grp.across_dist_L = self.min_across
        grp.across_dist_R = self.max_across

        elevations = grp.createVariable("elevation", "f4", ("vector", "position"), compression=DEFAULT_COMPRESSION_LIB)
        elevations.units = "meters"
        elevations.long_name = "elevation"
        elevations.standard_name = "elevation"
        elevations[0, 0] = self.top_left_elev
        elevations[0, 1] = self.top_right_elev
        elevations[1, 0] = self.bottom_left_elev
        elevations[1, 1] = self.bottom_right_elev

        longitude = grp.createVariable("longitude", "f8", ("vector", "position"), compression=DEFAULT_COMPRESSION_LIB)
        longitude.units = "degrees_east"
        longitude.long_name = "longitude"
        longitude.standard_name = "longitude"
        longitude[0, 0] = self.top_left_lon
        longitude[0, 1] = self.top_right_lon
        longitude[1, 0] = self.bottom_left_lon
        longitude[1, 1] = self.bottom_right_lon

        latitude = grp.createVariable("latitude", "f8", ("vector", "position"), compression=DEFAULT_COMPRESSION_LIB)
        latitude.units = "degrees_north"
        latitude.long_name = "latitude"
        latitude.standard_name = "latitude"
        latitude[0, 0] = self.top_left_lat
        latitude[0, 1] = self.top_right_lat
        latitude[1, 0] = self.bottom_left_lat
        latitude[1, 1] = self.bottom_right_lat

        buffer_slice = (
            slice(self.border_count, self.z_count + self.border_count),
            slice(self.border_count, self.y_count + self.border_count),
        )
        if wc_constants.BACKSCATTER_MEAN in self.layers:
            self.add_backscatter_variable(
                grp=grp,
                layer_name=wc_constants.BACKSCATTER_MEAN,
                data=self.buffer_backscatter_mean[buffer_slice],
            )

        if wc_constants.BACKSCATTER_MAX in self.layers:
            self.add_backscatter_variable(
                grp=grp,
                layer_name=wc_constants.BACKSCATTER_MAX,
                data=self.buffer_backscatter_max[buffer_slice],
            )

        if wc_constants.BACKSCATTER_COMP_MEAN in self.layers:
            self.add_backscatter_variable(
                grp=grp,
                layer_name=wc_constants.BACKSCATTER_COMP_MEAN,
                data=self.buffer_backscatter_comp_mean[buffer_slice],
            )

        if wc_constants.BACKSCATTER_COMP_MAX in self.layers:
            self.add_backscatter_variable(
                grp=grp,
                layer_name=wc_constants.BACKSCATTER_COMP_MAX,
                data=self.buffer_backscatter_comp_max[buffer_slice],
            )

    def flush_dataset(self):
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None

    @staticmethod
    def add_backscatter_variable(grp: Group, layer_name: str, data):
        backscatter = grp.createVariable(
            layer_name,
            "i2",
            ("height", "length"),
            compression=DEFAULT_COMPRESSION_LIB,
            fill_value=get_default_fillvalue(np.int16),
        )
        backscatter.scale_factor = 0.1
        backscatter.units = "dB"
        backscatter.long_name = layer_name
        backscatter.standard_name = layer_name
        # mask data with nan values in data array, to keep them as fill_value in output file
        data = np.ma.masked_array(data, mask=np.isnan(data))
        backscatter[:] = data[::-1, :]

    def fill_grid(self, sound_backscatter: np.ndarray, z_idx: np.ndarray, y_idx: np.ndarray, compensated: bool):
        """
        Param :
           - sound_lon / sound_lat / sound_elev : position of the sounds
           - sound_backscatter : value of the sound
           - y_idx : lateral index of each sound
        """

        self._fill_grid_xyz(y_idxs=y_idx, z_idxs=z_idx, backscatters=sound_backscatter, compensated=compensated)

    def _fill_grid_xyz(self, y_idxs: np.ndarray, z_idxs: np.ndarray, backscatters: np.ndarray, compensated: bool):
        """
        Remap spatial references point indices to grid reference point indices
        """
        if compensated:
            _fill_grid(
                row_idxs=z_idxs + self.border_count,
                col_idxs=y_idxs + self.border_count,
                backscatters=backscatters,
                o_mean_array=self.buffer_backscatter_comp_mean,
                o_max_array=self.buffer_backscatter_comp_max,
                o_count_array=self.buffer_nb_values_comp,
            )
        else:
            _fill_grid(
                row_idxs=z_idxs + self.border_count,
                col_idxs=y_idxs + self.border_count,
                backscatters=backscatters,
                o_mean_array=self.buffer_backscatter_mean,
                o_max_array=self.buffer_backscatter_max,
                o_count_array=self.buffer_nb_values,
            )


@nb.njit(cache=True, fastmath=True)
def _fill_grid(
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
    row_max_idx, col_max_idx = o_mean_array.shape
    for row_idx, col_idx, backscatter in zip(row_idxs, col_idxs, backscatters):
        # Sanity checks
        if 0 <= row_idx < row_max_idx and 0 <= col_idx < col_max_idx:
            prev_count = o_count_array[row_idx][col_idx]
            prev_mean = o_mean_array[row_idx][col_idx]

            o_mean_array[row_idx][col_idx] = (prev_count * prev_mean + backscatter) / (prev_count + 1)
            o_count_array[row_idx][col_idx] += 1
            o_max_array[row_idx][col_idx] = max(backscatter, o_max_array[row_idx][col_idx])
