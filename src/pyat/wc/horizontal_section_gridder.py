# pylint:disable=no-member
import math
import tempfile
from typing import List

import numba as nb
import numpy as np
from scipy.interpolate import griddata

import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.utils.pyat_logger as log
import pyat.wc.wc_constants as wc_constants
from pyat.dtm.dtm_driver import DtmDriver
from pyat.utils import numpy_utils
from pyat.utils.argument_utils import Geobox
from pyat.utils.signal import energy_to_db


class HorizontalSectionGridder:
    def __init__(
        self,
        geobox: Geobox,
        spatial_resolution: float,
        min_elevation: float,
        max_elevation: float,
        delta_elevation: float,
        layers: List[str],
    ):
        """
        Args:
        """
        self.geobox = geobox
        self.spatial_resolution = spatial_resolution
        self.max_elevation = max_elevation
        self.min_elevation = min_elevation
        self.delta_elevation = delta_elevation
        self.layers = layers

        # compute col_count
        self.col_count = int(math.ceil(self.geobox.get_delta_x() / self.spatial_resolution))
        # compute row_count
        self.row_count = int(math.ceil(self.geobox.get_delta_y() / self.spatial_resolution))
        # compute grid_count
        self.grid_count = int(math.ceil((self.max_elevation - self.min_elevation) / self.delta_elevation))

        # compute ref longitudes
        self.longitudes = np.arange(self.geobox.left, self.geobox.right, self.spatial_resolution)
        self.longitudes += self.spatial_resolution / 2

        # compute ref latitudes
        self.latitudes = np.arange(self.geobox.lower, self.geobox.upper, self.spatial_resolution)
        self.latitudes += self.spatial_resolution / 2

        self.logger = log.logging.getLogger(self.__class__.__name__)

        # raw data
        self.map_file_nb_values = None
        self.map_file_backscatter_mean = None
        self.map_file_backscatter_max = None

        # comp data
        self.map_file_nb_values_comp = None
        self.map_file_backscatter_comp_mean = None
        self.map_file_backscatter_comp_max = None

        # elevation data
        self.map_file_elevation = None
        self.map_file_elevation_count = None

    # pylint: disable=consider-using-with
    def initialize_grid(self):
        self.logger.info(f"Preparing {self.grid_count} grids, size {self.row_count}x{self.col_count} cells each")

        if wc_constants.contains_elevation_layer(self.layers):
            # Optimize access to data with numpy array
            temp_map_file_elevation = tempfile.NamedTemporaryFile(suffix=wc_constants.ELEVATION + ".memmap")
            self.map_file_elevation = np.memmap(
                temp_map_file_elevation,
                shape=(self.row_count, self.col_count),
                dtype=np.float32,
                mode="w+",
            )
            self.map_file_elevation.fill(np.nan)
            # count is mandatory to compute elevation mean
            temp_map_file_elevation_count = tempfile.NamedTemporaryFile(suffix=wc_constants.ELEVATION_COUNT + ".memmap")
            self.map_file_elevation_count = np.memmap(
                temp_map_file_elevation_count,
                shape=(self.row_count, self.col_count),
                dtype=int,
                mode="w+",
            )
            self.map_file_elevation_count.fill(0)

        if wc_constants.contains_raw_layer(self.layers):
            # Optimize access to data with numpy array
            temp_map_file_nb_value = tempfile.NamedTemporaryFile(
                suffix=wc_constants.BACKSCATTER_VALUE_COUNT + ".memmap"
            )
            self.map_file_nb_values = np.memmap(
                temp_map_file_nb_value,
                shape=(self.grid_count, self.row_count, self.col_count),
                dtype=int,
                mode="w+",
            )
            self.map_file_nb_values.fill(0)

            temp_map_file_echo_mean = tempfile.NamedTemporaryFile(suffix=wc_constants.BACKSCATTER_MEAN + ".memmap")
            self.map_file_backscatter_mean = np.memmap(
                temp_map_file_echo_mean,
                shape=(self.grid_count, self.row_count, self.col_count),
                dtype=np.float32,
                mode="w+",
            )
            self.map_file_backscatter_mean.fill(0)

            # Optimize access to data with numpy array
            temp_map_file_echo_max = tempfile.NamedTemporaryFile(suffix=wc_constants.BACKSCATTER_MAX + ".memmap")
            self.map_file_backscatter_max = np.memmap(
                temp_map_file_echo_max,
                shape=(self.grid_count, self.row_count, self.col_count),
                dtype=np.float32,
                mode="w+",
            )
            self.map_file_backscatter_max.fill(-np.inf)

        if wc_constants.contains_compensated_layer(self.layers):
            temp_map_file_nb_value_comp = tempfile.NamedTemporaryFile(
                suffix=wc_constants.BACKSCATTER_COMP_VALUE_COUNT + "memmap"
            )
            self.map_file_nb_values_comp = np.memmap(
                temp_map_file_nb_value_comp,
                shape=(self.grid_count, self.row_count, self.col_count),
                dtype=int,
                mode="w+",
            )
            self.map_file_nb_values_comp.fill(0)

            # Optimize access to data with numpy array
            temp_map_file_echo_comp_mean = tempfile.NamedTemporaryFile(
                suffix=wc_constants.BACKSCATTER_COMP_MEAN + ".memmap"
            )
            self.map_file_backscatter_comp_mean = np.memmap(
                temp_map_file_echo_comp_mean,
                shape=(self.grid_count, self.row_count, self.col_count),
                dtype=np.float32,
                mode="w+",
            )
            self.map_file_backscatter_comp_mean.fill(0)

            # Optimize access to data with numpy array
            temp_map_file_echo_comp_max = tempfile.NamedTemporaryFile(
                suffix=wc_constants.BACKSCATTER_COMP_MAX + ".memmap"
            )
            self.map_file_backscatter_comp_max = np.memmap(
                temp_map_file_echo_comp_max,
                shape=(self.grid_count, self.row_count, self.col_count),
                dtype=np.float32,
                mode="w+",
            )
            self.map_file_backscatter_comp_max.fill(-np.inf)

    def values_count(self):
        count = 0
        if self.map_file_nb_values is not None:
            count = np.nansum(self.map_file_nb_values)
        if self.map_file_nb_values_comp is not None:
            count = np.nansum(self.map_file_nb_values_comp, initial=count)
        return count

    def finalize(self):
        if wc_constants.BACKSCATTER_MEAN in self.layers:
            self.map_file_backscatter_mean[self.map_file_nb_values == 0] = np.nan
            # return values from energy to db
            # reflectivity mean
            energy_to_db(value=self.map_file_backscatter_mean, out=self.map_file_backscatter_mean)

        if wc_constants.BACKSCATTER_MAX in self.layers:
            self.map_file_backscatter_max[self.map_file_nb_values == 0] = np.nan
            # reflectivity max
            energy_to_db(value=self.map_file_backscatter_max, out=self.map_file_backscatter_max)

        if wc_constants.BACKSCATTER_COMP_MEAN in self.layers:
            self.map_file_backscatter_comp_mean[self.map_file_nb_values_comp == 0] = np.nan
            # comp mean
            energy_to_db(value=self.map_file_backscatter_comp_mean, out=self.map_file_backscatter_comp_mean)

        if wc_constants.BACKSCATTER_COMP_MAX in self.layers:
            self.map_file_backscatter_comp_max[self.map_file_nb_values_comp == 0] = np.nan
            # comp max
            energy_to_db(value=self.map_file_backscatter_comp_max, out=self.map_file_backscatter_comp_max)

    def generate_g3d_file(self, path_g3d: str):
        # DtmDriver is used here because g3d ElevationMappedTexture has the same structure as a DTM.
        i_driver = DtmDriver(path_g3d)

        metadata = {
            "title": "",
            "institution": "",
            "source": "",
            "references": "",
            "comment": "",
            "history": "Created by PyAT with HorizontalSection",
        }

        with i_driver.create_file(
            col_count=self.col_count,
            origin_x=self.longitudes[0],
            spatial_resolution_x=self.spatial_resolution,
            row_count=self.row_count,
            origin_y=self.latitudes[0],
            spatial_resolution_y=self.spatial_resolution,
            overwrite=True,
            metadata=metadata,
        ) as dataset:
            dataset.dataset_type = "ElevationMappedTexture"
            dataset.max_elevation = self.max_elevation
            dataset.min_elevation = self.min_elevation
            dataset.delta_elevation = self.delta_elevation
            dataset.slice_count = self.grid_count

            datalayers = self.layers.copy()
            if wc_constants.ELEVATION in self.layers:
                datalayers.remove(wc_constants.ELEVATION)

            dataset.createDimension("datalayer_count", len(datalayers))
            datalayer_variable_name = dataset.createVariable("datalayer_variable_name", str, ("datalayer_count",))
            for index, layer in enumerate(datalayers):
                datalayer_variable_name[index] = layer

            if wc_constants.ELEVATION in self.layers:
                self.add_elevation_variable(
                    i_driver=i_driver, name=wc_constants.ELEVATION, data=self.map_file_elevation
                )

            if wc_constants.BACKSCATTER_MEAN in self.layers:
                for grid_idx in range(self.grid_count):
                    self.add_backscatter_variable(
                        i_driver=i_driver,
                        base_name=wc_constants.BACKSCATTER_MEAN,
                        grid_idx=grid_idx,
                        data=self.map_file_backscatter_mean[grid_idx],
                    )

            if wc_constants.BACKSCATTER_MAX in self.layers:
                for grid_idx in range(self.grid_count):
                    self.add_backscatter_variable(
                        i_driver=i_driver,
                        base_name=wc_constants.BACKSCATTER_MAX,
                        grid_idx=grid_idx,
                        data=self.map_file_backscatter_max[grid_idx],
                    )

            if wc_constants.BACKSCATTER_COMP_MEAN in self.layers:
                for grid_idx in range(self.grid_count):
                    self.add_backscatter_variable(
                        i_driver=i_driver,
                        base_name=wc_constants.BACKSCATTER_COMP_MEAN,
                        grid_idx=grid_idx,
                        data=self.map_file_backscatter_comp_mean[grid_idx],
                    )

            if wc_constants.BACKSCATTER_COMP_MAX in self.layers:
                for grid_idx in range(self.grid_count):
                    self.add_backscatter_variable(
                        i_driver=i_driver,
                        base_name=wc_constants.BACKSCATTER_COMP_MAX,
                        grid_idx=grid_idx,
                        data=self.map_file_backscatter_comp_max[grid_idx],
                    )
        i_driver.close()

    def add_backscatter_variable(self, i_driver: DtmDriver, base_name: str, grid_idx: int, data):
        slice_prefix = f"{grid_idx + 1}".zfill(3) + "_"
        if wc_constants.ELEVATION in self.layers:
            vertical_offset = self.min_elevation + (grid_idx + 0.5) * self.delta_elevation
        else:
            vertical_offset = self.max_elevation - (grid_idx + 0.5) * self.delta_elevation
        slice_suffix = f"_(z={np.round(vertical_offset, 2)})"

        backscatter = i_driver.add_variable(
            slice_prefix + base_name + slice_suffix,
            datatype=float,
            dimensions=(DtmConstants.DIM_LAT, DtmConstants.DIM_LON),
            fill_value=float("nan"),
        )
        backscatter.units = "dB"
        backscatter.long_name = base_name
        backscatter.standard_name = base_name
        backscatter.grid_mapping = DtmConstants.CRS_NAME
        backscatter.vertical_offset = vertical_offset
        backscatter[:] = data[:]

    def add_elevation_variable(self, i_driver: DtmDriver, name: str, data):
        elevation = i_driver.add_variable(
            varname=name,
            datatype=float,
            dimensions=(DtmConstants.DIM_LAT, DtmConstants.DIM_LON),
            fill_value=float("nan"),
        )
        elevation.units = "meter"
        elevation.long_name = name
        elevation.standard_name = name
        elevation.grid_mapping = DtmConstants.CRS_NAME
        elevation[:] = data[:]

    def fill_elevations(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        elev: np.ndarray,
    ):
        """
        Param :
           - lon / lat / elev : position of the detection
        """
        # compute longitude indices
        x_idx = np.round((lon - self.geobox.left) / self.spatial_resolution - 0.5).astype(int)
        # compute latitude indices
        y_idx = np.round((lat - self.geobox.lower) / self.spatial_resolution - 0.5).astype(int)
        numpy_utils.compute_statistics(
            in_array=elev,
            x_array=x_idx,
            y_array=y_idx,
            out_mean_array=self.map_file_elevation,
            out_count_array=self.map_file_elevation_count,
        )

    def interpolate_elevations(self):
        if wc_constants.ELEVATION in self.layers:
            # extend elevation to full grid
            mask = np.isnan(self.map_file_elevation)
            X, Y = np.meshgrid(self.latitudes, self.longitudes, indexing="ij")
            result = griddata(
                (X[~mask].ravel(), Y[~mask].ravel()), self.map_file_elevation[~mask].ravel(), (X, Y), method="nearest"
            )
            self.map_file_elevation[:] = result[:]

    def fill_grid(
        self,
        sound_lon: np.ndarray,
        sound_lat: np.ndarray,
        sound_elev: np.ndarray,
        sound_backscatter: np.ndarray,
        compensated: bool,
    ):
        """
        Param :
           - sound_lon / sound_lat / sound_elev : position of the sounds
           - sound_backscatter : value of the sound
        """
        # compute longitude indices
        x_idx = np.round((sound_lon - self.geobox.left) / self.spatial_resolution - 0.5).astype(int)
        # compute latitude indices
        y_idx = np.round((sound_lat - self.geobox.lower) / self.spatial_resolution - 0.5).astype(int)
        # compute elevation indices
        if self.map_file_elevation is not None:
            z_ref = np.full_like(sound_elev, np.nan)
            mask = 0 <= x_idx
            mask &= x_idx < self.map_file_elevation.shape[1]
            mask &= 0 <= y_idx
            mask &= y_idx < self.map_file_elevation.shape[0]
            z_ref[mask] = self.map_file_elevation[y_idx[mask], x_idx[mask]]
            z_idx = np.round((sound_elev - self.min_elevation - z_ref) / self.delta_elevation - 0.5).astype(int)
        else:
            z_idx = np.round((self.max_elevation - sound_elev) / self.delta_elevation - 0.5).astype(int)

        if compensated:
            _fill_grids(
                grid_idxs=z_idx,
                row_idxs=y_idx,
                col_idxs=x_idx,
                backscatters=sound_backscatter,
                o_mean_array=self.map_file_backscatter_comp_mean,
                o_max_array=self.map_file_backscatter_comp_max,
                o_count_array=self.map_file_nb_values_comp,
            )
        else:
            _fill_grids(
                grid_idxs=z_idx,
                row_idxs=y_idx,
                col_idxs=x_idx,
                backscatters=sound_backscatter,
                o_mean_array=self.map_file_backscatter_mean,
                o_max_array=self.map_file_backscatter_max,
                o_count_array=self.map_file_nb_values,
            )


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
    Function aiming to fill grid with each sounder point (in longitudes/latitudes)
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
