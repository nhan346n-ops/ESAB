#! /usr/bin/env python3
# coding: utf-8

from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
from numpy.typing import ArrayLike
from osgeo import osr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from scipy import interpolate

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DTM
import pyat.dtm.utils.dtm_utils as dtm_utils
import pyat.utils.argument_utils as arg_util
import pyat.utils.numpy_utils as np_util
import pyat.utils.pyat_logger as log
from pyat.utils import signal
from pyat.utils.string_utils import xstr


class DtmGridder:
    """
    Utility class to build a DTM from georeferenced points

    How to use :
        Create a DtmGridder with a DtmDriver opened in write mode
        Optionally, set extra layer names to produce (function add_layer)
        Optionally, set extra layer names to compute automatically (function deal_with)
        Optionally, call restrict_elevations to define the elevations thresholds
        Call initialize_dtm_file to prepare the file metadata
        For some set of data :
            Call project_coords to compute cell positions and obtain the projection coords
            Call grid_elevations to add some elevation data
            call grid_keep_last or grid_min_max to layer set by add_layer
            Optionally, call grid_cdi to add some CDI
        If the standard deviation have to be computed, repeat for some set of data :
            Call project_coords to compute cell positions and obtain the projection coords
            Call grid_standard_deviation
        Call finalize_dtm to write grids in the DTM
    """

    @property
    def o_dtm_driver(self) -> dtm_driver.DtmDriver:
        return self._o_dtm_driver

    @property
    def layer_desc(self) -> Dict[str, Tuple[Any, Any]]:
        """
        All the layer description to produce in the DTM
        Key is the name of the layer
        Values are turples with (data type, missing value)
        """
        return self._layer_desc

    @property
    def layers_to_compute(self) -> Set[str]:
        """
        All the layer computed automatically by the gridder
        May be ELEVATION_MIN, ELEVATION_MAX or DTM.STDEV
        """
        return self._layers_to_compute

    @property
    def reference_cdis(self):
        return self._reference_cdis

    def __init__(
        self,
        dtm_driver_to_fill: dtm_driver.DtmDriver,
        geobox: arg_util.Geobox,
        spatial_resolution: float,
        depth_factor: float = 1.0,
        average_elevations=True,
        spatial_antialiasing=False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor.

        Set average_elevations to True to compute the mean elevation in variable DTM.ELEVATION_NAME.
        Otherwise, elevation in cell will be the last projected value.
        """
        self._o_dtm_driver = dtm_driver_to_fill
        self._geobox = geobox
        self._spatial_resolution = spatial_resolution
        self._spatial_antialiasing = spatial_antialiasing
        self._depth_factor = depth_factor
        self._average_elevations = average_elevations
        self._min_elevation = np.nan
        self._max_elevation = np.nan
        self.monitor = monitor
        self.logger = log.logging.getLogger(self.__class__.__name__)

        # List of layer to build. At least Elevation and value_count
        self._layer_desc = {}
        for layer_name in [DTM.ELEVATION_NAME, DTM.VALUE_COUNT]:
            self._layer_desc[layer_name] = (dtm_driver.get_type(layer_name), dtm_driver.get_missing_value(layer_name))

        # No layer to compute by default
        self._layers_to_compute: Set[str] = set()
        # Dict of layer name -> grid data
        self.layer_data: Dict[str, np.ndarray] = {}

        # temporary layer to store mean and std elevation.
        self._tmp_mean_elevation_data = None
        self._tmp_square_elevation_data = None

        self._tmp_mean_x_data = None
        self._tmp_mean_y_data = None

        # temporary layer to store backscatter data and weights for mean calculation.
        self._tmp_mean_backscatter_data = None
        self._tmp_backscatter_weights = None

        # Dictionnary of CDI.
        # Key is the CDI transmit by the process
        # Value is (id of the CDI, Transformed value of the CDI). See __register_cdi_in_reference
        self._reference_cdis: Dict[str, Tuple] = {}
        # Empty CDI means no CDI
        self.reference_cdis[""] = (-1, None)

    def add_layer(self, layer_name: str, data_type: Any = np.float32, missing_value: Any = np.nan) -> None:
        """
        Inform the gridder that an other layer has to be produced
        """
        if layer_name in dtm_driver.LAYER_TYPES:
            self.layer_desc[layer_name] = (dtm_driver.get_type(layer_name), dtm_driver.get_missing_value(layer_name))
        else:
            self.layer_desc[layer_name] = (data_type, missing_value)

    def remove_layer(self, layer_name: str) -> ArrayLike | None:
        """
        Cancel the production of one layer.
        Called before finalize_dtm, this avoids writing the layer to the DTM, even if it has been calculated

        Returns
        -------
        An array containing the produced data
        """
        if layer_name in self.layer_desc:
            del self.layer_desc[layer_name]
        if layer_name in self.layer_data:
            result = self.layer_data[layer_name]
            del self.layer_data[layer_name]
            return result

        return None

    def deal_with(self, layer_name: str) -> None:
        """
        Ask the gridder to compute this layer by it-self
        """
        self.layer_desc[layer_name] = (dtm_driver.get_type(layer_name), dtm_driver.get_missing_value(layer_name))
        self.layers_to_compute.add(layer_name)

    def initialize_dtm_file(self, history="", title="", institution="", source="", references="", comment="") -> None:
        """
        Intialize the DTM, grid size, metadata and history
        """

        # Grid size
        row_count = dtm_utils.estimate_row(self._geobox.upper, self._geobox.lower, self._spatial_resolution)
        col_count = dtm_utils.estimate_col(
            right_or_east=self._geobox.right,
            left_or_west=self._geobox.left,
            spatial_resolution=self._spatial_resolution,
        )

        self.logger.info(f"Initializing Dtm file with {col_count} columns and {row_count} rows")
        if 1 >= row_count >= 20000 and 1 >= col_count >= 20000:
            raise ValueError("Wrong spatial resolution, the resulting Dtm has a bad shape")

        dtm_file = self.o_dtm_driver.dtm_file
        dtm_file.col_count = col_count
        dtm_file.west = self._geobox.left
        dtm_file.spatial_resolution_x = self._spatial_resolution
        dtm_file.row_count = row_count
        dtm_file.south = self._geobox.lower
        dtm_file.spatial_resolution_y = self._spatial_resolution
        dtm_file.spatial_reference = self._geobox.spatial_reference

        metadata = {}
        metadata["title"] = xstr(title)
        metadata["institution"] = xstr(institution)
        metadata["source"] = xstr(source)
        metadata["references"] = xstr(references)
        metadata["comment"] = xstr(comment)
        self.o_dtm_driver.initialize_file(metadata)

        if history:
            self.o_dtm_driver.dataset.history = history

        # Creates all map files to store data temporarily
        for layer_name, description in self.layer_desc.items():
            self.layer_data[layer_name] = self.o_dtm_driver.prepare_memmap_data(
                layer_name, description[0], description[1]
            )

        # Use elevation array with double precision to avoid numerical issues
        self._tmp_mean_elevation_data = self.o_dtm_driver.prepare_memmap_data(
            "tmp_mean_elevation_data",
            np.float64,
            np.nan,
        )

        if self._spatial_antialiasing:
            self._tmp_mean_x_data = self.o_dtm_driver.prepare_memmap_data(
                "tmp_mean_x_data",
                np.float64,
                np.nan,
            )
            self._tmp_mean_y_data = self.o_dtm_driver.prepare_memmap_data(
                "tmp_mean_y_data",
                np.float64,
                np.nan,
            )

        if DTM.STDEV in self.layer_desc:
            self._tmp_square_elevation_data = self.o_dtm_driver.prepare_memmap_data(
                "tmp_square_elevation_data",
                np.float64,
                np.nan,
            )
        if DTM.BACKSCATTER in self.layer_desc:
            self._tmp_mean_backscatter_data = self.o_dtm_driver.prepare_memmap_data(
                "tmp_mean_backscatter_data",
                np.float64,
                np.nan,
            )
            self._tmp_backscatter_weights = self.o_dtm_driver.prepare_memmap_data(
                "tmp_backscatter_weights",
                np.float64,
                0.0,
            )

    def restrict_elevations(self, min_elevation: float, max_elevation: float):
        """Prepare the filtering of elevations."""
        # Swap elevation when min > max
        if min_elevation > max_elevation:
            min_elevation, max_elevation = max_elevation, min_elevation

        self._min_elevation = min_elevation
        self._max_elevation = max_elevation

    def project_coords(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        spatial_reference: osr.SpatialReference = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project all coordinates to the grid to obtain the cell position.
        spatial_reference is the SRS of the xs and ys. None means that the CRS is the same than GeoBox's one
        returns (columns, rows) calculated by the projection as float
        """
        return np_util.project_coords(xs, ys, self._geobox, self._spatial_resolution, spatial_reference)

    def project_coords_as_index(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        spatial_reference: osr.SpatialReference = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project all coordinates to the grid to obtain the cell position.
        spatial_reference is the SRS of the xs and ys. None means that the CRS is the same than GeoBox's one
        returns (columns, rows) calculated by the projection as int
        """
        return np_util.project_coords_as_index(xs, ys, self._geobox, self._spatial_resolution, spatial_reference)

    def grid_elevations(
        self, columns: np.ndarray, rows: np.ndarray, elevations: np.ndarray, cdi: Optional[str] = None
    ) -> None:
        """
        Project all elevations in the DTM.
        If optional layers are specified, they are calculated from elevation data
        If cdi is specified, it is set on projected cell (if none)
        rows, columns : coordinates in dtm grid as float
        """
        # Discard cell of filtered elevations
        if not np.isnan(self._min_elevation) and not np.isnan(self._max_elevation):
            elevations[self._min_elevation > elevations] = np.nan
            elevations[elevations > self._max_elevation] = np.nan

        out_mean_array = self._tmp_mean_elevation_data

        out_mean_x_array = None
        out_mean_y_array = None
        if self._spatial_antialiasing:
            out_mean_x_array = self._tmp_mean_x_data
            out_mean_y_array = self._tmp_mean_y_data

        out_last_array = None
        if not self._average_elevations:
            out_last_array = self.layer_data[DTM.ELEVATION_NAME]

        # Compute min/mean/max elevations if required
        out_min_array = (
            self.layer_data.get(DTM.ELEVATION_MIN, None) if DTM.ELEVATION_MIN in self.layers_to_compute else None
        )
        out_max_array = (
            self.layer_data.get(DTM.ELEVATION_MAX, None) if DTM.ELEVATION_MAX in self.layers_to_compute else None
        )
        out_filtered_array = (
            self.layer_data.get(DTM.FILTERED_COUNT, None) if DTM.FILTERED_COUNT in self.layers_to_compute else None
        )

        if self._depth_factor != 1.0:
            elevations = elevations * self._depth_factor

        np_util.compute_statistics(
            in_array=elevations,
            x_array=columns,
            y_array=rows,
            out_x_array=out_mean_x_array,
            out_y_array=out_mean_y_array,
            out_last_array=out_last_array,
            out_mean_array=out_mean_array,
            out_min_array=out_min_array,
            out_max_array=out_max_array,
            out_count_array=self.layer_data[DTM.VALUE_COUNT],
            out_filtered_array=out_filtered_array,
        )
        if cdi:
            self.__set_cdi_where_none(cdi)

    def grid_backscatter(
        self,
        columns: np.ndarray,
        rows: np.ndarray,
        elevations: np.ndarray,
        backscatter: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Grid backscatter values. Cells will contains the mean value of backscatter that should be linearized first.
        Conversion to dB will be done in finalize step.
        """

        # Discard cell of filtered elevations
        backscatter[np.isnan(elevations)] = np.nan

        # Compute iterative mean
        if weights is None:
            np_util.compute_statistics(
                in_array=backscatter,
                x_array=columns,
                y_array=rows,
                out_mean_array=self._tmp_mean_backscatter_data,
                out_count_array=self._tmp_backscatter_weights,
            )
        else:
            np_util.compute_weighted_statistics(
                in_array=backscatter,
                x_array=columns,
                y_array=rows,
                in_weights=weights,
                out_mean_array=self._tmp_mean_backscatter_data,
                out_weighted_count_array=self._tmp_backscatter_weights,
            )

    def grid_keep_last(
        self,
        layer_name: str,
        values: np.ndarray,
        columns: np.ndarray,
        rows: np.ndarray,
    ) -> None:
        """
        Grid values in a layer. Cells will contains only the last met value
        """
        factor = 1.0
        missing_value = self.layer_desc[layer_name][1] if layer_name in self.layer_desc else np.nan
        if layer_name in [DTM.ELEVATION_MIN, DTM.ELEVATION_MAX, DTM.ELEVATION_SMOOTHED_NAME]:
            factor = self._depth_factor
        np_util.project_into_grid_keep_last(values, columns, rows, self.layer_data[layer_name], missing_value, factor)

    def grid_min_max(
        self,
        min_layer_name: Optional[str],
        max_layer_name: Optional[str],
        values: np.ndarray,
        columns: np.ndarray,
        rows: np.ndarray,
    ) -> None:
        """
        Grid values in min and max layer at once
        Expeced min_layer_name or max_layer_name or both
        """
        np_util.compute_statistics(
            in_array=values,
            x_array=columns,
            y_array=rows,
            out_min_array=self.layer_data[min_layer_name] if min_layer_name in self.layer_data else None,
            out_max_array=self.layer_data[max_layer_name] if max_layer_name in self.layer_data else None,
        )

    def grid_cdi(self, cdis: np.ndarray, columns: np.ndarray, rows: np.ndarray, cdi_or_cprd_prefix: str) -> None:
        """
        Used this function when CDI may be different for each cell (as in CSV bathymetric files)
        """
        if DTM.CDI_INDEX in self.layer_data:
            # Add new cdi to the reference
            for in_cdis in set(cdis):
                self.__register_cdi_in_reference(in_cdis, cdi_or_cprd_prefix)

            # make an array with the new CDI index
            in_cdi_index = np.array([self.reference_cdis[in_cdi][0] for in_cdi in cdis])
            np_util.project_into_grid_keep_last(in_cdi_index, columns, rows, self.layer_data[DTM.CDI_INDEX], -1)

    def grid_standard_deviation(self, columns: np.ndarray, rows: np.ndarray, in_depths: np.ndarray) -> None:
        """
        First pass of the standard deviation computation
        """
        if DTM.STDEV in self.layers_to_compute:
            np_util.compute_standard_deviation_first_pass(in_depths, columns, rows, self._tmp_square_elevation_data)

    def reset_cell(self, threshold: int) -> None:
        """
        For all layers, set the cell's value to its missing value where the number of collected values is less than a threshold
        """
        self.logger.info(f"Reset cells with less than {threshold} soundings")
        value_count = self.layer_data[DTM.VALUE_COUNT]
        value_count[value_count < threshold] = dtm_driver.get_missing_value(DTM.VALUE_COUNT)
        for layer in [
            DTM.ELEVATION_NAME,
            DTM.ELEVATION_MIN,
            DTM.ELEVATION_MAX,
            DTM.STDEV,
            DTM.BACKSCATTER,
            DTM.MIN_ACROSS_DISTANCE,
            DTM.MAX_ACROSS_DISTANCE,
            DTM.MAX_ACCROSS_ANGLE,
        ]:
            if layer in self.layer_data:
                self.layer_data[layer][value_count <= 0] = dtm_driver.get_missing_value(layer)

        # reset temporary map files
        if self._tmp_mean_elevation_data is not None:
            self._tmp_mean_elevation_data[value_count <= 0] = np.nan
        if self._tmp_square_elevation_data is not None:
            self._tmp_square_elevation_data[value_count <= 0] = np.nan

    def __set_cdi_where_none(self, cdi: str) -> None:
        """Set the CDI"""
        if cdi not in self.reference_cdis:
            self.reference_cdis[cdi] = (len(self.reference_cdis) - 1, cdi)
        cdi_index = self.reference_cdis[cdi][0]

        # Initialization of CDI_INDEX layer
        if DTM.CDI_INDEX not in self.layer_data:
            self.layer_data[DTM.CDI_INDEX] = self.o_dtm_driver.prepare_memmap_data(DTM.CDI_INDEX)
        o_layer_cdi_index = self.layer_data[DTM.CDI_INDEX]

        # Set the CDI where elevations exist and index of CDI is absent
        value_count = self.layer_data[DTM.VALUE_COUNT]
        missing_value = dtm_driver.get_missing_value(DTM.CDI_INDEX)
        new_cdi_index = np.full_like(o_layer_cdi_index, cdi_index)
        new_cdi_index[value_count <= 0] = missing_value  # Mask cells without elevations
        new_cdi_index[o_layer_cdi_index != missing_value] = missing_value  # Mask cells with a CDI affected
        o_layer_cdi_index[new_cdi_index != missing_value] = cdi_index

    def __apply_antialiasing(self):
        """
        Apply antialiasing on unregular grid of dtm centroids (mean_x, mean_y, mean_elevation) and write result in elevation layer.
        Implementation using scipy.interpolate.griddata by block to optimise execution time and memory.
        """

        full_size = self._tmp_mean_elevation_data.shape
        full_mask = np.isfinite(self._tmp_mean_elevation_data)

        # Output Grid
        x = np.linspace(0.5, full_size[0] - 0.5, full_size[0])
        y = np.linspace(0.5, full_size[1] - 0.5, full_size[1])
        yy, xx = np.meshgrid(y, x)

        # Apply scipy.interpolate.griddata by block
        block_size = 200
        overlap = 3
        j = 0
        while j < full_size[1]:
            self.logger.info(f"antialias ({j} / {full_size[1]}) rows")
            # define source and dest rows slices
            j_begin_src = max(j - overlap, 0)
            j_end_src = min(j + block_size + overlap, full_size[1])
            j_begin_dst = j
            j_end_dst = min(j + block_size, full_size[1])
            i = 0
            while i < full_size[0]:
                # define source and dest columns slices
                i_begin_src = max(i - overlap, 0)
                i_end_src = min(i + block_size + overlap, full_size[0])
                i_begin_dst = i
                i_end_dst = min(i + block_size, full_size[0])

                slice_src = (slice(i_begin_src, i_end_src), slice(j_begin_src, j_end_src))
                slice_dst = (slice(i_begin_dst, i_end_dst), slice(j_begin_dst, j_end_dst))

                mask_src = full_mask[slice_src]
                mask_dst = full_mask[slice_dst]
                if np.any(mask_dst):
                    ref_z = self._tmp_mean_elevation_data[slice_src][mask_src]
                    ref_x = self._tmp_mean_x_data[slice_src][mask_src]
                    ref_y = self._tmp_mean_y_data[slice_src][mask_src]
                    if len(ref_z) >= 4:  # condition to compute griddata
                        interp_z = interpolate.griddata(
                            (ref_x, ref_y),
                            ref_z,
                            (
                                yy[slice_dst],
                                xx[slice_dst],
                            ),
                            method="linear",
                        )
                        mask_dst = mask_dst & ~np.isnan(interp_z)
                        self.layer_data[DTM.ELEVATION_NAME][slice_dst][mask_dst] = interp_z[mask_dst]
                i = i + block_size
            j = j + block_size

    def finalize_dtm(self, default_cdi: Optional[str] = None) -> None:
        """
        Finalize the DTM.
        The default_cdi parameter is used to set the CDI value on cell without one
        """
        if self._average_elevations and self._tmp_mean_elevation_data is not None:
            self.layer_data[DTM.ELEVATION_NAME][:] = self._tmp_mean_elevation_data[:]
            if self._spatial_antialiasing:
                self.logger.info("Processing anti-aliasing (centroid bilinear interpolation)")
                self.__apply_antialiasing()

        if DTM.STDEV in self.layers_to_compute:
            self.logger.info("Finalizing standard deviation calculation")
            stddev_array = self.layer_data[DTM.STDEV]
            np_util.compute_standard_deviation_second_pass(
                self.layer_data[DTM.VALUE_COUNT],
                self._tmp_mean_elevation_data,
                self._tmp_square_elevation_data,
                stddev_array,
            )

        # Convert back Reflectivity mean amplitude to dB
        if DTM.BACKSCATTER in self.layer_data:
            self.layer_data[DTM.BACKSCATTER][:] = signal.amplitude_to_db(self._tmp_mean_backscatter_data)

        self.logger.info("Writing the Dtm file")
        layer_count = len(self.layer_desc)
        current_layer_index = 1
        for layer_name, description in self.layer_desc.items():
            if layer_name in self.layer_data and (
                layer_name != DTM.VALUE_COUNT or DTM.VALUE_COUNT in self.layers_to_compute
            ):
                log.info_progress_layer(self.logger, " : writing layer ", layer_name, current_layer_index, layer_count)
                self.o_dtm_driver.add_layer(layer_name, self.layer_data[layer_name], description[0], description[1])
                current_layer_index += 1

        # Add the default CDI to the reference before writing them
        if default_cdi:
            self.__set_cdi_where_none(default_cdi)

        # Write layers CDI if any (CDI "" is ignored)
        if len(self.reference_cdis) > 1 and DTM.CDI_INDEX in self.layer_data:
            if DTM.CDI_INDEX not in self.o_dtm_driver:
                var_cdi_index = self.o_dtm_driver.add_layer(DTM.CDI_INDEX)
                var_cdi_index[:] = self.layer_data[DTM.CDI_INDEX]
            cdis = [cdi for cdiId, cdi in self.reference_cdis.values() if cdiId >= 0]
            self.o_dtm_driver.create_cdi_reference_variable(cdis=cdis)

        # Clean tmp file
        if self._tmp_mean_elevation_data is not None:
            del self._tmp_mean_elevation_data
        if self._tmp_mean_x_data is not None:
            del self._tmp_mean_x_data
        if self._tmp_mean_y_data is not None:
            del self._tmp_mean_y_data
        if self._tmp_square_elevation_data is not None:
            del self._tmp_square_elevation_data
        if self._tmp_mean_backscatter_data is not None:
            del self._tmp_mean_backscatter_data
        if self._tmp_backscatter_weights is not None:
            del self._tmp_backscatter_weights

    def __register_cdi_in_reference(self, cdi: str, cdi_or_cprd_prefix: str) -> None:
        """
        Add the CDI to the reference and affect an id
        """
        if cdi and cdi not in self.reference_cdis:
            if cdi == "INT":
                self.reference_cdis[cdi] = (len(self.reference_cdis) - 1, DTM.INTERPOLATED_CDI)
                self.logger.info(f"New CDI found : {self.reference_cdis[cdi][1]}")
            else:
                self.reference_cdis[cdi] = (len(self.reference_cdis) - 1, cdi_or_cprd_prefix + cdi)
                self.logger.info(f"New CDI/CPRD found : {self.reference_cdis[cdi][1]}")
