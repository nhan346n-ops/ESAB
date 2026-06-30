#! /usr/bin/env python3
# coding: utf-8

import tempfile
from typing import Any, Tuple, Union

import numpy as np
from osgeo import gdal
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.utils.dtm_utils as dtm_utils
import pyat.utils.argument_utils as arg_util
import pyat.utils.numpy_utils as np_util
import pyat.utils.pyat_logger as log


class TiffGridder:
    """
    Utility class to build a Tiff from georeferenced points

    How to use :
        Create a TiffGridder with a a path to the Tiff to build
        Call initialize_tiff_file to prepare the file metadata
        For some set of data :
            Call project_coords to compute cell positions and obtain the projection coords
            Call grid_keep_last to add some data
        Call finalize_tiff to write grids in the Tiff
    """

    def __init__(
        self,
        tiff_path: str,
        geobox: arg_util.Geobox,
        spatial_resolution: float,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor.
        """
        self._tiff_path = tiff_path
        self._geobox = geobox
        self._spatial_resolution = spatial_resolution
        self.monitor = monitor
        self.logger = log.logging.getLogger(self.__class__.__name__)

    # pylint: disable = consider-using-with
    def initialize_tiff_file(self, dtype: Any, fill_value: Union[float, int, None] = None) -> None:
        """
        Intialize the Tiff
        dtype is float or int
        """
        # Grid size
        row_count = dtm_utils.estimate_row(self._geobox.upper, self._geobox.lower, self._spatial_resolution)
        col_count = dtm_utils.estimate_col(
            right_or_east=self._geobox.right,
            left_or_west=self._geobox.left,
            spatial_resolution=self._spatial_resolution,
        )

        self.logger.info(f"Initializing Tiff file with {col_count} columns and {row_count} rows")
        if 1 >= row_count >= 20000 and 1 >= col_count >= 20000:
            raise ValueError("Wrong spatial resolution, the resulting Tiff has a bad shape")

        self.fill_value = fill_value
        if fill_value is None:
            self.fill_value = np.nan if dtype is float else 2**31 - 1

        self.temp_map_file = tempfile.TemporaryFile(suffix=".memmap", prefix=".tif")
        self.map_file = np.memmap(
            self.temp_map_file,
            shape=(row_count, col_count),
            dtype=np.float32 if dtype is float else np.int32,
            mode="w+",
        )
        self.map_file.fill(self.fill_value)

        # Need this temporary grid in case of average computation
        self.temp_value_count_file = tempfile.TemporaryFile(suffix=".memmap", prefix=".dat")
        self.temp_value_count = np.memmap(
            self.temp_value_count_file,
            shape=(row_count, col_count),
            dtype=np.int32,
            mode="w+",
        )
        self.temp_value_count.fill(0)

    def project_coords(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project all coordinates to the grid to obtain the cell position.
        returns (columns, rows) calculated by the projection
        """
        return np_util.project_coords_as_index(xs, ys, self._geobox, self._spatial_resolution)

    def grid_keep_last(self, columns: np.ndarray, rows: np.ndarray, values: np.ndarray, factor: float = 1.0) -> None:
        """
        Project all values in the grid.
        """
        np_util.project_into_grid_keep_last(values, columns, rows, self.map_file, self.fill_value, factor)

    def grid_average(self, columns: np.ndarray, rows: np.ndarray, values: np.ndarray) -> None:
        """
        Project all values in the grid and compute an average values for each cell.
        """
        np_util.compute_statistics(
            values, columns, rows, out_mean_array=self.map_file, out_count_array=self.temp_value_count
        )

    def finalize_tiff(self, compression: bool = True) -> None:
        # Create output
        driver = gdal.GetDriverByName("GTiff")
        creation_options = ["COMPRESS=DEFLATE"] if compression else None
        outRaster = driver.Create(
            utf8_path=self._tiff_path,
            xsize=self.map_file.shape[1],
            ysize=self.map_file.shape[0],
            bands=1,
            eType=gdal.GDT_Float32 if np.issubdtype(self.map_file.dtype, np.floating) else gdal.GDT_Int32,
            options=creation_options,
        )
        outRaster.SetGeoTransform(
            (self._geobox.left, self._spatial_resolution, 0.0, self._geobox.upper, 0.0, -self._spatial_resolution)
        )
        outband: gdal.Band = outRaster.GetRasterBand(1)
        outband.SetNoDataValue(self.fill_value)
        outRaster.SetProjection(self._geobox.spatial_reference.ExportToWkt())
        outband.WriteArray(self.map_file[::-1, :])
        outband.FlushCache()
        del outband, outRaster, self.map_file
