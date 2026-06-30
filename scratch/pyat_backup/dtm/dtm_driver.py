#! /usr/bin/env python3
# coding: utf-8
# pylint:disable=too-many-lines
import logging
import math
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import netCDF4 as nc
import numba
import numpy as np
from osgeo import gdal, osr

import pyat.common.geo_file as gf
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.numba.default_layers_functions as nb
import pyat.utils.netcdf_utils as nc_util
from pyat.dtm.utils.mercator_utils import clean_mercator
from pyat.utils import string_utils
from pyat.utils.nc_encoding import open_nc_file

# Layer's type in a dtm file

LAYER_TYPES = {
    DtmConstants.ELEVATION_NAME: np.float32,
    DtmConstants.ELEVATION_MIN: np.float32,
    DtmConstants.ELEVATION_MAX: np.float32,
    DtmConstants.VALUE_COUNT: np.int32,
    DtmConstants.STDEV: np.float32,
    DtmConstants.CDI: str,
    DtmConstants.CDI_INDEX: np.int32,
    DtmConstants.ELEVATION_SMOOTHED_NAME: np.float32,
    DtmConstants.INTERPOLATION_FLAG: np.int8,
    DtmConstants.BACKSCATTER: np.float32,
    DtmConstants.MIN_ACROSS_DISTANCE: np.float32,
    DtmConstants.MAX_ACROSS_DISTANCE: np.float32,
    DtmConstants.MAX_ACCROSS_ANGLE: np.float32,
    DtmConstants.FILTERED_COUNT: np.int32,
}
# All possible layer in a dtm file
LAYER_NAMES = list(LAYER_TYPES.keys())


def get_type(layerName):
    return LAYER_TYPES[layerName] if layerName in LAYER_TYPES else None


class DtmFile(gf.GeoFile):
    """
    dtm file's properties. This is a GeoFile with
       - Spatial resolution
       - Shape : a grid dimension (shape)
    """

    def __init__(self, filePath: str, spatial_reference: osr.SpatialReference = gf.SR_WGS_84):
        # The logger
        self.logger = logging.getLogger(self.__class__.__name__)
        super().__init__(filePath)
        self.spatial_reference = spatial_reference
        self.spatial_resolution_x = np.nan
        self.spatial_resolution_y = np.nan
        self.west = self.east = self.south = self.north = np.nan
        self.row_count = self.col_count = np.nan

        # For each layer, initialize min/max
        self._minmax = {layername: [np.nan, np.nan] for layername in LAYER_NAMES}

    @property
    def spatial_resolution_x(self):
        return self._spatial_resolution_x

    @spatial_resolution_x.setter
    def spatial_resolution_x(self, spatial_resolution_x: float):
        self._spatial_resolution_x = spatial_resolution_x

    @property
    def spatial_resolution_y(self):
        return self._spatial_resolution_y

    @spatial_resolution_y.setter
    def spatial_resolution_y(self, spatial_resolution_y: float):
        self._spatial_resolution_y = spatial_resolution_y

    @property
    def row_count(self):
        return self._rowCount

    @row_count.setter
    def row_count(self, rowCount: int):
        self._rowCount = rowCount

    @property
    def col_count(self):
        return self._colCount

    @col_count.setter
    def col_count(self, colCount: int):
        self._colCount = colCount

    def row(self, latitude: float):
        """
        Compute the row index for the given latitude
        :param latitude : instance of np.float64
        """
        return int(math.floor((latitude - self.south) / self.spatial_resolution_y))

    def column(self, longitude: float):
        """
        Compute the column index for the given longitude
        :param longitude : instance of np.float64
        """
        return int(math.floor((longitude - self.west) / self.spatial_resolution_x))

    def project(self, array: np.ndarray, longitudeColumn: int, latitudeColumn: int):
        """
        Compute the column and row indexes for each line of the specified array
        :param array : array containing longitudes and latitudes
        :param longitudeColumn : index of the longitudes column in array
        :param latitudeColumn : index of the latitudes column in array
        """
        DtmFile.__project(
            array,
            longitudeColumn,
            self.west,
            latitudeColumn,
            self.south,
            self.spatial_resolution_y,
            self.spatial_resolution_x,
        )

    # noinspection PyMethodParameters
    @staticmethod
    @numba.guvectorize(
        ["void(float64[:,:], int32, float64, int32, float64, float64, float64)"],
        "(r, c),(),(),(),(),(),()",
        target="parallel",
        nopython=True,
    )
    def __project(
        array: np.ndarray,
        longitudeColumn: int,
        west: float,
        latitudeColumn: int,
        south: float,
        spatialResolutionY: float,
        spatialResolutionX: float,
    ):
        for i in numba.prange(array.shape[0]):
            array[i, longitudeColumn] = round(
                (array[i, longitudeColumn] - (west + 0.5 * spatialResolutionX)) / spatialResolutionX
            )
            array[i, latitudeColumn] = round(
                (array[i, latitudeColumn] - (south + 0.5 * spatialResolutionY)) / spatialResolutionY
            )

    def compute_x_axis(self) -> np.ndarray:
        """
        Returns an array of longitudes/x covering the dtm
        """
        first_x = self.west + 0.5 * self.spatial_resolution_x
        last_x = first_x + (self.col_count - 1) * self.spatial_resolution_x
        result = np.linspace(first_x, last_x, self.col_count, dtype=float)
        if self.spatial_reference.IsGeographic():
            # Check if longitudes span the 180th meridian
            result = np.where(result > 180.0, result - 360.0, result)
        return result

    def compute_y_axis(self) -> np.ndarray:
        """
        Returns an array of latitudes/y covering the dtm
        """
        first_y = self.south + 0.5 * self.spatial_resolution_y
        last_y = first_y + (self.row_count - 1) * self.spatial_resolution_y
        result = np.linspace(first_y, last_y, self.row_count, dtype=float)
        return result

    def initialize_with_gdal_dataset(self, gdal_dataset: gdal.Dataset):
        """
        Initialize this dtm_file with the metadata contained in the gdal dataset
        """
        self.col_count = gdal_dataset.RasterXSize
        self.row_count = gdal_dataset.RasterYSize

        projection = gdal_dataset.GetProjection()
        if projection:
            spatial_reference = osr.SpatialReference()
            if spatial_reference.ImportFromWkt(projection) == gdal.ogr.OGRERR_NONE:
                self.spatial_reference = spatial_reference

        geo_transform = gdal_dataset.GetGeoTransform()
        if geo_transform:
            self.spatial_resolution_x = abs(geo_transform[1])
            self.spatial_resolution_y = abs(geo_transform[5])
            x0, y0 = gdal.ApplyGeoTransform(geo_transform, 0, 0)
            xn, yn = gdal.ApplyGeoTransform(geo_transform, self.col_count, self.row_count)
            self.west = x0 if geo_transform[1] >= 0 else xn
            self.east = x0 if geo_transform[1] < 0 else xn
            self.south = y0 if geo_transform[5] >= 0 else yn
            self.north = y0 if geo_transform[5] < 0 else yn


def copy_metadata(from_dtm: DtmFile, to_dtm: DtmFile):
    """
    Initialize the to_dtm with the metadata contained in the from_dtm DtmFile
    """
    to_dtm.col_count = from_dtm.col_count
    to_dtm.row_count = from_dtm.row_count
    to_dtm.spatial_reference = from_dtm.spatial_reference
    to_dtm.spatial_resolution_x = from_dtm.spatial_resolution_x
    to_dtm.spatial_resolution_y = from_dtm.spatial_resolution_y
    to_dtm.west = from_dtm.west
    to_dtm.east = from_dtm.east
    to_dtm.south = from_dtm.south
    to_dtm.north = from_dtm.north


# static function
def get_missing_value(layerName: str):
    """
    Define the missing value for a layer
    """
    layerType = get_type(layerName)
    if layerType == np.int8:
        return np.int8(0x7F)
    elif layerType == np.int32:
        return -1
    return np.nan


def __configure_elevation(variable: nc.Variable) -> None:
    """
    Configuration of the elevation variable
    """
    variable.long_name = "Elevation relative to Lowest Astronomical Tide datum"
    variable.units = "m"
    variable.comment = "Gridded data are stored as a two-dimensional array of float values of elevation in metres, with negative values for bathymetric depths and positive values for topographic heights"
    variable.sdn_parameter_urn = "SDN:P01::HGHTALAT"
    variable.sdn_parameter_name = (
        "Topographic height of seafloor relative to Lowest Astronomical Tide datum {sea-floor height}"
    )
    variable.sdn_uom_urn = "SDN:P06::ULAA"
    variable.sdn_uom_name = "Metres"
    variable.grid_mapping = DtmConstants.CRS_NAME


def __configure_elevation_min(variable: nc.Variable) -> None:
    """
    Configuration of the min elevation variable
    """
    variable.long_name = "Min elevation value over a cell, relative to Lowest Astronomical Tide"
    variable.units = "m"
    variable.sdn_uom_urn = "SDN:P06::ULAA"
    variable.sdn_uom_name = "Metres"
    # we need to add esri_pe_string for arcgis compliance
    # elevation.esri_pe_string = self.input.variables[ifr.VARIABLE_DEPTH].esri_pe_string
    variable.grid_mapping = DtmConstants.CRS_NAME


def __configure_elevation_max(variable: nc.Variable) -> None:
    """
    Configuration of the min elevation variable
    """
    variable.long_name = "Max elevation value over a cell, relative to Lowest Astronomical Tide"
    variable.units = "m"
    variable.sdn_uom_urn = "SDN:P06::ULAA"
    variable.sdn_uom_name = "Metres"
    # we need to add esri_pe_string for arcgis compliance
    # elevation.esri_pe_string = self.input.variables[ifr.VARIABLE_DEPTH].esri_pe_string
    variable.grid_mapping = DtmConstants.CRS_NAME


def __configure_depth_stdev(variable: nc.Variable) -> None:
    """
    Configuration of the STDEV variable
    """
    variable.long_name = "Standard Deviation of elevation data over cell"
    variable.units = "m"
    variable.sdn_uom_urn = "SDN:P06::ULAA"
    variable.sdn_uom_name = "Metres"
    # we need to add esri_pe_string for arcgis compliance
    # elevation.esri_pe_string = self.input.variables[ifr.VARIABLE_DEPTH].esri_pe_string
    variable.grid_mapping = DtmConstants.CRS_NAME


def __configure_depth_smooth(variable: nc.Variable) -> None:
    """
    Configuration of the STDEV variable
    """
    variable.long_name = "Smoothed elevation relative to sea level, computing with elevation variable"
    variable.units = "m"
    variable.sdn_uom_urn = "SDN:P06::ULAA"
    variable.sdn_uom_name = "Metres"
    variable.grid_mapping = DtmConstants.CRS_NAME


def __configure_value_count(variable: nc.Variable) -> None:
    """
    Configuration of the cell value count variable
    """
    variable.long_name = "Number of values used to compute the resulting elevation over the cell "
    variable.valid_range = [0, 0x7FFFFFFE]
    variable.grid_mapping = DtmConstants.CRS_NAME


def __configure_filtered_count(variable: nc.Variable) -> None:
    """
    Configuration of the cell filtered count variable
    """
    variable.long_name = "Number of values matching the cell coordinates but not used in mean elevation computation, these could be invalid soundings or rejected with filters "
    variable.valid_range = [0, 0x7FFFFFFE]
    variable.grid_mapping = DtmConstants.CRS_NAME


def __configure_cell_interpolation_flag(variable: nc.Variable) -> None:
    """
    Configuration of the cell value count variable
    """
    variable.long_name = (
        "Indicator of cell processed as extrapolation of the neighbouring cells (absence of real soundings data)."
    )
    variable.valid_range = (np.uint8(0), np.int8(1))
    variable.flag_values = variable.valid_range
    variable.flag_meaning = "not_interpolated interpolated"
    variable.grid_mapping = DtmConstants.CRS_NAME


def __configure_cdi_index(variable: nc.Variable) -> None:
    variable.long_name = (
        "CDI index of this cell. matching CDI information is retrieved from " + DtmConstants.CDI + " variable "
    )
    variable.valid_range = [0, 0x7FFFFFFE]
    variable.grid_mapping = DtmConstants.CRS_NAME
    variable.ancillary_variables = DtmConstants.CDI


def __configure_cdi(variable: nc.Variable) -> None:
    variable.long_name = (
        "ID of related CDI metadata record set complete with truncated Id = EDMO-code-provider_Local-CDI-Id"
    )


def __configure_backscatter(variable: nc.Variable) -> None:
    """
    Configuration of the cell value count variable
    """
    variable.long_name = "backscatter value"
    variable.grid_mapping = DtmConstants.CRS_NAME


def __configure_min_across_distance(variable: nc.Variable) -> None:
    """
    Configuration of the cell value count variable
    """
    variable.long_name = "min across distance value of sounding detection, the distance is the across distance from the detection to the platform at time of acquisition"
    variable.grid_mapping = DtmConstants.CRS_NAME


def __configure_max_across_distance(variable: nc.Variable) -> None:
    """
    Configuration of the cell value count variable
    """
    variable.long_name = "max across distance value of sounding detection, the distance is the across distance from the detection to the platform at time of acquisition"
    variable.grid_mapping = DtmConstants.CRS_NAME


def __configure_max_across_angle(variable: nc.Variable) -> None:
    """
    Configuration of the cell value count variable
    """
    variable.long_name = "max across angle value of sounding detection, the angle is the associated beam pointing angle to the transducer at time of acquisition"
    variable.grid_mapping = DtmConstants.CRS_NAME


LAYER_CONFIGURATOR_FUNCTIONS = {
    DtmConstants.ELEVATION_NAME: __configure_elevation,
    DtmConstants.ELEVATION_MIN: __configure_elevation_min,
    DtmConstants.ELEVATION_MAX: __configure_elevation_max,
    DtmConstants.VALUE_COUNT: __configure_value_count,
    DtmConstants.FILTERED_COUNT: __configure_filtered_count,
    DtmConstants.STDEV: __configure_depth_stdev,
    DtmConstants.CDI_INDEX: __configure_cdi_index,
    DtmConstants.CDI: __configure_cdi,
    DtmConstants.ELEVATION_SMOOTHED_NAME: __configure_depth_smooth,
    DtmConstants.INTERPOLATION_FLAG: __configure_cell_interpolation_flag,
    DtmConstants.MAX_ACCROSS_ANGLE: __configure_max_across_angle,
    DtmConstants.MAX_ACROSS_DISTANCE: __configure_max_across_distance,
    DtmConstants.MIN_ACROSS_DISTANCE: __configure_min_across_distance,
    DtmConstants.BACKSCATTER: __configure_backscatter,
}


class DtmDriver:
    @property
    def dtm_file(self) -> DtmFile:
        return self._dtm_file

    @property
    def dataset(self) -> nc.Dataset:
        return self._dataset

    def __init__(self, file_path: str):
        # The logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Opening {file_path}")

        self._dtm_file = DtmFile(file_path)
        self._dataset = None

        # Try to initialize _dtm_file with gdal / netCDF4
        if os.path.isfile(file_path):
            gdal_dataset = None

            try:
                # 1. First, always try to read the projection using netCDF4 from the crs variable
                crs_loaded = False
                with self.open():
                    if "crs" in self.dataset.variables:
                        crs_var = self.dataset.variables["crs"]
                        if "crs_wkt" in crs_var.ncattrs():
                            crs_wkt = crs_var.getncattr("crs_wkt")
                            spatial_reference = osr.SpatialReference()
                            if spatial_reference.ImportFromWkt(crs_wkt) == gdal.ogr.OGRERR_NONE:
                                self._dtm_file.spatial_reference = spatial_reference
                                crs_loaded = True
                    
                    if not crs_loaded:
                        # Guess based on variable names (x/y indicates projected)
                        if DtmConstants.DIM_ABSCISSA in self.dataset.variables or DtmConstants.DIM_ORDINATE in self.dataset.variables:
                            pass # We will rely on get_x_axis/get_y_axis checking variable existence later
                        else:
                            self._dtm_file.spatial_reference = gf.SR_WGS_84

                # 2. Try to initialize projection and get_geo_transform using GDAL
                gdal.UseExceptions()
                has_netcdf = gdal.GetDriverByName("netCDF") is not None
                if has_netcdf:
                    try:
                        # Reading metadata to find variable to read
                        with open_nc_file(file_path) as inDataset:
                            if DtmConstants.ELEVATION_NAME in inDataset.variables:
                                var_to_read = DtmConstants.ELEVATION_NAME
                            else:
                                var_to_read = DtmConstants.BACKSCATTER
                        gdal_dataset = gdal.Open(f'NETCDF:"{file_path}":{var_to_read}', gdal.GA_ReadOnly)

                        if gdal_dataset:
                            if not crs_loaded:
                                projection = gdal_dataset.GetProjection()
                                if projection:
                                    spatial_reference = osr.SpatialReference()
                                    if spatial_reference.ImportFromWkt(projection) == gdal.ogr.OGRERR_NONE:
                                        self._dtm_file.spatial_reference = spatial_reference

                            # retrieve the values with gdal
                            geo_transform = gdal_dataset.GetGeoTransform()
                            if geo_transform:
                                self.dtm_file.spatial_resolution_x = abs(geo_transform[1])
                                self.dtm_file.spatial_resolution_y = abs(geo_transform[5])
                                # Specific case : spanning 180th on the first cell
                                if self._dtm_file.spatial_reference.IsGeographic() and geo_transform[1] < 0.0:
                                    self.dtm_file.spatial_resolution_x = 360.0 - self.dtm_file.spatial_resolution_x
                    except Exception as gdal_err:
                        self.logger.debug(f"GDAL netCDF open failed: {gdal_err}")

                # 3. Initialize grid dimensions and resolutions (falling back to coordinate arrays if GDAL failed)
                with self.open():
                    abscissa = self.get_x_axis()
                    self.dtm_file.col_count = len(abscissa)
                    
                    res_x = self.dtm_file.spatial_resolution_x
                    if res_x is None or np.isnan(res_x) or res_x == 0:
                        res_x = abs(abscissa[1] - abscissa[0]) if len(abscissa) > 1 else 1.0
                        # Specific case : spanning 180th on the first cell
                        if self.dtm_file.spatial_reference.IsGeographic() and abscissa[0] >= 0.0 > abscissa[1]:
                            res_x = 360.0 - res_x
                        self.dtm_file.spatial_resolution_x = res_x

                    self.dtm_file.west = abscissa[0] - res_x / 2.0
                    self.dtm_file.east = (
                        abscissa[self.dtm_file.col_count - 1] + res_x / 2.0
                    )

                    ordinates = self.get_y_axis()
                    self.dtm_file.row_count = len(ordinates)
                    
                    res_y = self.dtm_file.spatial_resolution_y
                    if res_y is None or np.isnan(res_y) or res_y == 0:
                        res_y = abs(ordinates[1] - ordinates[0]) if len(ordinates) > 1 else 1.0
                        self.dtm_file.spatial_resolution_y = res_y
                        
                    self.dtm_file.south = ordinates[0] - res_y / 2.0
                    self.dtm_file.north = (
                        ordinates[self.dtm_file.row_count - 1] + res_y / 2.0
                    )

            except Exception as e:
                self.logger.error(f"Error initializing DTM file {file_path}: {e}")
            finally:
                gdal_dataset = None  # Close the file
                self._dataset = None  # See open()

    def get_file_path(self) -> str:
        return self.dtm_file.file_path

    def open(self, mode: str = "r") -> nc.Dataset:
        """
        Open the file and return the resulting Dataset
        """
        self._dataset = open_nc_file(self.dtm_file.file_path, mode=mode)
        if self.dataset.file_format != DtmConstants.FORMAT:
            self.dataset.close()
            raise ValueError(
                f"The format of the file {self.dtm_file.file_path} must be {DtmConstants.FORMAT} (instead of {self.dataset.file_format})."
            )

        return self.dataset

    def create_file(
        self,
        col_count: int,
        origin_x: float,
        spatial_resolution_x: float,
        row_count: int,
        origin_y: float,
        spatial_resolution_y: float,
        spatial_reference: osr.SpatialReference = gf.SR_WGS_84,
        overwrite: bool = False,
        metadata: Optional[Dict[str, str]] = None,
    ) -> nc.Dataset:
        """
        Create and open the netcdf file in a write mode
        Raised exception : OSError when file is not writable
        """
        if not os.path.exists(self.dtm_file.file_path) or overwrite:
            self._dataset = open_nc_file(self.dtm_file.file_path, mode="w")
            self.dtm_file.col_count = col_count
            self.dtm_file.west = origin_x
            self.dtm_file.spatial_resolution_x = spatial_resolution_x
            self.dtm_file.row_count = row_count
            self.dtm_file.south = origin_y
            self.dtm_file.spatial_resolution_y = spatial_resolution_y
            self.dtm_file.spatial_reference = spatial_reference
            self.initialize_file(metadata)
            return self.dataset
        else:
            raise FileExistsError(
                "File already exists and overwrite not allowed (allow overwrite with option : '-ow --overwrite)"
            )

    def close(self) -> None:
        """Close the dataset if opened"""
        if self.dataset and self.dataset.isopen():
            self.dataset.close()
        self._dataset = None

    def initialize_file(self, metadata: Optional[Dict[str, str]] = None) -> None:
        """
        Create metadada, dimensions and longitude/latitude variables
        Raised exception : IOError when file is not writable
        """
        self.create_metadata({} if metadata is None else metadata)
        self.create_dimension()
        self.create_grid_mapping_variables()

    def create_dimension(self) -> None:
        """
        Add dimension variables to the netcdf dataset
        """
        if self.dtm_file.spatial_reference.IsProjected():
            self.dataset.createDimension(DtmConstants.DIM_ORDINATE, self.dtm_file.row_count)
            self.dataset.createDimension(DtmConstants.DIM_ABSCISSA, self.dtm_file.col_count)
        else:
            self.dataset.createDimension(DtmConstants.DIM_LAT, self.dtm_file.row_count)
            self.dataset.createDimension(DtmConstants.DIM_LON, self.dtm_file.col_count)

    def __contains__(self, layer_name: str) -> bool:
        """return True if the DTM contains the layer_name"""
        return self.dataset.variables.__contains__(layer_name)  # pylint:disable=no-member

    def __getitem__(self, layer_name: str) -> nc.Variable:
        """return the layer called layer_name"""
        if layer_name != DtmConstants.VALUE_COUNT or DtmConstants.VALUE_COUNT in self:
            return self.dataset[layer_name]

        self.logger.debug(f"{DtmConstants.VALUE_COUNT} required but absent. Generates one")
        # value_count required but absent : generates one.
        elevation_reference = self.dataset[DtmConstants.ELEVATION_NAME]
        value_count = np.full_like(
            elevation_reference, get_missing_value(DtmConstants.VALUE_COUNT), dtype=get_type(DtmConstants.VALUE_COUNT)
        )
        i_data = elevation_reference[:].data
        m_val = elevation_reference._FillValue
        self.fill_default_layer_buffer(layer_name, value_count, i_data, invalid_value=m_val)
        return value_count

    def get_x_axis(self) -> nc.Variable:
        """return the layer containing the value of the columns (DIM_LON or DIM_ABSCISSA]"""
        if self.dtm_file.spatial_reference.IsProjected():
            if DtmConstants.DIM_ABSCISSA in self:
                return self[DtmConstants.DIM_ABSCISSA]
            return self[DtmConstants.DIM_LON]
        else:
            if DtmConstants.DIM_LON in self:
                return self[DtmConstants.DIM_LON]
            return self[DtmConstants.DIM_ABSCISSA]

    def get_y_axis(self) -> nc.Variable:
        """return the layer containing the value of the rows (DIM_LAT or DIM_ORDINATE]"""
        if self.dtm_file.spatial_reference.IsProjected():
            if DtmConstants.DIM_ORDINATE in self:
                return self[DtmConstants.DIM_ORDINATE]
            return self[DtmConstants.DIM_LAT]
        else:
            if DtmConstants.DIM_LAT in self:
                return self[DtmConstants.DIM_LAT]
            return self[DtmConstants.DIM_ORDINATE]

    def get_layers(self) -> Dict[str, nc.Variable]:
        """
        return the dictionary of all layers
        """
        return self.dataset.variables

    def add_layer(
        self, layer_name: str, data: np.ndarray = None, layer_type: Any = None, fill_value: Any = None
    ) -> nc.Variable:
        # pylint:disable=unsupported-membership-test
        layer_type = get_type(layer_name) if layer_type is None else layer_type
        fill_value = get_missing_value(layer_name) if fill_value is None else fill_value

        if layer_name == DtmConstants.CDI:
            if DtmConstants.DIM_CDI not in self.dataset.dimensions:
                self.dataset.createDimension(DtmConstants.DIM_CDI, size=None)
            layer = self.dataset.createVariable(layer_name, layer_type, dimensions=DtmConstants.DIM_CDI, fill_value="")
        elif DtmConstants.DIM_LAT in self.dataset.dimensions and DtmConstants.DIM_LON in self.dataset.dimensions:
            # WGS84, dtm not projected : dimensions are lon and lat
            layer = self.dataset.createVariable(
                layer_name,
                layer_type,
                (DtmConstants.DIM_LAT, DtmConstants.DIM_LON),
                fill_value=fill_value,
                compression=nc_util.DEFAULT_COMPRESSION_LIB,
            )
            layer.coordinates = f"{DtmConstants.DIM_LAT} {DtmConstants.DIM_LON}"
        elif (
            DtmConstants.DIM_ORDINATE in self.dataset.dimensions
            and DtmConstants.DIM_ABSCISSA in self.dataset.dimensions
        ):
            # dtm projected : dimensions are x and y
            layer = self.dataset.createVariable(
                layer_name,
                layer_type,
                (DtmConstants.DIM_ORDINATE, DtmConstants.DIM_ABSCISSA),
                fill_value=fill_value,
                compression=nc_util.DEFAULT_COMPRESSION_LIB,
            )
        else:
            raise ValueError(f"Can't create layer {layer_name} : no dimension specified")

        if data is not None:
            layer[:] = data

        if layer_name in LAYER_CONFIGURATOR_FUNCTIONS:
            LAYER_CONFIGURATOR_FUNCTIONS[layer_name](layer)

        return layer

    def create_missing_layer(self, layer_name: str, elevation_reference: nc.Dataset) -> None:
        """
        Create a default layer in the given dataset and fill it with default values depending of the layer
        Dimensions are copied from the input dataset
        This function requires that at least the Elevation layer exists in the input dataset
        Arguments:
               name {str} -- Name of the layer.
               elevation_reference {nc.Dataset} -- the elevation reference used to compute a validity mask for layers.
               output_dataset {nc.Dataset} -- output nc file.
        """
        if layer_name in self:
            # layer already exists do nothing
            return
        self.add_layer(layer_name)

        # Initialisation
        o_data = self.dataset[layer_name][:].data
        i_data = elevation_reference[:].data
        m_val = elevation_reference._FillValue

        o_data = self.fill_default_layer_buffer(layer_name, o_data, i_data, invalid_value=m_val)
        if o_data is not None:
            self.dataset[layer_name][:] = o_data

    def prepare_data(self, layer_name: str) -> np.ndarray:
        """
        Utility method to create and initialize an array of data for the specified layer.
        The result is not added to the DTM file
        """
        return np.full(
            shape=(self.dtm_file.row_count, self.dtm_file.col_count),
            fill_value=get_missing_value(layer_name),
            dtype=get_type(layer_name),
        )

    # pylint:disable=consider-using-with
    def prepare_memmap_data(self, layer_name: str, layer_type: Any = None, fill_value: Any = None) -> np.ndarray:
        """
        Utility method to create and initialize memory-map to an array stored in a binary file on disk for the specified layer.
        The result is not added to the DTM file
        Perform a del statement on the resulting array to close and delete the temporary file
        """
        layer_type = get_type(layer_name) if layer_type is None else layer_type
        fill_value = get_missing_value(layer_name) if fill_value is None else fill_value

        map_file = tempfile.TemporaryFile(suffix=".memmap", prefix=layer_name)
        result = np.memmap(
            map_file, shape=(self.dtm_file.row_count, self.dtm_file.col_count), dtype=layer_type, mode="w+"
        )
        result.fill(fill_value)
        return result

    def fill_default_layer_buffer(
        self, layer_name: str, o_data: np.ndarray, elevation_values: np.ndarray, invalid_value
    ) -> Optional[np.ndarray]:
        """Fill a buffer with default values given the expected elevation values"""
        if layer_name in [DtmConstants.ELEVATION_MAX, DtmConstants.ELEVATION_MIN]:
            # if min max is missing, copy values from elevation layer
            o_data[:] = elevation_values[:]
            return o_data
        elif layer_name == DtmConstants.VALUE_COUNT:
            return nb.create_layer(o_data, elevation_values, invalid_value, mode=1)
        elif layer_name == DtmConstants.FILTERED_COUNT:
            return np.where(np.isnan(elevation_values), -1, 0)
        elif layer_name in [DtmConstants.STDEV, DtmConstants.INTERPOLATION_FLAG]:
            return nb.create_layer(o_data, elevation_values, invalid_value, mode=2)
        elif layer_name in [DtmConstants.CDI, DtmConstants.CDI_INDEX]:
            return None
        return None

    def create_grid_mapping_variables(self):
        """
        Add variables lat and lon or x and y to the netcdf dataset
        """
        if self.dtm_file.spatial_reference.IsProjected():
            self.__create_x_y_variables()
        else:
            self.__create_lon_lat_variables()
        self.create_crs_variable()

    def __create_lon_lat_variables(self):
        lon = self.add_variable(
            DtmConstants.LON_NAME,
            float,
            DtmConstants.DIM_LON,
            float("nan"),
            standard_name="longitude",
            long_name="longitude",
            units="degrees_east",
            axis="X",
            sdn_parameter_urn="SDN:P01::ALONZZ01",
            sdn_parameter_name="Longitude east",
            sdn_uom_urn="SDN:P06::DEGE",
            sdn_uom_name="Degrees east",
            _CoordinateAxisType="Lon",
        )
        lon[:] = self.dtm_file.compute_x_axis()

        lat = self.add_variable(
            DtmConstants.LAT_NAME,
            float,
            DtmConstants.DIM_LAT,
            float("nan"),
            standard_name="latitude",
            long_name="latitude",
            units="degrees_north",
            axis="Y",
            sdn_parameter_urn="SDN:P01::ALATZZ01",
            sdn_parameter_name="Latitude north",
            sdn_uom_urn="SDN:P06::DEGN",
            sdn_uom_name="Degrees north",
            _CoordinateAxisType="Lat",
        )
        lat[:] = self.dtm_file.compute_y_axis()

    def __create_x_y_variables(self):
        """
        Add variables lat and lon to the netcdf dataset
        """
        x = self.add_variable(
            DtmConstants.ABSCISSA_NAME,
            float,
            DtmConstants.DIM_ABSCISSA,
            None,
            long_name="x coordinate of projection",
            standard_name="projection_x_coordinate",
            units="m",
        )
        x[:] = self.dtm_file.compute_x_axis()

        y = self.add_variable(
            DtmConstants.ORDINATE_NAME,
            float,
            DtmConstants.DIM_ORDINATE,
            None,
            long_name="y coordinate of projection",
            standard_name="projection_y_coordinate",
            units="m",
        )
        y[:] = self.dtm_file.compute_y_axis()

    def create_crs_variable(self):
        spatial_ref = self.dtm_file.spatial_reference
        if spatial_ref and spatial_ref.IsProjected():
            self.__create_projected_crs_variable()
        else:
            self.__create_longitude_latitude_crs_variable()

    def __create_projected_crs_variable(self):
        spatial_reference = clean_mercator(self.dtm_file.spatial_reference)
        self.add_variable(
            DtmConstants.CRS_NAME,
            int,
            (),
            None,
            comment="see Appendix F of cf convention 1.7",
            crs_wkt=spatial_reference.ExportToPrettyWkt(),
            **nc_util.translate_spatial_reference(spatial_reference),
        )

    def __create_longitude_latitude_crs_variable(self):
        # create CRS variable
        #  A container variable storing information about the grid_mapping.
        #  by default for qgis/gdal do not try to define other  grid_mapping_name is
        #  All the attributes within a grid_mapping variable are described in
        #  http://cfconventions.org/Data/cf-conventions/cf-conventions-1.6/build/cf-conventions.html#grid-mappings-and-projections. For all the measurements based on WSG84, the default coordinate system used for GPS measurements, the values shown here should be used.
        self.add_variable(
            DtmConstants.CRS_NAME,
            int,
            (),
            None,
            grid_mapping_name="latitude_longitude",  # Latitude and longitude on the WGS 1984 datum
        )

    def add_variable(self, varname, datatype, dimensions, fill_value, **kwargs):
        """
        Add a variable to the netcdf dataset and its attributes (kwargs)
        """
        # Activate compression by default
        # deactivate compression for string variables
        if datatype is str:
            compression = None
        else:
            compression = nc_util.DEFAULT_COMPRESSION_LIB

        result = self.dataset.createVariable(
            varname, datatype, dimensions, fill_value=fill_value, compression=compression
        )
        self.__set_attributes(result, **kwargs)
        return result

    def __set_attributes(self, variable, **kwargs):
        """
        Set all attributes (kwargs) to the variable (or any other netcdf object)
        """
        for key, value in kwargs.items():
            variable.setncattr(key, value)

    def create_metadata(self, metadata: Dict[str, str]):
        """
        Add global attributes to the netcdf dataset
        """
        self.dataset.dtm_convention_version = "1.0"
        self.dataset.Conventions = "SeaDataNet_1.0 CF-1.7"

        self.dataset.title = metadata["title"] if "title" in metadata else "The EMODnet Grid"
        self.dataset.institution = (
            metadata["institution"]
            if "institution" in metadata
            else "On behalf of the EMODnet project, http://www.emodnet-bathymetry.eu/."
        )
        self.dataset.source = (
            metadata["source"]
            if "source" in metadata
            else "source of the data can be found in the dataset or in the documentation available from  http://www.emodnet-bathymetry.eu/"
        )
        self.dataset.history = (
            metadata["history"]
            if "history" in metadata
            else "Information on the development of the data set and the source data sets included in the grid can be found in the data set documentation available from http://www.emodnet-bathymetry.eu/"
        )
        self.dataset.references = (
            metadata["references"]
            if "references" in metadata
            else "WORK IN PROGRESS 2020 lastest release is DOI: 10.12770/18ff0d48-b203-4a65-94a9-5fd8b0ec35f6"
        )
        self.dataset.comment = (
            metadata["comment"]
            if "comment" in metadata
            else "The data in the EMODnet Grid should not be used for navigation or any purpose relating to safety at sea."
        )

    def create_cdi_reference_variable(self, cdis: list):
        """
        Add variables cdi reference to the netcdf dataset
        """
        # Remove useless cdis
        cdis = string_utils.trim_string_array(cdis)
        self.logger.info(f"Processing layer {DtmConstants.CDI}")
        self.dataset.createDimension(DtmConstants.DIM_CDI, size=None)  # create unlimited dimension
        cdi_variable = self.add_variable(
            DtmConstants.CDI,
            str,
            DtmConstants.DIM_CDI,
            "",
            long_name="ID of related CDI metadata record"
            " set complete with truncated Id = EDMO-code-provider_Local-CDI-Id",
        )

        for i, val in enumerate(cdis):
            cdi_variable[i] = val

    def create_interpolation_layer(self):
        """
        create a default interpolation layer, by default if layer does not exist, it is created and populated with
        missing value if depth is invalid or not_interpolated value otherwise
        """
        if not DtmConstants.INTERPOLATION_FLAG in self.dataset.variables.keys():  # pylint:disable=no-member
            # Create interpolation layer if not exist
            self.add_layer(DtmConstants.INTERPOLATION_FLAG)

            interpolation_values = self.dataset[DtmConstants.INTERPOLATION_FLAG][:].data
            elevation_values = self.dataset[DtmConstants.ELEVATION_NAME][:].data
            # now fill to zero = not_interpolated where bathymetry is valid
            interpolation_values[~np.isnan(elevation_values[:])] = 0
            self.dataset[DtmConstants.INTERPOLATION_FLAG][:] = interpolation_values[:]

    def update_elevation(self, new_elevations: np.ndarray, row_start: int = 0, col_start: int = 0) -> np.ndarray:
        """
        Update the DTM file with the interpolated array

        Args:
            new_elevations: Array of new elevations
            row_start: Starting row for the update (default: 0)
            col_start: Starting column for the update (default: 0)

        Returns:
            Boolean array indicating where elevations have been replaced
        """
        if row_start > 0 or col_start > 0:
            self.logger.info(f"Starting update elevation at row {row_start}, col {col_start}")

        elevation_var = self[DtmConstants.ELEVATION_NAME]

        # Determine the region to update
        row_end = row_start + new_elevations.shape[0]
        col_end = col_start + new_elevations.shape[1]

        # Define the slice for the region
        region_slice = (slice(row_start, row_end), slice(col_start, col_end))

        # Load only the concerned sub-region
        region_elevations = elevation_var[region_slice]

        # Compute modified values flag for the region
        # Value is marked as modified if not nan and nan in origin data
        flag_region = ~np.isnan(new_elevations) & region_elevations.mask

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"new_elevations shape: {new_elevations.shape}")
            self.logger.debug(f"region_elevations shape: {region_elevations.shape}")
            self.logger.debug(f"NaN in new_elevations: {np.sum(~np.isnan(new_elevations))}")
            self.logger.debug(f"NaN in region_elevations: {np.sum(np.isnan(region_elevations.data))}")
            self.logger.debug(f"flag_region sum: {np.sum(flag_region)}")
            self.logger.debug(
                f"region_elevations min/max: {np.nanmin(region_elevations)}/{np.nanmax(region_elevations)}"
            )
            self.logger.debug(f"region_elevations unique values: {np.unique(region_elevations)[:10]}")

        # Update elevations in the region (only modified values)
        if np.any(flag_region):
            self.logger.info(f"Updating layer '{DtmConstants.ELEVATION_NAME}'")
            region_elevations[flag_region] = new_elevations[flag_region]
            elevation_var[region_slice] = region_elevations

            # Update interpolation flag
            if DtmConstants.INTERPOLATION_FLAG in self:
                self.logger.info(f"Updating layer '{DtmConstants.INTERPOLATION_FLAG}'")
                interpolation = self[DtmConstants.INTERPOLATION_FLAG]
                interp_region = interpolation[region_slice].data
                interp_region[flag_region] = 1
                interpolation[region_slice] = interp_region

            # Update elevation_max
            if DtmConstants.ELEVATION_MAX in self:
                self.logger.info(f"Updating layer '{DtmConstants.ELEVATION_MAX}'")
                elevation_max = self[DtmConstants.ELEVATION_MAX]
                max_region = elevation_max[region_slice].data
                max_region[flag_region] = new_elevations[flag_region]
                elevation_max[region_slice] = max_region

            # Update elevation_min
            if DtmConstants.ELEVATION_MIN in self:
                self.logger.info(f"Updating layer '{DtmConstants.ELEVATION_MIN}'")
                elevation_min = self[DtmConstants.ELEVATION_MIN]
                min_region = elevation_min[region_slice].data
                min_region[flag_region] = new_elevations[flag_region]
                elevation_min[region_slice] = min_region

            # Update value_count
            if DtmConstants.VALUE_COUNT in self:
                self.logger.info(f"Updating layer '{DtmConstants.VALUE_COUNT}'")
                value_count = self[DtmConstants.VALUE_COUNT]
                count_region = value_count[region_slice].data
                count_region[flag_region] = 1
                value_count[region_slice] = count_region
        else:
            self.logger.info("No elevations to update in the specified region")

        # Create a flag array of full size for return value
        flag = np.zeros(elevation_var.shape, dtype=bool)
        flag[region_slice] = flag_region

        return flag

    def apply_mask(self, interpolated_array, mask):
        """
        apply a kml mask to the interpolated array
        """
        interpolated_masked_array = np.where(mask, interpolated_array, np.nan)
        elevation = self[DtmConstants.ELEVATION_NAME]
        elevation[:] = interpolated_masked_array[:]

        if DtmConstants.ELEVATION_MIN in self:
            elevation_min = self[DtmConstants.ELEVATION_MIN]
            elevation_min_values = elevation_min[:].data
            elevation_min[:] = np.where(mask, elevation_min_values, np.nan)

        if DtmConstants.ELEVATION_MAX in self:
            elevation_max = self[DtmConstants.ELEVATION_MAX]
            elevation_max_values = elevation_max[:].data
            elevation_max[:] = np.where(mask, elevation_max_values, np.nan)

        if DtmConstants.VALUE_COUNT in self:
            value_count = self[DtmConstants.VALUE_COUNT]
            value_count_values = value_count[:].data
            value_count[:] = np.where(mask, value_count_values, np.nan)


@contextmanager
def open_dtm(file_path: str, mode: str = "r") -> Generator[DtmDriver, None, None]:
    """
    Define a With Statement Context Managers for a DtmDriver
    Allow opening a DtmDriver in a With Statement
    """
    driver = DtmDriver(file_path)
    driver.open(mode)
    try:
        yield driver
    finally:
        driver.close()
