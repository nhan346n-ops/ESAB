"""
G3D NetCDF File Writer

This module provides functionality for exporting gridded geospatial data to G3D NetCDF format.
"""

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from pygws.service.progress_monitor import ProgressMonitor

from pyat.utils.nc_encoding import open_nc_file
from pyat.utils.netcdf_utils import (
    create_crs_variable,
    create_latitude_variable,
    create_longitude_variable,
)

logger = logging.getLogger(__name__)


@dataclass
class Coords:
    """
    Geographic coordinates and elevation bounds for grid positions.

    This dataclass encapsulates the spatial reference information needed for
    G3D file creation, including horizontal coordinates and vertical extent.

    Attributes:
        latitudes: 1D array of latitude values in degrees North (shape: [n])
        longitudes: 1D array of longitude values in degrees East (shape: [n])
        min_elevation: Minimum elevation in meters (typically bottom/seafloor for underwater data)
        max_elevation: Maximum elevation in meters (typically surface for underwater data)

    """

    latitudes: np.ndarray  # 1D array of latitude coordinates (degrees North)
    longitudes: np.ndarray  # 1D array of longitude coordinates (degrees East)
    min_elevation: float  # Minimum elevation (meters, e.g., seafloor depth)
    max_elevation: float  # Maximum elevation (meters, e.g., sea surface)

    @property
    def length(self) -> int:
        """
        Get the number of coordinate positions.
        """
        return len(self.latitudes)


class Variables(dict):
    """
    Dictionary container for gridded variable data.

    Stores multiple data layers as numpy arrays, where each layer represents
    a physical variable (e.g., temperature, salinity, velocity). All arrays
    must have the same shape (height, length).
    """

    def __init__(self, layers: List[str]):
        """
        Initialize the variables container.
        """
        super().__init__()
        self.layers: List[str] = layers

    @property
    def height(self) -> int:
        """
        Get the height (number of rows) of the grid.
        """
        if not self:
            return 0  # No variables stored yet
        first_array = next(iter(self.values()))
        return first_array.shape[0]

    @property
    def length(self) -> int:
        """
        Get the length (number of columns) of the grid.
        """
        if not self:
            return 0  # No variables stored yet
        first_array = next(iter(self.values()))
        return first_array.shape[1]


def write_vertical_textures(coords: Coords, variables: Variables, file_path: str) -> None:
    """
    Export vertical texture data to a G3D NetCDF file (FlyTexture format).

    Args:
        coords: Geographic coordinates and elevation bounds for the profile
        variables: Dictionary of variable names to 2D numpy arrays (height × length)
        file_path: Output path for the G3D NetCDF file (e.g., "output.g3d")
    """
    logger.info(f"Writing vertical texture G3D file: {file_path}")

    try:
        with open_nc_file(file_path, mode="w") as dataset:

            # Set global attributes identifying this as a FlyTexture dataset
            dataset.dataset_type = "FlyTexture"
            dataset.history = "Created by PyAT with Sensor-netCDF files"

            # Create dimension for data layers and store layer names
            dataset.createDimension("datalayer_count", len(variables))
            datalayer_variable_name = dataset.createVariable("datalayer_variable_name", str, ("datalayer_count",))
            for index, layer in enumerate(variables):
                datalayer_variable_name[index] = layer

            # Create the main data group '001'
            grp = dataset.createGroup("001")

            # Define dimensions for the data grid
            grp.createDimension("height", variables.height)  # Vertical resolution
            grp.createDimension("length", variables.length)  # Horizontal resolution (along path)
            grp.createDimension("vector", 2)  # For top/bottom elevation pair
            grp.createDimension("position", coords.length)  # Number of waypoints

            # Create elevation coordinate variable (2 values per position: top and bottom)
            elevation = grp.createVariable("elevation", "f4", ("vector", "position"))
            elevation.units = "meters"
            elevation.long_name = "elevation"
            elevation.standard_name = "elevation"
            elevation[0, :] = coords.max_elevation  # Top elevation (e.g., sea surface)
            elevation[1, :] = coords.min_elevation  # Bottom elevation (e.g., seafloor)

            # Create longitude coordinate variable
            longitude = grp.createVariable("longitude", "f8", ("vector", "position"))
            longitude.units = "degrees_east"
            longitude.long_name = "longitude"
            longitude.standard_name = "longitude"
            longitude[:] = coords.longitudes

            # Create latitude coordinate variable
            latitude = grp.createVariable("latitude", "f8", ("vector", "position"))
            latitude.units = "degrees_north"
            latitude.long_name = "latitude"
            latitude.standard_name = "latitude"
            latitude[:] = coords.latitudes

            # Write all data variables
            logger.info(f"Writing {len(variables)} data variable(s)...")
            for var_name, values in variables.items():
                variable = grp.createVariable(var_name, "f4", ("height", "length"))
                variable[:] = values

    except OSError as e:
        logger.error(f"Failed to write G3D file: {file_path}")
        logger.error(f"OS error: {e}")
        logger.exception("Detailed error information:")
        raise


def write_horizontal_textures(
    coords: Coords,
    elevations: np.ndarray,
    layers: List[str],
    slice_count: int,
    values_provider: Callable[[str, int], Optional[np.ndarray]],
    file_path: str,
    monitor: ProgressMonitor,
) -> None:
    """
    Export horizontal texture data to a G3D NetCDF file (ElevationMappedTexture format).

    Creates a G3D file with horizontal slices at different elevation levels. Each slice
    is a 2D grid in geographic coordinates (lat/lon). This format is ideal for datasets
    that have been gridded at regular elevation intervals.

    Args:
        coords: Geographic coordinates and elevation bounds for the grid
        layers: List of data layer names (e.g., ['temperature', 'salinity'])
        slice_count: Number of horizontal slices (elevation levels)
        values_provider: Callback function(layer: str, grid_idx: int) -> np.ndarray
                        that returns the 2D grid for a specific layer and elevation index.
                        Should return None if data is unavailable.
        file_path: Output path for the G3D NetCDF file (e.g., "output.g3d")
    """
    monitor.begin_task("Generationg G3D", len(layers) + 1)
    max_elevation = np.nanmax(elevations)
    min_elevation = np.nanmin(elevations)
    elevation_range = max_elevation - min_elevation
    delta_elevation = elevation_range / slice_count

    try:
        with open_nc_file(file_path, mode="w") as dataset:

            # Set global attributes following CF and SeaDataNet conventions
            dataset.Conventions = "SeaDataNet_1.0 CF-1.7"
            dataset.history = "Created by PyAT with Sensor-netCDF files"
            dataset.dataset_type = "ElevationMappedTexture"
            dataset.dtm_convention_version = "1.0"

            # Store elevation metadata as global attributes
            dataset.max_elevation = max_elevation
            dataset.min_elevation = min_elevation
            dataset.slice_count = slice_count
            dataset.delta_elevation = delta_elevation

            # Create coordinate variables following CF conventions
            logger.debug("Creating coordinate variables...")
            create_longitude_variable(dataset, "lon", "lon", coords.longitudes)
            create_latitude_variable(dataset, "lat", "lat", coords.latitudes)
            create_crs_variable(dataset, "crs")

            # Create dimension for data layers and store layer names
            logger.debug("Creating variable datalayer_variable_name...")
            dataset.createDimension("datalayer_count", len(layers))
            datalayer_variable_name = dataset.createVariable("datalayer_variable_name", str, ("datalayer_count",))
            for index, layer in enumerate(layers):
                datalayer_variable_name[index] = layer

            # Create and write the elevation variable
            logger.info("Creating elevation variable...")
            variable = dataset.createVariable("elevation", "f4", ("lat", "lon"), fill_value=np.nan)
            variable.units = "meter"
            variable.long_name = "elevation"
            variable.standard_name = "elevation"
            variable.grid_mapping = "crs"  # Link to coordinate reference system
            variable[:] = elevations
            monitor.worked(1)

            for layer in layers:
                logger.info(f"Processing layer '{layer}' with {slice_count} elevation slices...")
                # Process elevation slices from top to bottom
                for elevation_idx in range(slice_count):
                    # Convert elevation index to grid index (inverted order)
                    # elevation_idx: 0 = top (highest), slice_count-1 = bottom (lowest)
                    # grid_idx: matches the storage order in the gridder
                    grid_idx = slice_count - elevation_idx - 1

                    # Calculate the actual depth for this slice (center of the slice)
                    vertical_offset = 0.0  # (grid_idx + 0.5) * delta_elevation

                    # Create variable name following G3D convention
                    # Format: {slice_number}_{layer_name}_(z={elevation})
                    var_name = f"{elevation_idx + 1}".zfill(3) + "_" + layer + f"_(z={np.round(vertical_offset, 2)})"

                    # Retrieve grid data from the provider callback
                    grid_data = values_provider(layer, slice_count - grid_idx - 1)

                    if grid_data is None:
                        logger.warning(
                            f"No data available for layer '{layer}', slice {elevation_idx + 1} (elevation {vertical_offset:.2f} m), skipping"
                        )
                        continue

                    # Create and write the variable
                    variable = dataset.createVariable(var_name, "f4", ("lat", "lon"), fill_value=np.nan)
                    variable.grid_mapping = "crs"  # Link to coordinate reference system
                    variable.vertical_offset = vertical_offset  # Store elevation as attribute
                    variable[:] = grid_data

                monitor.worked(1)

    except OSError as e:
        logger.error(f"Failed to write G3D file: {file_path}")
        logger.error(f"OS error: {e}")
        raise
