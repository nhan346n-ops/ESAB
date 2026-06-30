import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import netCDF4 as nc
import numpy as np
import pytechsas.sensor.constant_mapping as cm
import pytechsas.sensor.netcdf_constant as nc
import pytechsas.sensor.sensor_constant as sc
import xarray as xr
from geopy.distance import geodesic
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from pytechsas.sensor.quality_flag import (
    create_valid_quality_mask,
    filter_dataset_by_quality,
)

from pyat.sensor.slice.g3d_writer import Coords, Variables, write_vertical_textures
from pyat.sensor.slice.interpolation import interpolate

logger = logging.getLogger(__name__)


def export_files(
    i_paths: List[str],
    o_path: str,
    overwrite: bool = False,
    monitor: ProgressMonitor = DefaultMonitor,
    **kwargs,  # interpolation_algo, ...
) -> None:
    """
    Export NetCDF files to G3D.

    Args:
        i_paths: List of paths to input NetCDF files.
        o_path: Path to output CSV file.
        overwrite: If True, overwrites existing output file.
        monitor: Progress monitor for tracking export progress.
        **kwargs: Additional parameters passed to pyat export function (e.g., interpolation_algo).
    """
    monitor.begin_task("'Export to G3D'", 100)

    # Check if output file exists and prevent overwrite if not allowed
    if os.path.exists(o_path) and not overwrite:
        logger.warning("File %s already exists, export aborted.", o_path)
        return
    monitor.worked(10)

    # Delegate to pyat export function
    try:
        coords, variables = _convert_files(i_paths=i_paths, **kwargs)
        if len(variables) == 0:
            raise ValueError("No variable produced. Export aborted")
        monitor.worked(80)
        write_vertical_textures(coords, variables, o_path)
    except Exception as e:
        logger.error("Error during file conversion: %s", e)
    monitor.done()


SENSOR_VARS = cm.SENSOR_VAR_NAMES


@dataclass
class RawCoordinates:
    """Raw scattered coordinates before gridding."""

    latitudes: np.ndarray
    longitudes: np.ndarray
    depths: np.ndarray
    distances: np.ndarray


def _convert_files(
    i_paths: List[str],
    grid_length: int = 1200,
    grid_height: int = 800,
    interpolation_algo: str = "linear",
) -> Tuple[Coords, Variables]:
    """
    Convert multiple Sensor-netCDF files into a single gridded dataset.

    Workflow:
    1. Load and filter data from all NetCDF files
    2. Calculate cumulative distances along the transect
    3. Project scattered data onto a regular 2D grid
    4. Average multiple values in the same grid cell
    5. Interpolate missing values in the grid

    Args:
        i_paths: Paths to input NetCDF files
        grid_length: Number of grid cells along the horizontal (distance) axis
        grid_height: Number of grid cells along the vertical (depth) axis
        interpolation_algo: Interpolation algorithm

    Returns:
        coords: Geographic coordinates for each grid position
        variables: Dictionary of gridded variables, each with shape (grid_height, grid_length)
                  Both raw and interpolated versions are included (var and var_ib)
    """
    logger.info(f"Loading {len(i_paths)} NetCDF files...")

    # Load all files and extract raw data
    raw_coords_list, raw_variables_dict = _load_all_files(i_paths)

    # Combine coordinates from all files
    combined_raw_coords = _combine_raw_coordinates(raw_coords_list)
    logger.info(f"Total valid data points: {len(combined_raw_coords.distances)}")

    # Create regular grid
    distance_axis, depth_axis = _create_grid_axes(combined_raw_coords, grid_length, grid_height)

    # Identify variables to process
    variable_names = list(raw_variables_dict.keys())
    logger.info(f"Processing {len(variable_names)} variables: {variable_names}")

    # Grid all variables with interpolation
    gridded_variables = _grid_all_variables(
        raw_variables_dict,
        raw_coords_list,
        variable_names,
        distance_axis,
        depth_axis,
        interpolation_algo,
    )

    # Create output coordinates
    output_coords = _create_output_coords(combined_raw_coords, distance_axis)

    return output_coords, gridded_variables


def _compute_cumulative_distance(dataset: xr.Dataset) -> np.ndarray:
    """
    Compute cumulative distance along the transect using geodesic calculations.

    Args:
        dataset: Input dataset containing latitude and longitude variables

    Returns:
        Cumulative distance array in meters
    """
    lats = dataset[SENSOR_VARS.VAR_LATITUDE].values
    lons = dataset[SENSOR_VARS.VAR_LONGITUDE].values

    distances = np.zeros(len(lats), dtype=np.float64)
    for i in range(1, len(lats)):
        point_a = (float(lats[i - 1]), float(lons[i - 1]))
        point_b = (float(lats[i]), float(lons[i]))
        distances[i] = geodesic(point_a, point_b).meters

    return distances


def _load_all_files(
    file_paths: List[str],
) -> Tuple[List[RawCoordinates], Dict[str, List[np.ndarray]]]:
    """
    Load all NetCDF files and extract coordinates and variables.

    Args:
        file_paths: List of paths to NetCDF files

    Returns:
        raw_coords_list: List of RawCoordinates from each file
        variables_by_file: Dictionary mapping variable names to lists of arrays from each file
    """
    raw_coords_list: List[RawCoordinates] = []
    variables_by_file: Dict[str, List[np.ndarray]] = {}

    for idx, path in enumerate(file_paths):
        logger.info(f"Processing file {idx + 1}/{len(file_paths)}: {path}")
        try:
            with xr.open_dataset(path) as ds:
                # Drop NaN latitudes and longitudes
                mask = ds[SENSOR_VARS.VAR_LATITUDE].notnull() & ds[SENSOR_VARS.VAR_LONGITUDE].notnull()
                ds = ds.where(mask, drop=True)

                # Apply quality filter when the quality flag variable is present in the dataset
                logger.info("Applying quality flag filter to sensor data.")
                ds = filter_dataset_by_quality(ds)

                # Compute distances and add as coordinate
                cumulative_distance = _compute_cumulative_distance(ds)
                ds = ds.assign_coords(distance=([SENSOR_VARS.VAR_TIME], cumulative_distance))

                # Extract coordinates and validity mask
                raw_coords, valid_mask = _extract_raw_coordinates(ds)
                raw_coords_list.append(raw_coords)

                # Extract variable data
                file_variables = _extract_variable_data(ds, valid_mask)
                for var_name, var_data in file_variables.items():
                    if var_name not in variables_by_file:
                        variables_by_file[var_name] = []
                    variables_by_file[var_name].append(var_data)
        except PermissionError as e:
            logger.error("Permission error while accessing file %s: %s", path, e)
            raise
    return raw_coords_list, variables_by_file


def _extract_raw_coordinates(dataset: xr.Dataset) -> Tuple[RawCoordinates, np.ndarray]:
    """
    Extract and filter coordinate arrays from dataset.

    Args:
        dataset: Input xarray Dataset

    Returns:
        raw_coords: RawCoordinates object with filtered coordinates
        valid_mask: Boolean mask indicating valid data points
    """
    lats = dataset[SENSOR_VARS.VAR_LATITUDE].values
    lons = dataset[SENSOR_VARS.VAR_LONGITUDE].values
    depths = dataset[sc.SENSOR_VAR_DEPTH].values
    distances = dataset["distance"].values

    # Apply quality_flag filter if available
    if sc.SENSOR_VAR_QUALITY_FLAG in dataset:
        valid_mask = create_valid_quality_mask(
            quality_flags_var_name=sc.SENSOR_VAR_QUALITY_FLAG, quality_flags=dataset[sc.SENSOR_VAR_QUALITY_FLAG].values
        )
        num_invalid = np.sum(~valid_mask)
        logger.info(f"Filtered {num_invalid} invalid points")

        raw_coords = RawCoordinates(
            latitudes=lats[valid_mask],
            longitudes=lons[valid_mask],
            depths=depths[valid_mask],
            distances=distances[valid_mask],
        )
    else:
        valid_mask = np.ones(len(lats), dtype=bool)
        raw_coords = RawCoordinates(
            latitudes=lats,
            longitudes=lons,
            depths=depths,
            distances=distances,
        )

    return raw_coords, valid_mask


def _extract_variable_data(dataset: xr.Dataset, valid_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract sensor variable data from dataset, applying validity mask.

    Args:
        dataset: Input xarray Dataset
        valid_mask: Boolean mask for filtering invalid data

    Returns:
        Dictionary mapping variable names to filtered 1D arrays
    """
    excluded_vars = {
        SENSOR_VARS.VAR_LATITUDE,
        SENSOR_VARS.VAR_LONGITUDE,
        sc.SENSOR_VAR_DEPTH,
        nc.VAR_CRS,
        sc.SENSOR_VAR_QUALITY_FLAG,
    }

    result: Dict[str, np.ndarray] = {}
    for var_name in dataset.data_vars.keys():
        is_floating_type = np.issubdtype(dataset[var_name].dtype, np.floating)
        if is_floating_type and var_name not in excluded_vars:
            var_data = dataset[var_name].values
            result[var_name] = var_data[valid_mask]

    return result


def _combine_raw_coordinates(coords_list: List[RawCoordinates]) -> RawCoordinates:
    """
    Combine raw coordinates from multiple files into single arrays.

    Args:
        coords_list: List of RawCoordinates from each file

    Returns:
        Combined RawCoordinates with cumulative distances
    """
    combined_lats = np.concatenate([c.latitudes for c in coords_list])
    combined_lons = np.concatenate([c.longitudes for c in coords_list])
    combined_depths = np.concatenate([c.depths for c in coords_list])

    # Cumulative sum for distances across files
    all_distances = np.concatenate([c.distances for c in coords_list])
    cumulative_distances = np.cumsum(all_distances)

    return RawCoordinates(
        latitudes=combined_lats,
        longitudes=combined_lons,
        depths=combined_depths,
        distances=cumulative_distances,
    )


def _create_grid_axes(
    raw_coords: RawCoordinates,
    grid_length: int,
    grid_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create regular grid axes based on data extent.

    Args:
        raw_coords: Combined raw coordinates from all files
        grid_length: Number of points along distance axis
        grid_height: Number of points along depth axis

    Returns:
        distance_axis: 1D array of distance values (length: grid_length)
        depth_axis: 1D array of depth values (length: grid_height)

    Raises:
        ValueError: If computed grid extent is degenerate
    """
    # Filter out NaN values for extent calculation
    valid_mask = ~(np.isnan(raw_coords.distances) | np.isnan(raw_coords.depths))

    distance_min = float(raw_coords.distances[valid_mask].min())
    distance_max = float(raw_coords.distances[valid_mask].max())
    depth_min = float(raw_coords.depths[valid_mask].min())
    depth_max = float(raw_coords.depths[valid_mask].max())

    distance_axis = np.linspace(distance_min, distance_max, grid_length)
    depth_axis = np.linspace(depth_min, depth_max, grid_height)

    logger.info(
        f"Grid extent: distance=[{distance_axis[0]:.1f}, {distance_axis[-1]:.1f}] m, depth=[{depth_axis[0]:.1f}, {depth_axis[-1]:.1f}] m"
    )

    if distance_axis[0] == distance_axis[-1] or depth_axis[0] == depth_axis[-1]:
        raise ValueError("Grid extent is degenerate (min equals max)")

    return distance_axis, depth_axis


def _grid_all_variables(
    variables_by_file: Dict[str, List[np.ndarray]],
    raw_coords_list: List[RawCoordinates],
    variable_names: List[str],
    distance_axis: np.ndarray,
    depth_axis: np.ndarray,
    interpolation_algo: str,
) -> Variables:
    """
    Grid all variables onto regular 2D grid with averaging and interpolation.

    Args:
        variables_by_file: Dictionary mapping variable names to lists of arrays from files
        raw_coords_list: List of raw coordinates from each file
        variable_names: Names of variables to process
        distance_axis: Distance grid axis
        depth_axis: Depth grid axis
        interpolation_algo: Interpolation algorithm

    Returns:
        Dictionary of gridded variables with shape (grid_height, grid_length)
        Includes both raw (var_name) and interpolated (var_name_ib) versions
    """
    gridded_vars = Variables(variable_names)
    combined_raw_coords = _combine_raw_coordinates(raw_coords_list)

    for var_name in variable_names:
        logger.info(f"Gridding variable: {var_name}")

        # Combine all file data for this variable
        combined_values = np.concatenate(variables_by_file[var_name])

        # Project onto grid with cell averaging
        grid_mean = _project_onto_grid(
            combined_raw_coords.distances,
            combined_raw_coords.depths,
            combined_values,
            distance_axis,
            depth_axis,
        )

        # Store raw and interpolated versions
        gridded_vars[var_name] = grid_mean
        gridded_vars[f"{var_name}_i"] = interpolate(grid_mean, interpolation_algo)

    return gridded_vars


def _project_onto_grid(
    distances: np.ndarray,
    depths: np.ndarray,
    values: np.ndarray,
    distance_axis: np.ndarray,
    depth_axis: np.ndarray,
) -> np.ndarray:
    """
    Project scattered data points onto a regular 2D grid with cell averaging.

    For each data point:
    1. Find the nearest grid cell
    2. Accumulate values in that cell
    3. Average multiple values in the same cell

    Args:
        distances: Distance values of data points
        depths: Depth values of data points
        values: Data values to project
        distance_axis: Regular distance grid
        depth_axis: Regular depth grid

    Returns:
        2D grid array with shape (len(depth_axis), len(distance_axis))
        Cells with no data contain NaN
    """
    n_depth, n_distance = len(depth_axis), len(distance_axis)

    # Initialize accumulation arrays
    grid_sum = np.full((n_depth, n_distance), np.nan, dtype=np.float64)
    grid_count = np.zeros((n_depth, n_distance), dtype=np.int64)

    # Find nearest grid indices for all points (vectorized)
    depth_indices = np.abs(depths[:, None] - depth_axis[None, :]).argmin(axis=1)
    distance_indices = np.abs(distances[:, None] - distance_axis[None, :]).argmin(axis=1)

    # Accumulate values into grid cells
    for i, value in enumerate(values):
        if np.isnan(value):
            continue

        depth_idx = int(depth_indices[i])
        dist_idx = int(distance_indices[i])

        if np.isnan(grid_sum[depth_idx, dist_idx]):
            grid_sum[depth_idx, dist_idx] = value
            grid_count[depth_idx, dist_idx] = 1
        else:
            grid_sum[depth_idx, dist_idx] += value
            grid_count[depth_idx, dist_idx] += 1

    # Compute cell averages
    cells_with_data = grid_count > 0
    grid_sum[cells_with_data] /= grid_count[cells_with_data]

    return grid_sum


def _create_output_coords(
    raw_coords: RawCoordinates,
    distance_axis: np.ndarray,
) -> Coords:
    """
    Create output Coords object with geographic positions for each grid point.

    For each grid point along the distance axis:
    - Find the nearest raw data point
    - Extract its latitude and longitude
    - Compute elevation range (surface to bottom)

    Args:
        raw_coords: Combined raw coordinates from all files
        distance_axis: Distance values for the grid

    Returns:
        Coords object with arrays of length grid_length
    """
    # Filter valid coordinates
    valid_mask = ~(
        np.isnan(raw_coords.distances)
        | np.isnan(raw_coords.latitudes)
        | np.isnan(raw_coords.longitudes)
        | np.isnan(raw_coords.depths)
    )

    clean_distances = raw_coords.distances[valid_mask]
    clean_lats = raw_coords.latitudes[valid_mask]
    clean_lons = raw_coords.longitudes[valid_mask]
    clean_depths = raw_coords.depths[valid_mask]

    # Find closest raw point for each grid point
    nearest_indices = np.abs(clean_distances[:, None] - distance_axis[None, :]).argmin(axis=0)

    # Extract coordinates at grid positions
    grid_latitudes = clean_lats[nearest_indices].astype(np.float64)
    grid_longitudes = clean_lons[nearest_indices].astype(np.float64)

    # Compute elevation range (negative depths for elevation above sea level)
    surface_elevation = -float(clean_depths.min())
    bottom_elevation = -float(clean_depths.max())

    return Coords(
        latitudes=grid_latitudes,
        longitudes=grid_longitudes,
        min_elevation=surface_elevation,
        max_elevation=bottom_elevation,
    )
