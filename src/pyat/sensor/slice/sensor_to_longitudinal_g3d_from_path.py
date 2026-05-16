import logging
import math
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pytechsas.sensor.sensor_constant as sc
import xarray as xr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from pytechsas.sensor.quality_flag import filter_dataset_by_quality

from pyat.sensor.slice.longitudinal_section_gridder import LongitudinalSectionGridder
from pyat.sensor.slice.sensor_utils import read_elevation_range
from pyat.sensor.slice.slice_path import SlicePath
from pyat.utils.kml_splitter import extract_polylines_from_kml

logger = logging.getLogger(__name__)


@dataclass
class LongitudinalSectionArgs:
    """
    Configuration arguments for longitudinal section processing.

    This class encapsulates all parameters needed to configure the sensor data
    slicing and gridding process. See pyat/app/emodnet/conf/compute_quality_indicator.json
    for configuration examples.

    Attributes:
        i_paths: list of input sensor file paths
        o_path: output G3D file path
        path: KML or Shapefile containing the path positions
        width: width around the path in meters (default: 1.0)
        grid_length: number of cells in longitudinal direction (default: 800)
        delta_along: longitudinal cell size in meters (0 means use grid_length)
        grid_height: number of cells in vertical direction (default: 100)
        delta_elevation: vertical cell size in meters (0 means use grid_height)
        interpolation_algo: interpolation algorithm ("linear" or "None")
        layers: list of layer names to export
        overwrite: whether to overwrite existing output files
        monitor: progress monitor for tracking execution
    """

    # Input sensor file list
    i_paths: List[str]
    # Output G3D file
    o_path: str

    # KML or Shapefile with the path positions
    path: str

    # Width around the path (m)
    width: float = 1.0

    # Longitudinal resolution defined by the number of cells
    grid_length: int = 800
    # Longitudinal resolution defined by the cell size
    delta_along: float = 0.0

    # Vertical resolution defined by the number of cells
    grid_height: int = 100
    # Vertical resolution defined by the cell size
    delta_elevation: float = 0.0

    # Algo to interpolate and fill gaps
    interpolation_algo: str = "None"

    # Layer to export
    layers: List[str] = field(default_factory=list)

    overwrite: bool = False
    monitor: ProgressMonitor = DefaultMonitor


def slice_sensor_files(**kwargs) -> None:
    """
    Process sensor files to create longitudinal sections.

    This is a convenience function that accepts configuration as keyword arguments
    and delegates to the main processing function.

    Args:
        **kwargs: configuration arguments matching LongitudinalSectionArgs attributes
    """
    args = LongitudinalSectionArgs(**kwargs)
    slice_with_longitudinal_section_args(args)


def slice_with_longitudinal_section_args(args: LongitudinalSectionArgs) -> None:
    """
    Process sensor files using provided configuration arguments.

    This function orchestrates the entire processing pipeline:
    1. Read the slice path from KML/Shapefile
    2. Process each input file or merge multiple files
    3. Generate gridded G3D output

    Args:
        args: configuration arguments for the processing
    """
    args.monitor.begin_task("'Longitudinal slicing'", 110)

    logger.info("Reading path file")
    if not os.path.exists(args.path):
        logger.error("Path file not found %s", args.path)
        return

    try:
        slice_path = _read_slice_path(args)
        if not slice_path:
            return
        args.monitor.worked(10)

        _slice_sensor_files(
            slice_path=slice_path,
            args=args,
            monitor=args.monitor.split(100),
        )
    except Exception as e:
        logger.error("Error during file conversion: %s", e)

    args.monitor.done()


def _slice_sensor_files(slice_path: SlicePath, args: LongitudinalSectionArgs, monitor: ProgressMonitor) -> None:
    """
    Process sensor files and generate a G3D output file.

    This function handles the core processing pipeline:
    1. Load sensor data
    2. Create and initialize gridder
    3. Fill gridder with sensor values
    4. Finalize and write output

    Args:
        slice_path: geographic path for the slice
        args: processing configuration
        monitor: progress monitor
    """
    sensor_data: List[xr.Dataset] = []
    try:
        monitor.begin_task(f"Generating {args.o_path}", 100)
        # Check if output file exists
        if os.path.exists(args.o_path) and not args.overwrite:
            logger.error(f"File {args.o_path} already exists. Use overwrite=True to overwrite")
            return

        logger.info(f"Loading {len(args.i_paths)} sensor file(s) and applying quality flag filter")
        try:
            for path in args.i_paths:
                ds = xr.open_dataset(path)
                # Apply quality filter when the quality flag variable is present in the dataset
                ds = filter_dataset_by_quality(ds)
                sensor_data.append(ds)
        except PermissionError as e:
            logger.error("Permission error while accessing file %s: %s", path, e)
            raise

        monitor.worked(10)

        # Create gridder with appropriate dimensions
        gridder = _create_gridder(sensor_data, slice_path, args)
        if gridder is None:
            return

        monitor.worked(10)

        # Initialize gridder with path positions
        _prepare_gridder(slice_path=slice_path, gridder=gridder)
        monitor.worked(10)

        # Fill gridder with sensor data
        _fill_gridder(sensor_data, gridder, args=args, monitor=monitor.split(30))

        # Finalize gridder (interpolation, etc.)
        interpolation_method = args.interpolation_algo if args.interpolation_algo != "None" else None
        gridder.finalize(monitor=monitor.split(30), interpolation_method=interpolation_method)

        gridder.generate_g3d_file(args.o_path)

    finally:
        monitor.done()
        for ds in sensor_data:
            ds.close()


def _read_slice_path(args: LongitudinalSectionArgs) -> SlicePath | None:
    """
    Read slice path from KML or Shapefile.

    Args:
        args: configuration arguments containing path file location

    Returns:
        SlicePath object if successful, None otherwise
    """
    try:
        paths = extract_polylines_from_kml(args.path)

        if not paths:
            logger.error(f"No polylines found in path file: {args.path}")
            return None

        if len(paths) > 1:
            logger.warning("Multiple polylines found, using the first one")

        # Extract coordinates from first polyline
        positions = paths[0]
        latitudes = np.array([latitude for latitude, _ in positions])
        longitudes = np.array([longitude for _, longitude in positions])

        result = SlicePath(latitudes=latitudes, longitudes=longitudes)
        logger.info(f"Loaded path with {len(positions)} positions, total length: {result.length:.1f} m")
        return result

    except Exception as e:
        logger.error(f"Failed to read path file {args.path}: {e}")
        return None


def _create_gridder(
    sensor_data: List[xr.Dataset], slice_path: SlicePath, args: LongitudinalSectionArgs
) -> LongitudinalSectionGridder | None:
    """
    Create a gridder with dimensions appropriate for the sensor data and path.

    This function computes the grid dimensions based on:
    - The slice path length (for longitudinal dimension)
    - The elevation range in the sensor data (for vertical dimension)
    - User-specified resolution parameters

    Args:
        sensor_data: list of sensor datasets
        slice_path: geographic path for the slice
        args: processing configuration

    Returns:
        initialized LongitudinalSectionGridder, or None if creation fails
    """
    # Compute longitudinal dimension
    col_count = args.grid_length
    if args.delta_along > 0.0:
        col_count = math.ceil(slice_path.length / args.delta_along)
        logger.info("Computed grid length : %d", col_count)

    if col_count <= 0:
        logger.error("Invalid grid length : %d", col_count)
        return None

    # Read elevation range from sensor data
    elevation_range = read_elevation_range(sensor_data)
    if not elevation_range:
        return None
    min_elevation, max_elevation = elevation_range

    # Compute vertical dimension
    row_count = args.grid_height
    delta_elevation = args.delta_elevation
    if delta_elevation > 0.0:
        # Compute row count from delta_elevation
        row_count = int(math.ceil((max_elevation - min_elevation) / delta_elevation))
        logger.info(f"Computed grid height from delta_elevation: {row_count} cells")
    else:
        # Compute delta_elevation from row count
        delta_elevation = (max_elevation - min_elevation) / row_count
        logger.info(f"Computed elevation step from grid height: {delta_elevation:.3f} m")

    if row_count <= 0:
        logger.error(f"Invalid grid height: {row_count}")
        return None

    # Validate layers
    if not args.layers:
        logger.error("No layers specified for export")
        return None

    # Create gridder
    return LongitudinalSectionGridder(
        x_count=col_count,
        z_count=row_count,
        min_elevation=min_elevation,
        max_elevation=max_elevation,
        delta_elevation=delta_elevation,
        layers=args.layers,
    )


def _prepare_gridder(slice_path: SlicePath, gridder: LongitudinalSectionGridder) -> None:
    """
    Initialize the gridder with reference positions along the slice path.

    This function:
    1. Initializes grid data arrays
    2. Resamples the path to match the grid resolution
    3. Assigns geographic coordinates to each grid column

    Args:
        slice_path: geographic path for the slice
        gridder: gridder to initialize
    """
    gridder.initialize_grid()

    # Resample path to match grid resolution
    resampled_slice_path = slice_path.resample(gridder.col_count)
    # For debug : export resampled path for visualization
    # resampled_slice_path.write_to_geojson(r"d:\temp\resampled_slice_path.geojson")

    # Assign coordinates to each grid column
    for longitudinal_index, (lon, lat) in enumerate(
        zip(resampled_slice_path.latitudes, resampled_slice_path.longitudes)
    ):
        gridder.add(lon=lon, lat=lat, x_idx=longitudinal_index)


def _fill_gridder(
    sensor_data: List[xr.Dataset],
    gridder: LongitudinalSectionGridder,
    args: LongitudinalSectionArgs,
    monitor: ProgressMonitor,
) -> None:
    """
    Fill the gridder with values from sensor datasets.

    This function processes each sensor dataset and extracts values for the
    requested layers, then adds them to the gridder.

    Args:
        sensor_data: list of sensor datasets to process
        gridder: gridder to fill with data
        args: processing configuration containing layer names
    """

    monitor.begin_task("'Filling'", len(sensor_data) * len(args.layers))
    total_points_processed = 0

    for dataset_idx, sensor_ds in enumerate(sensor_data):
        # Check for required variables
        if sc.SENSOR_VAR_DEPTH not in sensor_ds.data_vars:
            logger.warning(f"Dataset {dataset_idx} missing depth variable, skipping")
            continue

        if sc.SENSOR_VAR_LONGITUDE not in sensor_ds.data_vars or sc.SENSOR_VAR_LATITUDE not in sensor_ds.data_vars:
            logger.warning(f"Dataset {dataset_idx} missing position variables, skipping")
            continue

        longitudes = sensor_ds[sc.SENSOR_VAR_LONGITUDE].values
        latitudes = sensor_ds[sc.SENSOR_VAR_LATITUDE].values
        elevations = -sensor_ds[sc.SENSOR_VAR_DEPTH].values

        # Process each requested layer
        for layer in args.layers:
            if layer not in sensor_ds.data_vars:
                logger.warning(f"Layer '{layer}' not found in dataset {dataset_idx}, skipping")
                continue

            # Extract layer values
            values = sensor_ds[layer].values

            # Validate array shapes
            if values.shape != longitudes.shape:
                logger.warning(
                    f"Shape mismatch for layer '{layer}': values={values.shape}, positions={longitudes.shape}"
                )
                continue

            # Fill gridder
            try:
                if gridder.fill_grid(
                    layer=layer,
                    longitudes=longitudes,
                    latitudes=latitudes,
                    elevations=elevations,
                    values=values,
                    max_distance=args.width / 2.0,
                ):
                    # Count valid points
                    valid_points = np.sum(~np.isnan(values))
                    total_points_processed += valid_points
                    logger.debug(f"Processed {valid_points} points for layer '{layer}' in dataset {dataset_idx}")
                else:
                    logger.warning(
                        f"The file '{sensor_ds.encoding.get('source')}' is completely outside the processing area and will be ignored"
                    )
                    break

            except Exception as e:
                logger.error(f"Error filling gridder for layer '{layer}': {e}")
            monitor.worked(1)

    logger.info(f"Total points processed: {total_points_processed}")
    monitor.done()
