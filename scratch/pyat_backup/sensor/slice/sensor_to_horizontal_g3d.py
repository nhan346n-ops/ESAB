"""
Sensor to Horizontal G3D Converter

This module provides utilities to interpolate sensor data (stored as NetCDF files)
onto a regular geographic grid and export the result to the G3D format.
The horizontal section is defined by a geobox and a target spatial resolution.
"""

import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytechsas.sensor.sensor_constant as sc
import xarray as xr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from pytechsas.sensor.quality_flag import filter_dataset_by_quality
from scipy.interpolate import griddata

from pyat.sensor.slice.g3d_writer import Coords, write_horizontal_textures
from pyat.utils.argument_utils import parse_float, parse_geobox

logger = logging.getLogger(__name__)


@dataclass
class HorizontalSectionArgs:
    """
    Configuration arguments for horizontal section processing.

    This dataclass encapsulates all parameters needed to slice sensor data into
    horizontal sections and export to G3D format.

    Attributes:
        i_paths: List of paths to the input sensor NetCDF files to be combined.
        o_path: Path to the output G3D NetCDF file to be created.
        coord: Dictionary defining the spatial extent (geobox) with keys such as
            'left', 'right', 'lower', 'upper' expressed in decimal degrees.
        target_resolution: Target spatial resolution of the output grid, in decimal
            degrees. Defaults to ~1 arc-second (0.000277778°).
        layers: List of sensor variable names to interpolate and include in the
            output G3D file. An empty list means no additional layer beyond elevation.
        overwrite: If True, an existing output file at ``o_path`` will be replaced.
            Defaults to False.
        monitor: Progress monitor instance used to report processing progress.
            Defaults to ``DefaultMonitor``.
    """

    # List of input sensor NetCDF file paths to process
    i_paths: List[str]
    # Destination path for the output G3D NetCDF file
    o_path: str

    # Geobox dictionary defining the spatial extent (left, right, lower, upper in degrees)
    coord: Dict[str, float]
    # Spatial resolution of the output regular grid, expressed in decimal degrees
    target_resolution: float = 0.000277778

    # Sensor variable names to export as layers in the G3D file
    layers: List[str] = field(default_factory=list)

    # Allow overwriting an existing output file when True
    overwrite: bool = False
    # Progress monitor used to track and report task advancement
    monitor: ProgressMonitor = DefaultMonitor

    def check_arguments(self) -> bool:
        """
        Validate all configuration arguments before processing starts.

        Checks that:
        - The output file does not already exist (unless ``overwrite`` is True).
        - The ``coord`` geobox is present and can be parsed correctly.
        - The ``target_resolution`` is a positive float.

        Returns:
            bool: True if all arguments are valid and processing can proceed,
                False if one or more arguments are invalid.
        """
        result = True

        # Prevent accidental overwrite of an existing output file
        if os.path.exists(self.o_path) and not self.overwrite:
            logger.error(
                f"Output file already exists: {self.o_path}. " "Set overwrite=True to replace the existing file."
            )
            result = False

        # Ensure the geobox parameter is provided and syntactically valid
        if not self.coord:
            logger.error("Missing required 'coord' parameter defining the geobox.")
            result = False
        else:
            try:
                parse_geobox("coord", self.coord)
            except ValueError as err:
                logger.error(f"Invalid geobox specification: {err}")
                result = False

        # Ensure the target resolution can be parsed and is strictly positive
        try:
            self.target_resolution = parse_float("target_resolution", self.target_resolution)
            if self.target_resolution <= 0.0:
                logger.error(
                    f"Invalid target_resolution: {self.target_resolution}. " "Value must be strictly greater than 0."
                )
                result = False
        except ValueError as err:
            logger.error(f"Failed to parse target_resolution: {err}")
            result = False

        return result


def slice_sensor_files(**kwargs) -> None:
    """
    Process sensor files to create horizontal sections (convenience wrapper).
    """
    logger.info("Initializing horizontal section processing with provided arguments.")
    args = HorizontalSectionArgs(**kwargs)
    slice_with_horizontal_section_args(args)


def slice_with_horizontal_section_args(args: HorizontalSectionArgs) -> None:
    """
    Execute the full horizontal grid processing pipeline.

    Args:
        args: A fully populated :class:`HorizontalSectionArgs` instance.
    """
    # Abort early if configuration is invalid to avoid partial or corrupt outputs
    if not args.check_arguments():
        logger.error("Argument validation failed. Aborting processing.")
        return

    args.monitor.begin_task("'Horizontal slicing'", 100)
    try:
        logger.info(f"Opening {len(args.i_paths)} input sensor file(s).")
        with xr.open_mfdataset(args.i_paths, combine="by_coords") as sensor_ds:
            args.monitor.worked(5)

            # Apply quality filter when the quality flag variable is present in the dataset
            logger.info("Applying quality flag filter to sensor data.")
            sensor_ds = filter_dataset_by_quality(sensor_ds)
            args.monitor.worked(5)

            # Build the regular lon/lat grid onto which sensor data will be projected
            interpolation_points = _compute_grid(args)
            args.monitor.worked(5)

            # Interpolate the depth layer first; its values become the elevation reference
            # for the G3D file. We use a two-step process: 'linear' interpolation followed
            # by 'nearest' to fill any remaining gaps (NaNs).
            interpolated_elevations = _interpolate_layer(
                layer_name=sc.SENSOR_VAR_DEPTH,
                slice_index=0,
                ds=sensor_ds,
                interpolation_points=interpolation_points,
                method="linear",
            )
            if np.isnan(interpolated_elevations).any():
                logger.info("Filling remaining NaN values in elevation using 'nearest' interpolation.")
                nearest_elevations = _interpolate_layer(
                    layer_name=sc.SENSOR_VAR_DEPTH,
                    slice_index=0,
                    ds=sensor_ds,
                    interpolation_points=interpolation_points,
                    method="nearest",
                )
                interpolated_elevations = np.where(
                    np.isnan(interpolated_elevations), nearest_elevations, interpolated_elevations
                )
            args.monitor.worked(5)

            _generate_g3d_file(sensor_ds, interpolation_points, interpolated_elevations, args, args.monitor.split(80))

    except Exception as e:
        logger.error(f"Unexpected error during horizontal slicing: {e}", exc_info=True)
    finally:
        # Always signal task completion so the monitor is left in a clean state
        args.monitor.done()


def _generate_g3d_file(
    sensor_ds: xr.Dataset,
    interpolation_points: Tuple[np.ndarray, np.ndarray],
    interpolated_elevations: np.ndarray,
    args: HorizontalSectionArgs,
    monitor: ProgressMonitor,
) -> None:
    """
    Export gridded sensor data to a G3D NetCDF file.

    Args:
        sensor_ds: The sensor dataset opened from the input files.
        interpolation_points: Tuple of ``(lon_grid, lat_grid)`` 1-D arrays defining
            the regular output grid.
        interpolated_elevations: 2-D array of elevation values (in metres, positive up)
            already interpolated onto the output grid.
        args: Processing configuration, including the output path and layer list.
    """
    logger.info(f"Generating G3D file: {args.o_path}")
    longitudes, latitudes = interpolation_points

    # Collect spatial and vertical extent for the G3D coordinate metadata
    coords = Coords(
        latitudes=latitudes,
        longitudes=longitudes,
        min_elevation=np.nanmin(interpolated_elevations),
        max_elevation=np.nanmax(interpolated_elevations),
    )
    try:
        # Write elevation and all requested sensor layers into the G3D file.
        # ``values_provider`` is a callback invoked by the writer for each layer;
        # it receives ``layer_name`` and ``slice_index`` as positional arguments.
        write_horizontal_textures(
            coords=coords,
            elevations=interpolated_elevations,
            layers=args.layers,
            slice_count=1,
            values_provider=partial(_interpolate_layer, ds=sensor_ds, interpolation_points=interpolation_points),
            file_path=args.o_path,
            monitor=monitor,
        )
        logger.info(f"Successfully wrote G3D file: {args.o_path}")
    except Exception as e:
        logger.error(f"Failed to write G3D file '{args.o_path}': {e}", exc_info=True)
        raise


def _compute_grid(args: HorizontalSectionArgs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the regular longitude/latitude grid for interpolation.

    Args:
        args: Processing configuration providing the geobox (``coord``) and
            the ``target_resolution`` in decimal degrees.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A ``(lon_grid, lat_grid)`` tuple where
            each element is a 1-D array of evenly spaced coordinate values.
    """
    geobox = parse_geobox("coord", args.coord)

    # Build latitude array from southern to northern bound (inclusive)
    lat_grid = np.arange(geobox.lower, geobox.upper + args.target_resolution, args.target_resolution)
    # Build longitude array from western to eastern bound (inclusive)
    lon_grid = np.arange(geobox.left, geobox.right + args.target_resolution, args.target_resolution)

    return (lon_grid, lat_grid)


def _interpolate_layer(
    layer_name: str,
    slice_index: int,
    ds: xr.Dataset,
    interpolation_points: Tuple[np.ndarray, np.ndarray],
    method: str = "linear",
) -> Optional[np.ndarray]:
    """
    Interpolate a single sensor variable onto the regular output grid.

    Extracts the raw point-cloud values for ``layer_name`` from the dataset,
    optionally converts depth to elevation (sign flip), then uses
    ``scipy.interpolate.griddata`` to project the scattered sensor data onto
    the regular mesh defined by ``interpolation_points``.

    Args:
        layer_name: Name of the variable in ``ds`` to interpolate (e.g. depth,
            temperature, salinity).
        slice_index: Index of the slice to process.  Must be 0 for G3D output;
            any other value results in ``None`` being returned.
        ds: Quality-filtered sensor dataset containing the variable to interpolate
            as well as longitude and latitude coordinate variables.
        interpolation_points: Tuple of ``(lon_grid, lat_grid)`` 1-D arrays that
            define the target regular grid passed to ``np.meshgrid``.
        method: Interpolation method forwarded to ``griddata``.  One of
            ``'linear'``, ``'nearest'``, or ``'cubic'``.  Defaults to
            ``'linear'``.

    Returns:
        Optional[np.ndarray]: A 2-D NumPy array of shape
            ``(len(lat_grid), len(lon_grid))`` containing interpolated values,
            or ``None`` if the slice index is not 0 or the variable is absent
            from the dataset.  Missing values are filled with ``np.nan``.
    """
    logger.info("Interpolating layer '%s' (method='%s').", layer_name, method)

    if slice_index != 0:
        logger.warning(
            "slice_index=%d is not supported for G3D output (expected 0). Skipping layer '%s'.",
            slice_index,
            layer_name,
        )
        return None

    if layer_name not in ds.data_vars:
        logger.warning("Layer '%s' not found in dataset. Skipping interpolation.", layer_name)
        return None

    values = ds[layer_name].values

    # Convert depth (positive down, metres) to elevation (positive up, metres)
    if layer_name == sc.SENSOR_VAR_DEPTH:
        values = values * -1.0

    # Stack lon/lat into a (N, 2) array of scattered source points
    lon_lats = np.column_stack([ds[sc.SENSOR_VAR_LONGITUDE].values, ds[sc.SENSOR_VAR_LATITUDE].values])

    # Project scattered sensor points onto the regular mesh via griddata
    interpolated_values = griddata(
        points=lon_lats,
        values=values,
        xi=np.meshgrid(*interpolation_points),
        method=method,
        fill_value=np.nan,
    )

    return interpolated_values
