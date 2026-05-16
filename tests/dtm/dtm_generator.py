"""
Pytest fixture for generating test DTM files.

This module provides a fixture that creates representative DTM NetCDF files
for unit testing purposes, with configurable dimensions and data patterns.
"""

from typing import Callable, List

import numpy as np
import pytest
from osgeo import osr

from pyat.dtm.dtm_driver import DtmConstants, DtmDriver, get_missing_value

# ============================================================================
# PARAMETRIZED DTM FILE GENERATION FIXTURES
# ============================================================================


@pytest.fixture
def dtm_file_factory(tmp_path):
    """
    Factory fixture for creating parametrized DTM test files.

    Returns a callable that generates DTM files with customizable parameters.

    Usage:
        def test_something(dtm_file_factory):
            dtm_path = dtm_file_factory(
                grid_size=25,
                with_nodata=False,
                layers='full'
            )

    Args:
        grid_size (int): Size of the grid (default: 50)
        with_nodata (bool): Whether to include no-data zones (default: True)
        nodata_zones (list): Custom no-data zones as list of tuples (default: None)
        layers (str): 'full', 'simple', or 'custom' (default: 'full')
        custom_layers (list): List of layer names if layers='custom' (default: None)
        cell_size (float): Cell resolution in degrees (default: 0.001)
        origin_lon (float): Origin longitude (default: -5.0)
        origin_lat (float): Origin latitude (default: 45.0)
        projected (bool): Use UTM projection instead of WGS84 (default: False)
        metadata (dict): Custom metadata (default: None)

    Returns:
        callable: Function that creates DTM files with specified parameters
    """
    created_files = []

    def _create_dtm(
        grid_size=50,
        vertical_offset=0.0,
        with_nodata=True,
        nodata_zones=None,
        layers="full",
        custom_layers=None,
        cell_size=0.001,
        origin_lon=-5.0,
        origin_lat=45.0,
        projected=False,
        metadata=None,
        filename=None,
    ):
        """Create a DTM file with specified parameters."""

        # Generate filename
        if filename is None:
            filename = f"test_dtm_{len(created_files)}.nc"
        test_file = tmp_path / filename
        created_files.append(test_file)

        # Initialize driver
        driver = DtmDriver(str(test_file))

        # Create spatial reference
        spatial_ref = osr.SpatialReference()
        if projected:
            spatial_ref.ImportFromEPSG(32630)  # UTM Zone 30N
            origin_x = 500000
            origin_y = 5000000
            spatial_resolution = 100
        else:
            spatial_ref.ImportFromEPSG(4326)  # WGS84
            origin_x = origin_lon
            origin_y = origin_lat
            spatial_resolution = cell_size

        # Default metadata
        if metadata is None:
            metadata = {
                "title": f"Test DTM Grid {grid_size}x{grid_size}",
                "institution": "Test Institution",
                "source": "Generated for unit testing",
                "comment": "Test data - not for navigation",
            }

        # Create file
        driver.create_file(
            col_count=grid_size,
            origin_x=origin_x,
            spatial_resolution_x=spatial_resolution if not projected else 100,
            row_count=grid_size,
            origin_y=origin_y,
            spatial_resolution_y=spatial_resolution if not projected else 100,
            spatial_reference=spatial_ref,
            overwrite=True,
            metadata=metadata,
        )

        # Generate elevation data
        elevation_data = _generate_elevation_pattern(grid_size, vertical_offset)

        # Add no-data zones if requested
        if with_nodata:
            if nodata_zones is None:
                # Default no-data zones scaled to grid size
                nodata_zones = _get_default_nodata_zones(grid_size)
            elevation_data = _add_nodata_zones(elevation_data, nodata_zones)

        # Add elevation layer (always present)
        driver.add_layer(DtmConstants.ELEVATION_NAME, data=elevation_data)

        # Add additional layers based on configuration
        if layers == "full":
            _add_elevation_statistics_layers(driver, elevation_data)
            _add_count_layers(driver, elevation_data)
            _add_interpolated_layers(driver, elevation_data)
            _add_geometry_layers(driver, elevation_data)
            _add_metadata_layers(driver, elevation_data)
        elif layers == "statistics":
            _add_elevation_statistics_layers(driver, elevation_data)
        elif layers == "custom" and custom_layers:
            for layer_name in custom_layers:
                if layer_name in [DtmConstants.ELEVATION_MIN, DtmConstants.ELEVATION_MAX, DtmConstants.STDEV]:
                    _add_elevation_statistics_layers(driver, elevation_data)
                elif layer_name in [DtmConstants.VALUE_COUNT, DtmConstants.FILTERED_COUNT]:
                    _add_count_layers(driver, elevation_data)
                elif layer_name == DtmConstants.INTERPOLATION_FLAG:
                    _add_interpolated_layers(driver, elevation_data)
                elif layer_name in [
                    DtmConstants.BACKSCATTER,
                    DtmConstants.MIN_ACROSS_DISTANCE,
                    DtmConstants.MAX_ACROSS_DISTANCE,
                    DtmConstants.MAX_ACCROSS_ANGLE,
                ]:
                    _add_geometry_layers(driver, elevation_data)
                elif layer_name in [DtmConstants.CDI, DtmConstants.CDI_INDEX]:
                    _add_metadata_layers(driver, elevation_data)

        driver.close()
        return str(test_file)

    yield _create_dtm

    # Cleanup
    for file in created_files:
        if file.exists():
            file.unlink()


# Type alias for polygon coordinates
PolygonCoordinates = List[List[float]]


@pytest.fixture
def compute_polygon_over_dtm() -> Callable[[slice, slice, float, float, float, float], PolygonCoordinates]:
    """
    Factory fixture that returns a function to compute geographic polygons from DTM grid zones.

    This fixture is useful for:
    - Creating KML masks that correspond to specific DTM grid areas
    - Testing spatial operations on DTM zones
    - Generating bounding boxes with optional expansion

    Returns:
        Callable function that converts grid slices to geographic coordinates
    """

    def _compute(
        row_slice: slice, col_slice: slice, west: float, south: float, resolution: float, expand: float = 0.0
    ) -> PolygonCoordinates:
        """
        Converts a DTM grid zone (row/col slices) to geographic polygon coordinates.

        The function maps grid cell indices to geographic coordinates (longitude/latitude)
        using the DTM's coordinate reference system. An optional expansion parameter
        allows creating polygons slightly larger than the exact grid bounds.

        Coordinate system:
        - Rows increase from north to south (row 0 = northernmost)
        - Columns increase from west to east (col 0 = westernmost)
        - Origin point is at (west, south) - southwestern corner

        Args:
            row_slice: Slice defining the row range (north-south extent)
            col_slice: Slice defining the column range (west-east extent)
            west: Western boundary of the DTM in degrees longitude (e.g., -5.0)
            south: Southern boundary of the DTM in degrees latitude (e.g., 45.0)
            resolution: Cell size in degrees (e.g., 0.001 = ~111m at equator)
            expand: Number of cells to expand the polygon beyond slice boundaries.
                   Useful for creating buffers or ensuring complete coverage.
                   Default: 0.5 cells (half-cell expansion on each side)

        Returns:
            List of [longitude, latitude] coordinate pairs forming a closed polygon.
            Coordinates are ordered clockwise: top-left, top-right, bottom-right, bottom-left

        Example:
            >>> compute = compute_polygon_over_dtm()
            >>> polygon = compute(
            ...     row_slice=slice(5, 8),
            ...     col_slice=slice(5, 8),
            ...     west=-5.0,
            ...     south=45.0,
            ...     resolution=0.001,
            ...     expand=0.5
            ... )
            >>> # Returns polygon covering cells [5:8, 5:8] with 0.5-cell buffer
        """
        # Extract slice boundaries and apply expansion
        # Expansion creates a buffer zone around the exact grid cells
        row_start = row_slice.start - expand
        row_stop = row_slice.stop + expand
        col_start = col_slice.start - expand
        col_stop = col_slice.stop + expand

        # Convert grid indices to geographic coordinates
        # Longitude increases with column index (west to east)
        lon_min = west + col_start * resolution
        lon_max = west + col_stop * resolution

        # Latitude calculation: rows increase southward
        # row_start (smaller index) = northern edge (higher latitude)
        # row_stop (larger index) = southern edge (lower latitude)
        lat_max = south + row_start * resolution  # Northern edge
        lat_min = south + row_stop * resolution  # Southern edge

        # Create closed polygon in clockwise order
        # Format matches KML requirements: [longitude, latitude] pairs
        return [
            [lon_min, lat_max],  # Top-left (northwest corner)
            [lon_max, lat_max],  # Top-right (northeast corner)
            [lon_max, lat_min],  # Bottom-right (southeast corner)
            [lon_min, lat_min],  # Bottom-left (southwest corner)
        ]

    return _compute


@pytest.fixture
def dtm_test_file_full(dtm_file_factory):
    """
    DTM file 50x50 with all layers and no-data zones.

    Returns:
        str: Path to DTM file
        list : no-data zones
    """
    return dtm_file_factory(grid_size=50, with_nodata=True, layers="full")


def _get_default_nodata_zones(grid_size):
    """
    Generate default no-data zones scaled to grid size.

    Creates 3 zones:
    - Small zone in upper left (~6% of grid)
    - Medium zone in center-right (~10% of grid)
    - Lower zone in center bottom (~8% of grid)

    Args:
        grid_size (int): Size of the grid

    Returns:
        list: List of (row_slice, col_slice) tuples
    """
    small_size = max(2, int(grid_size * 0.06))
    medium_size = max(3, int(grid_size * 0.10))
    lower_size = max(3, int(grid_size * 0.08))

    small_pos = int(grid_size * 0.10)
    medium_row = int(grid_size * 0.40)
    medium_col = int(grid_size * 0.70)
    lower_row = int(grid_size * 0.80)
    lower_col = int(grid_size * 0.45)

    return [
        (slice(small_pos, small_pos + small_size), slice(small_pos, small_pos + small_size)),
        (slice(medium_row, medium_row + medium_size), slice(medium_col, medium_col + medium_size)),
        (slice(lower_row, lower_row + lower_size), slice(lower_col, lower_col + lower_size)),
    ]


def _generate_elevation_pattern(size, vertical_offset):
    """
    Generate realistic bathymetry pattern with depth gradient.

    Creates a pattern simulating:
    - Shallow coastal waters (0 to -20m)
    - Continental shelf slope (-20 to -150m)
    - Some topographic features (seamounts, ridges)

    Args:
        size: Grid dimension (size x size)
        vertical_offset : offset to apply to elevation

    Returns:
        np.ndarray: Elevation values in meters (negative for bathymetry)
    """
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)

    base_depth = -20 - 130 * X
    seamount = 50 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
    ridge = 20 * np.exp(-((X - Y) ** 2) / 0.01)

    np.random.seed(42)
    roughness = np.random.normal(0, 2, (size, size))

    elevation = base_depth + seamount + ridge + roughness + vertical_offset
    return elevation.astype(np.float32)


def _add_nodata_zones(elevation_data, nodata_zones):
    """Add no-data (NaN) zones to the elevation data."""
    data = elevation_data.copy()
    for row_slice, col_slice in nodata_zones:
        data[row_slice, col_slice] = np.nan
    return data


def _add_elevation_statistics_layers(driver, elevation_data):
    """Add elevation min, max, and standard deviation layers."""
    elev_min = elevation_data - np.abs(np.random.normal(0.5, 0.2, elevation_data.shape))
    elev_min = np.where(np.isnan(elevation_data), np.nan, elev_min)
    driver.add_layer(DtmConstants.ELEVATION_MIN, data=elev_min.astype(np.float32))

    elev_max = elevation_data + np.abs(np.random.normal(0.5, 0.2, elevation_data.shape))
    elev_max = np.where(np.isnan(elevation_data), np.nan, elev_max)
    driver.add_layer(DtmConstants.ELEVATION_MAX, data=elev_max.astype(np.float32))

    depth_magnitude = np.abs(elevation_data)
    stdev = 0.1 + 0.01 * depth_magnitude
    stdev = np.where(np.isnan(elevation_data), np.nan, stdev)
    driver.add_layer(DtmConstants.STDEV, data=stdev.astype(np.float32))


def _add_count_layers(driver, elevation_data):
    """Add value_count and filtered_count layers."""

    depth_magnitude = np.abs(elevation_data)
    base_count = np.clip(50 - depth_magnitude / 3, 5, 50)

    valid_mask = ~np.isnan(base_count)
    count_noise = np.zeros_like(base_count, dtype=np.int32)
    count_noise[valid_mask] = np.random.poisson(base_count[valid_mask]).astype(np.int32)

    value_count = np.where(np.isnan(elevation_data), get_missing_value(DtmConstants.VALUE_COUNT), count_noise)
    driver.add_layer(DtmConstants.VALUE_COUNT, data=value_count.astype(np.int32))

    filtered_count = np.where(
        np.isnan(elevation_data),
        get_missing_value(DtmConstants.FILTERED_COUNT),
        np.random.poisson(2, elevation_data.shape),
    )
    driver.add_layer(DtmConstants.FILTERED_COUNT, data=filtered_count.astype(np.int32))


def _add_interpolated_layers(driver, elevation_data):
    """Add interpolation flag layer."""
    interp_flag = np.where(np.isnan(elevation_data), get_missing_value(DtmConstants.INTERPOLATION_FLAG), 0).astype(
        np.int8
    )
    driver.add_layer(DtmConstants.INTERPOLATION_FLAG, data=interp_flag)


def _add_geometry_layers(driver, elevation_data):
    """Add backscatter and beam geometry layers."""
    size = elevation_data.shape

    backscatter = np.random.uniform(-35, -5, size)
    backscatter = np.where(np.isnan(elevation_data), np.nan, backscatter)
    driver.add_layer(DtmConstants.BACKSCATTER, data=backscatter.astype(np.float32))

    min_across = np.random.uniform(0, 100, size)
    min_across = np.where(np.isnan(elevation_data), np.nan, min_across)
    driver.add_layer(DtmConstants.MIN_ACROSS_DISTANCE, data=min_across.astype(np.float32))

    max_across = np.random.uniform(500, 2000, size)
    max_across = np.where(np.isnan(elevation_data), np.nan, max_across)
    driver.add_layer(DtmConstants.MAX_ACROSS_DISTANCE, data=max_across.astype(np.float32))

    max_angle = np.random.uniform(0, 65, size)
    max_angle = np.where(np.isnan(elevation_data), np.nan, max_angle)
    driver.add_layer(DtmConstants.MAX_ACCROSS_ANGLE, data=max_angle.astype(np.float32))


def _add_metadata_layers(driver, elevation_data):
    """Add CDI metadata reference layers."""
    sample_cdis = [
        "1234-TEST-CDI-001",
        "1234-TEST-CDI-002",
        "1234-TEST-CDI-003",
        "1234-TEST-CDI-004",
    ]
    driver.create_cdi_reference_variable(sample_cdis)

    cdi_index = np.full_like(elevation_data, fill_value=-1, dtype=np.int32)
    size = elevation_data.shape[0]
    half = size // 2

    cdi_index[:half, :half] = 0
    cdi_index[:half, half:] = 1
    cdi_index[half:, half:] = 2
    cdi_index[half:, :half] = 3
    cdi_index = np.where(np.isnan(elevation_data), -1, cdi_index)

    driver.add_layer(DtmConstants.CDI_INDEX, data=cdi_index.astype(np.int32))
