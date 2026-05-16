#! /usr/bin/env python3
# coding: utf-8

"""
Integration tests for DTM (Digital Terrain Model) merge simple process.

This module tests the merging of multiple NetCDF files containing DTM data
with complementary no-data zones. The test strategy generates multiple DTM
files, each missing different zones, to verify that the merge process
correctly combines them into a complete result.

Key concepts:
- No-data zones: Rectangular regions in DTM files with missing data
- Complementary coverage: Each input DTM has data where others don't
- KML masking: Optional spatial filtering using KML polygons
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import numpy.testing as npt
import pytest

from pyat.dtm.dtm_driver import (
    LAYER_NAMES,
    DtmConstants,
    get_missing_value,
    open_dtm,
)
from pyat.dtm.merge.merge_simple import MergeSimpleProcess
from tests.generator import kml_generator

logger = logging.getLogger()

# ============================================================================
# CONSTANTS
# ============================================================================

# DTM coordinate system parameters
WEST = -5.0  # Western boundary in degrees
SOUTH = 45.0  # Southern boundary in degrees
GRID_SIZE = 50  # Grid dimensions (50x50 cells)
RESOLUTION = 0.001  # Cell resolution in degrees

# Type aliases for better readability
ZoneSlice = Tuple[slice, slice]  # (row_slice, col_slice) for a rectangular zone
ZoneSlices = List[ZoneSlice]  # List of no-data zones
Coordinates = List[List[float]]  # Polygon coordinates


# ============================================================================
# TEST CONFIGURATION FIXTURES
# ============================================================================


@pytest.fixture
def all_nodata_zones() -> ZoneSlices:
    """
    Define all no-data zones for testing.

    These zones represent rectangular regions where data will be missing
    in individual DTM files. The merge process should correctly fill
    these zones from complementary files.

    Returns:
        List of (row_slice, col_slice) tuples defining rectangular regions
    """
    return [
        (slice(5, 8), slice(5, 8)),  # Small zone at top-left (3x3 cells)
        (slice(40, 44), slice(22, 26)),  # Zone at bottom-center (4x4 cells)
        (slice(20, 25), slice(35, 40)),  # Zone at middle-right (5x5 cells)
    ]


@pytest.fixture
def generates_input_dtms(dtm_file_factory: Callable, all_nodata_zones: ZoneSlices) -> List[str]:
    """
    Generate multiple test DTMs with complementary no-data coverage.

    Strategy: Create N DTM files (where N = number of zones), with each
    file containing data everywhere EXCEPT in zone i. When merged together,
    all zones will have complete coverage from at least one source file.

    Args:
        dtm_file_factory: Factory fixture for creating DTM files
        all_nodata_zones: List of all no-data zones to test

    Returns:
        List of paths to generated DTM files
    """
    dtm_files: List[str] = []

    # Generate one DTM per zone, each missing that specific zone
    for i in range(len(all_nodata_zones)):
        # Copy zones list and remove zone i to create data there
        partial_nodata_zones = all_nodata_zones.copy()
        partial_nodata_zones.pop(i)

        # Create DTM with vertical offset to distinguish source files
        dtm_file = dtm_file_factory(
            grid_size=GRID_SIZE,
            with_nodata=True,
            nodata_zones=partial_nodata_zones,
            layers="full",
            vertical_offset=-5.0 * i,  # Offset helps identify source in merge
            cell_size=RESOLUTION,
            origin_lon=WEST,
            origin_lat=SOUTH,
        )
        dtm_files.append(dtm_file)

    return dtm_files


@pytest.fixture
def kml_areas(
    request: pytest.FixtureRequest, tmp_path: Path, compute_polygon_over_dtm: Callable, all_nodata_zones: ZoneSlices
) -> Tuple[str | None, ZoneSlices | None]:
    """
    Generate KML mask files for spatial filtering during merge.

    The fixture is parameterized to generate different KML configurations:
    - "kml_0": No mask (returns None)
    - "kml_1": Mask covering zone 1 only
    - "kml_2": Mask covering zones 1 and 2
    - "kml_3": Mask covering all three zones

    Args:
        request: Pytest fixture request with parameter
        tmp_path: Temporary directory for KML file
        compute_polygon_over_dtm: Helper to convert grid zones to polygons
        all_nodata_zones: List of all no-data zones

    Returns:
        Tuple of (kml_file_path, list_of_masked_zones) or (None, None)
    """
    if request.param == "kml_0":
        return None, None  # No mask applied

    areas: Dict[str, Coordinates] = {}
    nodata_zones: ZoneSlices = []

    # Area 1: Zone (5:8, 5:8) - Top-left zone
    zone_1 = all_nodata_zones[0]
    areas["area1"] = compute_polygon_over_dtm(zone_1[0], zone_1[1], WEST, SOUTH, RESOLUTION, 0.0)
    nodata_zones.append(zone_1)
    if request.param == "kml_1":
        return kml_generator.create_kml(tmp_path, areas), nodata_zones

    # Area 2: Zone (40:44, 22:26) - Bottom-center zone
    zone_2 = all_nodata_zones[1]
    areas["area2"] = compute_polygon_over_dtm(zone_2[0], zone_2[1], WEST, SOUTH, RESOLUTION, 0.0)
    nodata_zones.append(zone_2)
    if request.param == "kml_2":
        return kml_generator.create_kml(tmp_path, areas), nodata_zones

    # Area 3: Zone (20:25, 35:40) - Middle-right zone
    zone_3 = all_nodata_zones[2]
    areas["area3"] = compute_polygon_over_dtm(zone_3[0], zone_3[1], WEST, SOUTH, RESOLUTION, 0.0)
    nodata_zones.append(zone_3)
    return kml_generator.create_kml(tmp_path, areas), nodata_zones


# ============================================================================
# LAYER COMPUTATION ALGORITHMS
# ============================================================================


def _sum_with_missing_value(layers_data: List[np.ndarray], missing_value: float) -> np.ndarray:
    """
    Sum layers while treating missing values as zero.

    Args:
        layers_data: List of 2D arrays to sum
        missing_value: Value representing missing data

    Returns:
        Summed array with missing values excluded
    """
    stack = np.stack(layers_data)
    masked = np.ma.masked_equal(stack, missing_value)
    return masked.sum(axis=0)


def _compute_cdi(merged_zones: ZoneSlices, all_nodata_zones: ZoneSlices) -> np.ndarray:
    """
    Compute CDI (Composite Data Index) layer.

    Creates a quadrant-based index pattern for the grid.

    Returns:
        2D array with quadrant indices (0-3)
    """
    logger.info(f"_compute_cdi on {merged_zones}")
    # Specific case with kml_2 test. Only 3 CDI remain in the merged file :
    # - 1234-TEST-CDI-001, index 0 (unchanged from the input DTM)
    # - 1234-TEST-CDI-002, dismissed, no cell remaining with this CDI
    # - 1234-TEST-CDI-003, index becomes 1 (2 in the the input DTM)
    # - 1234-TEST-CDI-004, index becomes 2 (3 in the the input DTM)
    is_kml_2 = merged_zones == [all_nodata_zones[0], all_nodata_zones[1]]

    missing_value = get_missing_value(DtmConstants.CDI_INDEX)
    result = np.full(shape=(GRID_SIZE, GRID_SIZE), fill_value=missing_value, dtype=np.int32)
    half = GRID_SIZE // 2
    # Assign quadrant values: top-left=0, top-right=1, bottom-right=2, bottom-left=3
    result[:half, :half] = 0
    result[:half, half:] = missing_value if is_kml_2 else 1  # No CDI expected here if case of kml_2
    result[half:, half:] = 1 if is_kml_2 else 2
    result[half:, :half] = 2 if is_kml_2 else 3
    return result


def _stddev(layers_data: List[np.ndarray], all_nodata_zones: ZoneSlices) -> np.ndarray:
    """
    Calculate standard deviation accounting for no-data zones.

    In no-data zones, only one DTM file provides values, so the standard
    deviation is calculated differently (division by 1 instead of N).

    Args:
        layers_data: List of 2D arrays from different DTM files
        all_nodata_zones: Zones where only one file has data

    Returns:
        2D array of standard deviation values
    """
    logger.info(f"Calculating stddev with nodata zones: {all_nodata_zones} " f"on {len(layers_data)} layers")

    # Sum of squared values across all layers
    square_values = np.nansum(np.square(layers_data), axis=0)

    # Default factor: divide by number of layers
    factor = np.full_like(square_values, fill_value=len(layers_data))

    # Adjust factor for no-data zones (only 1 DTM has data there)
    for row_slice, col_slice in all_nodata_zones:
        factor[row_slice, col_slice] = 1

    return np.sqrt(square_values / factor)


# Layer-specific computation algorithms
NP_ALGO: Dict[str, Callable] = {
    DtmConstants.ELEVATION_NAME: lambda layers_data: np.nanmean(layers_data, axis=0),
    DtmConstants.ELEVATION_MIN: lambda layers_data: np.nanmin(layers_data, axis=0),
    DtmConstants.ELEVATION_MAX: lambda layers_data: np.nanmax(layers_data, axis=0),
    DtmConstants.VALUE_COUNT: lambda layers_data: _sum_with_missing_value(
        layers_data, get_missing_value(DtmConstants.VALUE_COUNT)
    ),
    DtmConstants.STDEV: _stddev,
    DtmConstants.CDI_INDEX: _compute_cdi,
    DtmConstants.INTERPOLATION_FLAG: lambda layers_data: np.zeros_like(layers_data[0]),
    DtmConstants.BACKSCATTER: lambda layers_data: np.nanmean(layers_data, axis=0),
    DtmConstants.MIN_ACROSS_DISTANCE: lambda layers_data: np.nanmin(layers_data, axis=0),
    DtmConstants.MAX_ACROSS_DISTANCE: lambda layers_data: np.nanmax(layers_data, axis=0),
    DtmConstants.MAX_ACCROSS_ANGLE: lambda layers_data: np.nanmax(layers_data, axis=0),
    DtmConstants.FILTERED_COUNT: lambda layers_data: _sum_with_missing_value(
        layers_data, get_missing_value(DtmConstants.FILTERED_COUNT)
    ),
}


# ============================================================================
# VERIFICATION HELPER FUNCTIONS
# ============================================================================


def _mask_merged_zones(layer_name: str, values: np.ndarray, merged_zones: ZoneSlices) -> None:
    """
    Apply no-data mask to zones that should be excluded by KML filtering.

    Args:
        layer_name: Name of the DTM layer
        values: 2D array to mask
        merged_zones: Zones to set as no-data
    """
    logger.info(f"Masking merged zones for layer '{layer_name}': {merged_zones}")
    no_data_value = get_missing_value(layer_name)

    # Create mask: True where data should be removed
    mask = np.ones_like(values, dtype=bool)
    for row_slice, col_slice in merged_zones:
        mask[row_slice, col_slice] = False

    values[mask] = no_data_value


def _verify_layers_match(
    i_dtms: List, result_dtm, merged_zones: ZoneSlices | None, all_nodata_zones: ZoneSlices
) -> None:
    """
    Verify that merged DTM layers match expected computed values.

    Compares each layer in the result DTM against the expected values
    computed from the input DTMs using the appropriate algorithm.

    Args:
        i_dtms: List of input DTM file handles
        result_dtm: Merged result DTM file handle
        merged_zones: Zones masked by KML filtering (if any)
        nodata_zones: Original no-data zones from input files
    """
    # Update stddev algorithm with current nodata_zones
    logger.info(f"nodata_zones {all_nodata_zones}")
    NP_ALGO[DtmConstants.STDEV] = lambda layers_data: _stddev(layers_data, all_nodata_zones)
    NP_ALGO[DtmConstants.CDI_INDEX] = lambda _: _compute_cdi(merged_zones, all_nodata_zones)

    for layer_name in LAYER_NAMES:
        # Skip layers not present in input DTMs
        if layer_name not in i_dtms[0]:
            continue

        assert layer_name in result_dtm, f"Layer {layer_name} missing in merged DTM"

        result_data = result_dtm[layer_name][:].data

        # Verify only spatial data layers (50x50 grid)
        if result_data.shape == (GRID_SIZE, GRID_SIZE):
            logger.info(f"Verifying layer '{layer_name}'")

            # Collect data from all input DTMs for this layer
            dtm_data = [i_dtm[layer_name][:].data for i_dtm in i_dtms]

            # Compute expected values using appropriate algorithm
            expected_values = NP_ALGO[layer_name](dtm_data)

            # Apply KML mask if present
            if merged_zones:
                _mask_merged_zones(layer_name, expected_values, merged_zones)

            # Log values in no-data zones for debugging
            for row_slice, col_slice in all_nodata_zones:
                logger.debug(
                    f"Zone {row_slice}, {col_slice}: "
                    f"expected={expected_values[row_slice, col_slice]}, "
                    f"result={result_data[row_slice, col_slice]}"
                )

            # Compare actual vs expected with tolerance
            npt.assert_allclose(
                actual=result_data,
                desired=expected_values,
                atol=1e-6,
                err_msg=f"Values in layer '{layer_name}' not correctly merged",
                verbose=True,
                equal_nan=True,
            )


def _verify_merged_zone(
    dtm_file_1: str,
    dtm_file_2: str,
    dtm_file_3: str,
    result_dtm_path: Path,
    merged_zones: ZoneSlices | None,
    all_nodata_zones: ZoneSlices,
) -> None:
    """
    Open all DTM files and verify the merge result.

    Args:
        dtm_file_1: Path to first input DTM
        dtm_file_2: Path to second input DTM
        dtm_file_3: Path to third input DTM
        result_dtm_path: Path to merged result DTM
        merged_zones: Zones masked by KML (if any)
        nodata_zones: Original no-data zones from inputs
    """
    with (
        open_dtm(dtm_file_1) as i_dtm_1,
        open_dtm(dtm_file_2) as i_dtm_2,
        open_dtm(dtm_file_3) as i_dtm_3,
        open_dtm(result_dtm_path) as result,
    ):
        _verify_layers_match([i_dtm_1, i_dtm_2, i_dtm_3], result, merged_zones, all_nodata_zones)


# ============================================================================
# TESTS
# ============================================================================


@pytest.mark.parametrize("kml_areas", ["kml_0", "kml_1", "kml_2", "kml_3"], indirect=True)
def test_merge_dtms_with_mask(
    tmp_path: Path,
    generates_input_dtms: List[str],
    all_nodata_zones: ZoneSlices,
    kml_areas: Tuple[str | None, ZoneSlices | None],
) -> None:
    """
    Test DTM merge process with KML spatial masking.

    This test verifies that:
    1. Multiple DTM files with complementary no-data zones merge correctly
    2. KML masks properly filter the merged result
    3. All layers compute expected values using appropriate algorithms
    4. No-data zones are handled correctly in statistical computations

    Args:
        tmp_path: Pytest temporary directory
        generates_input_dtms: Fixture providing input DTM file paths
        all_nodata_zones: Fixture providing no-data zone definitions
        kml_areas: Fixture providing KML mask file and masked zones
    """
    # Unpack test inputs
    dtm_file_1, dtm_file_2, dtm_file_3 = generates_input_dtms
    kml_file, merged_zones = kml_areas

    # Configure merge process
    o_path = tmp_path / "merged.dtm.nc"
    process = MergeSimpleProcess(
        i_paths=[dtm_file_1, dtm_file_2, dtm_file_3],
        o_path=o_path,
        coord={
            "north": SOUTH + GRID_SIZE * RESOLUTION,
            "south": SOUTH,
            "west": WEST,
            "east": WEST + GRID_SIZE * RESOLUTION,
        },
        mask=kml_file,
    )

    # Execute merge
    process()

    # Verify merge results
    _verify_merged_zone(dtm_file_1, dtm_file_2, dtm_file_3, o_path, merged_zones, all_nodata_zones)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
