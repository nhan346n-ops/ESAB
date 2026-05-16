#! /usr/bin/env python3
# coding: utf-8

"""
Integration tests for DTM (Digital Terrain Model) merge fill process.
Tests the merging of multiple NetCDF files with no-data zones.
"""

import logging
from typing import List, Tuple

import numpy as np
import numpy.testing as npt
import pytest

import pyat.dtm.dtm_driver as dtm_driver
from pyat.dtm.dtm_driver import get_missing_value
from pyat.dtm.merge.merge_fill import MergeFillProcess
from tests.generator import kml_generator

logger = logging.getLogger()

# DTM coordinate system parameters
WEST = -5.0
SOUTH = 45.0
GRID_SIZE = 50
RESOLUTION = 0.001  # degrees per cell

# Type aliases for better readability
ZoneSlice = Tuple[slice, slice]
NoDataZones = List[ZoneSlice]
Coordinates = List[List[float]]


# ============================================================================
# TEST CONFIGURATION FIXTURES
# ============================================================================


@pytest.fixture
def all_nodata_zones() -> NoDataZones:
    """
    Provides all no-data zones as a list of slice tuples.

    Returns:
        List of (row_slice, col_slice) tuples defining rectangular no-data regions
    """
    return [
        (slice(5, 8), slice(5, 8)),  # Small zone at top-left
        (slice(40, 44), slice(22, 26)),  # Zone at bottom-center
        (slice(20, 25), slice(35, 40)),  # Zone at middle-right
    ]


@pytest.fixture
def generates_input_dtms(dtm_file_factory, all_nodata_zones: NoDataZones) -> Tuple[str, List[str]]:
    """
    Generates a reference DTM and multiple test DTMs with complementary no-data zones.

    Strategy: Each test DTM has all zones except one, so together they provide
    complete coverage when merged.

    Returns:
        Tuple of (reference_dtm_path, list_of_test_dtm_paths)
    """
    # Reference DTM with all no-data zones
    dtm_ref_file = dtm_file_factory(grid_size=GRID_SIZE, with_nodata=True, nodata_zones=all_nodata_zones, layers="full")

    # Generate one DTM per zone, each missing that specific zone
    dtm_files: List[str] = []
    for i in range(len(all_nodata_zones)):
        partial_nodata_zone = all_nodata_zones.copy()
        partial_nodata_zone.pop(i)  # Remove zone i to create data there

        dtm_file = dtm_file_factory(
            grid_size=GRID_SIZE,
            with_nodata=True,
            nodata_zones=partial_nodata_zone,
            layers="full",
            vertical_offset=-5.0,
            cell_size=RESOLUTION,
            origin_lon=WEST,
            origin_lat=SOUTH,
        )
        dtm_files.append(dtm_file)

    return dtm_ref_file, dtm_files


@pytest.fixture
def kml_areas(
    request, tmp_path, compute_polygon_over_dtm, all_nodata_zones: NoDataZones
) -> Tuple[str | None, NoDataZones]:
    """
    Generates KML mask files with varying numbers of defined areas.

    Args:
        request.param: One of "kml_0", "kml_1", "kml_2", or "kml_3"

    Returns:
        Tuple of (kml_file_path, corresponding_nodata_zones)
    """
    if request.param == "kml_0":
        return None, all_nodata_zones

    areas = {}
    nodata_zones = []

    # Area 1: Zone (5:8, 5:8) - Top-left zone
    zone_1 = all_nodata_zones[0]
    areas["area1"] = compute_polygon_over_dtm(zone_1[0], zone_1[1], WEST, SOUTH, RESOLUTION, 0.5)
    nodata_zones.append(zone_1)
    if request.param == "kml_1":
        return kml_generator.create_kml(tmp_path, areas), nodata_zones

    # Area 2: Zone (40:44, 22:26) - Bottom-center zone
    zone_2 = all_nodata_zones[1]
    areas["area2"] = compute_polygon_over_dtm(zone_2[0], zone_2[1], WEST, SOUTH, RESOLUTION, 0.5)
    nodata_zones.append(zone_2)
    if request.param == "kml_2":
        return kml_generator.create_kml(tmp_path, areas), nodata_zones

    # Area 3: Zone (20:25, 35:40) - Middle-right zone
    zone_3 = all_nodata_zones[2]
    areas["area3"] = compute_polygon_over_dtm(zone_3[0], zone_3[1], WEST, SOUTH, RESOLUTION, 0.5)
    nodata_zones.append(zone_3)
    return kml_generator.create_kml(tmp_path, areas), nodata_zones


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _verify_merged_zone(
    ref_dtm_path: str, dtm_files: List[str], result_dtm_path: str, nodata_zones: NoDataZones
) -> None:
    """
    Verifies that specified zones have been correctly merged from source DTMs.

    Args:
        ref_dtm_path: Path to reference DTM file (with all no-data zones)
        dtm_files: List of source DTM file paths used for merging
        result_dtm_path: Path to the merged result DTM file
        nodata_zones: List of zones that should have been filled
    """
    with dtm_driver.open_dtm(ref_dtm_path) as i_ref, dtm_driver.open_dtm(result_dtm_path) as result:
        # Check each source DTM contributed its data zone correctly
        for dtm_file_path, data_zone in zip(dtm_files, nodata_zones):
            with dtm_driver.open_dtm(dtm_file_path) as i_dtm:
                logger.info(f"Verifying {dtm_file_path} for zone {data_zone}")

                _verify_layers_match(i_dtm, result, data_zone)
                _verify_reference_unchanged(i_ref, result)


def _verify_layers_match(source_dtm, result_dtm, data_zone: ZoneSlice) -> None:
    """
    Verifies that all layers from source DTM are present and correct in result.

    Args:
        source_dtm: Source DTM dataset
        result_dtm: Result DTM dataset
        data_zone: Zone to verify (row_slice, col_slice)
    """
    for layer_name in dtm_driver.LAYER_NAMES:
        if layer_name not in source_dtm:
            continue

        assert layer_name in result_dtm, f"Layer {layer_name} missing in merged DTM"

        result_data = result_dtm[layer_name][:].data

        # Log NaN count for elevation layer
        if layer_name == dtm_driver.DtmConstants.ELEVATION_NAME:
            nan_count = np.isnan(result_data).sum()
            logger.info(f"NaN count in elevation layer: {nan_count}")

        # Verify only spatial data layers (50x50 grid)
        if result_data.shape == (GRID_SIZE, GRID_SIZE):
            logger.info(f"Verifying layer '{layer_name}' for zone {data_zone}")
            row_slice, col_slice = data_zone

            npt.assert_array_equal(
                actual=source_dtm[layer_name][row_slice, col_slice].data,
                desired=result_data[row_slice, col_slice],
                err_msg=f"Zone {data_zone} in layer '{layer_name}' not correctly merged",
                verbose=True,
            )


def _verify_reference_unchanged(ref_dtm, result_dtm) -> None:
    """
    Verifies that data outside merged zones remains unchanged from reference.

    Args:
        ref_dtm: Reference DTM dataset
        result_dtm: Result DTM dataset
        excluded_zone: Zone to exclude from verification
    """
    for layer_name in dtm_driver.LAYER_NAMES:
        if layer_name not in ref_dtm:
            continue

        ref_data = ref_dtm[layer_name][:].data
        result_data = result_dtm[layer_name][:].data

        if ref_data.shape == (GRID_SIZE, GRID_SIZE):
            _verify_no_change_outside_zone(layer_name, ref_data, result_data)


def _verify_no_change_outside_zone(layer_name: str, in_data: np.ndarray, result_data: np.ndarray) -> None:
    """
    Verifies that valid data (non-missing) remains unchanged after merge.

    Args:
        layer_name: Name of the layer being verified
        in_data: Original data array
        result_data: Result data array after merge
    """
    logger.info(f"Verifying layer '{layer_name}' outside merged zones unchanged")

    no_data_value = get_missing_value(layer_name)

    # Create mask for valid data
    if no_data_value is np.nan:
        mask = ~np.isnan(in_data)
    else:
        mask = in_data != no_data_value

    assert np.all(
        in_data[mask] == result_data[mask]
    ), f"Data outside interpolated zones changed in layer '{layer_name}'"


# ============================================================================
# TESTS
# ============================================================================


@pytest.mark.parametrize("kml_areas", ["kml_0", "kml_1", "kml_2", "kml_3"], indirect=True)
def test_merge_dtms_with_mask(
    tmp_path, generates_input_dtms: Tuple[str, List[str]], kml_areas: Tuple[str | None, NoDataZones]
) -> None:
    """
    Integration test for merging multiple DTM files with KML mask.

    Tests that:
    1. Multiple DTM files can be merged into a single output
    2. Only zones defined in KML mask are merged
    3. Data outside merged zones remains unchanged
    4. All layers are correctly processed
    """
    dtm_ref_file, dtm_files = generates_input_dtms
    mask, nodata_zones = kml_areas

    # Prepare input: reference DTM + complementary DTMs
    i_paths = [dtm_ref_file]
    i_paths.extend(dtm_files)

    # Run merge process
    o_path = tmp_path / "merged.dtm.nc"
    process = MergeFillProcess(
        i_paths=i_paths,
        o_path=o_path,
        coord={
            "north": SOUTH + GRID_SIZE * RESOLUTION,
            "south": SOUTH,
            "west": WEST,
            "east": WEST + GRID_SIZE * RESOLUTION,
        },
        mask=mask,
    )
    process()

    # Verify merge results
    _verify_merged_zone(dtm_ref_file, dtm_files, o_path, nodata_zones)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
