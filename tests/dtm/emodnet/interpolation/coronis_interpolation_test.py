#! /usr/bin/env python3
# coding: utf-8

import logging

import numpy as np
import pytest

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.transform.interpolation.coronis.amle_interpolation as amle
import pyat.dtm.transform.interpolation.coronis.ccst_interpolation as ccst
import pyat.dtm.transform.interpolation.coronis.cubic_interpolation as cubic
import pyat.dtm.transform.interpolation.coronis.harmonic_interpolation as harmonic
import pyat.dtm.transform.interpolation.coronis.linear_interpolation as linear
import pyat.dtm.transform.interpolation.coronis.navier_stokes_interpolation as navier_stokes
import pyat.dtm.transform.interpolation.coronis.nearest_interpolation as nearest
import pyat.dtm.transform.interpolation.coronis.purbf_interpolation as purbf
import pyat.dtm.transform.interpolation.coronis.rbf_interpolation as rbf
import pyat.dtm.transform.interpolation.coronis.telea_interpolation as telea
import pyat.dtm.transform.interpolation.coronis.tv_interpolation as tv
from pyat.dtm.dtm_driver import get_missing_value
from pyat.dtm.transform.interpolation.coronis.interpolation import interpolate_dtms
from tests.generator import kml_generator

logger = logging.getLogger()

# DTM coordinate system parameters
WEST = -5.0
SOUTH = 45.0
RESOLUTION = 0.001  # degrees per cell

# No-data areas
NODATA_ZONE_1 = (slice(5, 8), slice(5, 8))
NODATA_ZONE_2 = (slice(20, 25), slice(35, 40))
NODATA_ZONE_3 = (slice(40, 44), slice(22, 26))
# ============================================================================
# TEST CONFIGURATION FIXTURES
# ============================================================================


@pytest.fixture
def all_nodata_zones():
    """
    Fixture providing all no-data zones as a list.

    Returns:
        list: List of (row_slice, col_slice) tuples for all no-data zones
    """
    return [NODATA_ZONE_1, NODATA_ZONE_2, NODATA_ZONE_3]


@pytest.fixture
def dtm_test_file(request, dtm_file_factory, all_nodata_zones):
    """
    Fixture providing the DTM test file based on the requested type.
    """
    if request.param == "full":
        return dtm_file_factory(grid_size=50, with_nodata=True, nodata_zones=all_nodata_zones, layers="full")
    elif request.param == "simple":
        return dtm_file_factory(grid_size=50, with_nodata=True, nodata_zones=all_nodata_zones, layers="simple")
    else:
        raise ValueError("Unknown dataset type")


@pytest.fixture
def interpolated_layers():
    """
    Fixture providing the list of layers that should be interpolated.

    Returns:
        list: Layer names that interpolation processes should handle
    """
    return [
        dtm_driver.DtmConstants.ELEVATION_NAME,
        dtm_driver.DtmConstants.ELEVATION_MIN,
        dtm_driver.DtmConstants.ELEVATION_MAX,
    ]


@pytest.fixture
def total_nodata_cells():
    """
    Fixture providing the total number of no-data cells in the test DTM.

    This corresponds to the sum of all no-data zones:
    - Small zone (3x3): 9 cells
    - Medium zone (5x5): 25 cells
    - Lower zone (4x4): 16 cells
    Total: 50 cells

    Returns:
        int: Total number of no-data cells
    """
    return 50


@pytest.fixture
def kml_nodata_zone_2(tmp_path, compute_polygon_over_dtm):
    """
    Fixture generating a KML with a single area covering the medium no-data zone.

    This area covers approximately the center-right region [20:25, 35:40].

    Returns:
        Path: Path to the generated KML file
    """
    return kml_generator.create_kml(
        tmp_path, {"area2": compute_polygon_over_dtm(NODATA_ZONE_2[0], NODATA_ZONE_2[1], WEST, SOUTH, RESOLUTION, 0.6)}
    )


@pytest.fixture
def kml_nodata_zone_2_and_3(tmp_path, compute_polygon_over_dtm):
    """
    Fixture generating a KML with two areas covering two no-data zones.

    - area1: covers the medium zone [20:25, 35:40]
    - area2: covers the lower zone [40:44, 22:26]

    Returns:
        Path: Path to the generated KML file
    """
    return kml_generator.create_kml(
        tmp_path,
        {
            "area2": compute_polygon_over_dtm(NODATA_ZONE_2[0], NODATA_ZONE_2[1], WEST, SOUTH, RESOLUTION, 0.6),
            "area3": compute_polygon_over_dtm(NODATA_ZONE_3[0], NODATA_ZONE_3[1], WEST, SOUTH, RESOLUTION, 0.6),
        },
    )


@pytest.fixture
def kml_all_areas(tmp_path, compute_polygon_over_dtm):
    """
    Fixture generating a KML with all areas covering all no-data zones.

    Returns:
        Path: Path to the generated KML file
    """
    return kml_generator.create_kml(
        tmp_path,
        {
            "area1": compute_polygon_over_dtm(NODATA_ZONE_1[0], NODATA_ZONE_1[1], WEST, SOUTH, RESOLUTION, 0.6),
            "area2": compute_polygon_over_dtm(NODATA_ZONE_2[0], NODATA_ZONE_2[1], WEST, SOUTH, RESOLUTION, 0.6),
            "area3": compute_polygon_over_dtm(NODATA_ZONE_3[0], NODATA_ZONE_3[1], WEST, SOUTH, RESOLUTION, 0.6),
        },
    )


# Interpolation algorithms for all tests.
INTERPOLATION_ALGORITHMS = [
    (nearest.NearestInterpolationProcess, {}),
    (linear.LinearInterpolationProcess, {}),
    (cubic.CubicInterpolationProcess, {}),
    (harmonic.HarmonicInterpolationProcess, {}),
    (ccst.CcstInterpolationProcess, {}),
    (tv.TVInterpolationProcess, {}),
    (amle.AMLEInterpolationProcess, {}),
    (rbf.RbfInterpolationProcess, {"rbf_parameters": rbf.RbfParameters(rbf_type="cubic")}),
    (navier_stokes.NavierStokesInterpolationProcess, {"radius": 10}),
    (telea.TeleaInterpolationProcess, {"radius": 10}),
    (purbf.PurbfInterpolationProcess, {"pu_min_point_in_cell": 10}),
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _verify_nodata_before_interpolation(dtm_path, all_nodata_zones, total_nodata_cells, interpolated_layers):
    """
    Verify that the input DTM has the expected no-data cells before interpolation.

    Args:
        dtm_path: Path to the DTM file
        all_nodata_zones: List of (row_slice, col_slice) tuples for no-data zones
        total_nodata_cells: Expected total number of no-data cells
        interpolated_layers: List of layer names to check
    """
    with dtm_driver.open_dtm(dtm_path) as i_dtm:
        # Check elevation layers
        for layer_name in interpolated_layers:
            if layer_name in i_dtm:
                layer_data = i_dtm[layer_name][:].data
                nan_count = np.isnan(layer_data).sum()
                assert (
                    nan_count == total_nodata_cells
                ), f"Expected {total_nodata_cells} empty cells in {layer_name}, found {nan_count}"

        # Check interpolation flag
        layer_name = dtm_driver.DtmConstants.INTERPOLATION_FLAG
        if layer_name in i_dtm:
            interp_flag = i_dtm[layer_name][:].data
            missing_flag_count = np.sum(interp_flag == 127)
            assert (
                missing_flag_count == total_nodata_cells
            ), f"Expected {total_nodata_cells} empty cells in {layer_name}, found {missing_flag_count}"

        # Check value count
        layer_name = dtm_driver.DtmConstants.VALUE_COUNT
        if layer_name in i_dtm:
            value_count = i_dtm[layer_name][:].data
            missing_count = np.sum(value_count == -1)
            assert (
                missing_count == total_nodata_cells
            ), f"Expected {total_nodata_cells} empty cells in {layer_name}, found {missing_count}"

        # Check CDI index
        layer_name = dtm_driver.DtmConstants.CDI_INDEX
        if layer_name in i_dtm:
            cdi_index = i_dtm[layer_name][:].data
            missing_cdi = np.sum(cdi_index == -1)
            assert (
                missing_cdi == total_nodata_cells
            ), f"Expected {total_nodata_cells} empty cells in {layer_name}, found {missing_cdi}"


def _verify_full_interpolation(in_dtm_path, result_dtm_path, all_nodata_zones, total_nodata_cells, interpolated_layers):
    """
    Verify that all no-data zones have been interpolated.

    Args:
        result_dtm_path: Path to the interpolated DTM file
        all_nodata_zones: List of (row_slice, col_slice) tuples for no-data zones
        total_nodata_cells: Expected number of cells that were interpolated
        interpolated_layers: List of layer names to check
    """
    with dtm_driver.open_dtm(in_dtm_path) as i_dtm, dtm_driver.open_dtm(result_dtm_path) as result:
        # Check elevation layers have no NaN
        for layer_name in interpolated_layers:
            if layer_name in i_dtm:
                assert layer_name in result, f"Layer {layer_name} missing in result DTM"

                layer_data = result[layer_name][:].data
                assert layer_data.shape == (50, 50), f"Bad shape for layer {layer_name}"
                nan_count = np.isnan(layer_data).sum()
                assert nan_count == 0, f"Layer {layer_name} still has {nan_count} NaN values after interpolation"

                _verify_no_change_outside_zone(layer_name, i_dtm[layer_name][:].data, layer_data)

        # Check interpolation flag: should mark all previously empty cells as interpolated
        layer_name = dtm_driver.DtmConstants.INTERPOLATION_FLAG
        interp_flag = result[layer_name][:].data
        interpolated_count = np.sum(interp_flag == 1)
        assert (
            interpolated_count == total_nodata_cells
        ), f"Expected {total_nodata_cells} interpolated cells, found {interpolated_count}"

        # Verify each zone is marked as interpolated
        for row_slice, col_slice in all_nodata_zones:
            zone_flags = interp_flag[row_slice, col_slice]
            assert np.all(
                zone_flags == 1
            ), f"Zone {row_slice}, {col_slice} should be fully interpolated in {layer_name}"

        if layer_name in i_dtm:
            _verify_no_change_outside_zone(layer_name, i_dtm[layer_name][:].data, interp_flag)

        # Check value_count: should be 1 for interpolated cells
        layer_name = dtm_driver.DtmConstants.VALUE_COUNT
        if layer_name in i_dtm:
            assert layer_name in result, f"Layer {layer_name} missing in result DTM"

            value_count = result[layer_name][:].data
            for row_slice, col_slice in all_nodata_zones:
                zone_counts = value_count[row_slice, col_slice]
                assert np.all(zone_counts == 1), f"Zone {row_slice}, {col_slice} should have value_count=1"
            _verify_no_change_outside_zone(layer_name, i_dtm[layer_name][:].data, value_count)

        # Check CDI index: should no longer be -1
        layer_name = dtm_driver.DtmConstants.CDI_INDEX
        if layer_name in i_dtm:
            assert layer_name in result, f"Layer {layer_name} missing in result DTM"

            cdi_index = result[layer_name][:].data
            for row_slice, col_slice in all_nodata_zones:
                zone_cdi = cdi_index[row_slice, col_slice]
                assert np.all(zone_cdi != -1), f"Zone {row_slice}, {col_slice} should have valid CDI index"
            _verify_no_change_outside_zone(layer_name, i_dtm[layer_name][:].data, cdi_index)


def _verify_partial_interpolation(
    in_dtm_path,
    result_dtm_path,
    all_nodata_zones,
    interpolated_zones,
    non_interpolated_zones,
    expected_interpolated_cells,
    interpolated_layers,
):
    """
    Verify that only specific zones have been interpolated.

    Args:
        dtm_path: Path to the interpolated DTM file
        all_nodata_zones: List of all no-data zone tuples
        interpolated_zones: Indices of zones that should be interpolated
        non_interpolated_zones: Indices of zones that should NOT be interpolated
        expected_interpolated_cells: Expected total number of interpolated cells
        interpolated_layers: List of layer names to check
    """
    with dtm_driver.open_dtm(in_dtm_path) as i_dtm, dtm_driver.open_dtm(result_dtm_path) as result:
        # Check elevation layers
        for layer_name in interpolated_layers:
            if layer_name in i_dtm:
                assert layer_name in result, f"Layer {layer_name} missing in result DTM"
                layer_data = result[layer_name][:].data
                assert layer_data.shape == (50, 50), f"Bad shape for layer {layer_name}"

                # Verify interpolated zones have no NaN
                for zone_idx in interpolated_zones:
                    row_slice, col_slice = all_nodata_zones[zone_idx]
                    zone_data = layer_data[row_slice, col_slice]
                    nan_count = np.isnan(zone_data).sum()
                    assert (
                        nan_count == 0
                    ), f"Zone {zone_idx} {row_slice}, {col_slice} should be interpolated in {layer_name}"

                # Verify non-interpolated zones still have NaN
                for zone_idx in non_interpolated_zones:
                    row_slice, col_slice = all_nodata_zones[zone_idx]
                    zone_data = layer_data[row_slice, col_slice]
                    zone_size = (row_slice.stop - row_slice.start) * (col_slice.stop - col_slice.start)
                    nan_count = np.isnan(zone_data).sum()
                    assert (
                        nan_count == zone_size
                    ), f"Zone {zone_idx} {row_slice}, {col_slice} should NOT be interpolated in {layer_name}"

                _verify_no_change_outside_zone(layer_name, i_dtm[layer_name][:].data, layer_data)

        # Check interpolation flag
        layer_name = dtm_driver.DtmConstants.INTERPOLATION_FLAG
        interp_flag = result[layer_name][:].data
        interpolated_count = np.sum(interp_flag == 1)
        assert (
            interpolated_count == expected_interpolated_cells
        ), f"Expected {expected_interpolated_cells} interpolated cells, found {interpolated_count}"

        # Verify interpolated zones
        for zone_idx in interpolated_zones:
            row_slice, col_slice = all_nodata_zones[zone_idx]
            zone_flags = interp_flag[row_slice, col_slice]
            assert np.all(zone_flags == 1), f"Zone {zone_idx} should be marked as interpolated"

        # Verify non-interpolated zones
        for zone_idx in non_interpolated_zones:
            row_slice, col_slice = all_nodata_zones[zone_idx]
            zone_flags = interp_flag[row_slice, col_slice]
            assert np.all(zone_flags == 127), f"Zone {zone_idx} should NOT be marked as interpolated"

        if layer_name in i_dtm:
            _verify_no_change_outside_zone(layer_name, i_dtm[layer_name][:].data, interp_flag)

        # Check value_count
        layer_name = dtm_driver.DtmConstants.VALUE_COUNT
        if layer_name in i_dtm:
            assert layer_name in result, f"Layer {layer_name} missing in result DTM"

            value_count = result[layer_name][:].data
            for zone_idx in interpolated_zones:
                row_slice, col_slice = all_nodata_zones[zone_idx]
                zone_counts = value_count[row_slice, col_slice]
                assert np.all(zone_counts == 1), f"Zone {zone_idx} should have value_count=1"

            for zone_idx in non_interpolated_zones:
                row_slice, col_slice = all_nodata_zones[zone_idx]
                zone_counts = value_count[row_slice, col_slice]
                assert np.all(zone_counts == -1), f"Zone {zone_idx} should still have value_count=-1"

            _verify_no_change_outside_zone(layer_name, i_dtm[layer_name][:].data, value_count)

        # Check CDI index
        layer_name = dtm_driver.DtmConstants.CDI_INDEX
        if layer_name in i_dtm:
            assert layer_name in result, f"Layer {layer_name} missing in result DTM"

            cdi_index = result[layer_name][:].data
            for zone_idx in interpolated_zones:
                row_slice, col_slice = all_nodata_zones[zone_idx]
                zone_cdi = cdi_index[row_slice, col_slice]
                assert np.all(zone_cdi != -1), f"Zone {zone_idx} should have valid CDI index"

            for zone_idx in non_interpolated_zones:
                row_slice, col_slice = all_nodata_zones[zone_idx]
                zone_cdi = cdi_index[row_slice, col_slice]
                assert np.all(zone_cdi == -1), f"Zone {zone_idx} should still have CDI index=-1"

            _verify_no_change_outside_zone(layer_name, i_dtm[layer_name][:].data, cdi_index)


def _verify_no_change_outside_zone(layer_name, in_data, result_data):
    """
    Verify that data outside the interpolated zones remains unchanged.
    """
    no_data_value = get_missing_value(layer_name)
    mask = ~np.isnan(in_data) if no_data_value is np.nan else (in_data != no_data_value)
    assert np.all(
        in_data[mask] == result_data[mask]
    ), f"Data outside interpolated zones should remain unchanged in '{layer_name}'"


# ============================================================================
# TESTS
# ============================================================================


@pytest.mark.parametrize("algo,algo_params", INTERPOLATION_ALGORITHMS)
@pytest.mark.parametrize("dtm_test_file", ["full", "simple"], indirect=True)
def test_interpolation_full_dtm(
    dtm_test_file, tmp_path, all_nodata_zones, total_nodata_cells, interpolated_layers, algo, algo_params
):
    """
    Test full DTM interpolation without area constraints.

    Verifies that all no-data zones are properly interpolated and that
    all metadata layers are correctly updated.
    """
    path_i_dtm = dtm_test_file
    path_o_dtm = tmp_path / "interpolated_full.dtm.nc"

    # Verify input DTM has expected no-data cells
    _verify_nodata_before_interpolation(path_i_dtm, all_nodata_zones, total_nodata_cells, interpolated_layers)

    # Perform interpolation
    process = algo(**algo_params)
    logger.info(f"Interpolation algo : {process.__class__.__name__} with params {algo_params}")
    interpolate_dtms([path_i_dtm], [path_o_dtm], process.interpolates)

    # Verify all zones are interpolated
    _verify_full_interpolation(path_i_dtm, path_o_dtm, all_nodata_zones, total_nodata_cells, interpolated_layers)


@pytest.mark.parametrize("algo,algo_params", INTERPOLATION_ALGORITHMS)
@pytest.mark.parametrize("dtm_test_file", ["full", "simple"], indirect=True)
def test_interpolation_with_all_areas(
    dtm_test_file,
    tmp_path,
    all_nodata_zones,
    total_nodata_cells,
    interpolated_layers,
    kml_all_areas,
    algo,
    algo_params,
):
    """
    Test full DTM interpolation with all area constraints.

    Verifies that all no-data zones are properly interpolated and that
    all metadata layers are correctly updated.
    """
    path_i_dtm = dtm_test_file
    path_o_dtm = tmp_path / "interpolated_all_areas.dtm.nc"

    # Verify input DTM has expected no-data cells
    _verify_nodata_before_interpolation(path_i_dtm, all_nodata_zones, total_nodata_cells, interpolated_layers)

    # Perform interpolation
    process = algo(**algo_params)
    logger.info(f"Interpolation algo : {process.__class__.__name__} with params {algo_params}")
    interpolate_dtms([path_i_dtm], [path_o_dtm], process.interpolates, areas=kml_all_areas)

    # Verify all zones are interpolated
    _verify_full_interpolation(path_i_dtm, path_o_dtm, all_nodata_zones, total_nodata_cells, interpolated_layers)


@pytest.mark.parametrize("algo,algo_params", INTERPOLATION_ALGORITHMS)
@pytest.mark.parametrize("dtm_test_file", ["full", "simple"], indirect=True)
def test_interpolation_with_single_area(
    dtm_test_file, tmp_path, all_nodata_zones, interpolated_layers, kml_nodata_zone_2, algo, algo_params
):
    """
    Test interpolation of a single area defined by KML.

    The KML area covers only the medium no-data zone [20:25, 35:40].
    Other zones should remain unchanged.
    """
    path_i_dtm = dtm_test_file
    path_o_dtm = tmp_path / "interpolated_single_area.dtm.nc"

    # Perform interpolation with area constraint
    process = algo(**algo_params)
    logger.info(f"Interpolation algo : {process.__class__.__name__} with params {algo_params}")
    interpolate_dtms([path_i_dtm], [path_o_dtm], process.interpolates, areas=kml_nodata_zone_2)

    # Verify only the medium zone (index 1) is interpolated
    # Zone 0 (small upper left) and zone 2 (lower center) should remain unchanged
    _verify_partial_interpolation(
        path_i_dtm,
        path_o_dtm,
        all_nodata_zones,
        interpolated_zones=[1],  # Only medium zone
        non_interpolated_zones=[0, 2],  # Small and lower zones
        expected_interpolated_cells=25,  # 5x5 zone
        interpolated_layers=interpolated_layers,
    )


@pytest.mark.parametrize("algo,algo_params", INTERPOLATION_ALGORITHMS)
@pytest.mark.parametrize("dtm_test_file", ["full", "simple"], indirect=True)
def test_interpolation_with_multiple_areas(
    dtm_test_file, tmp_path, all_nodata_zones, interpolated_layers, kml_nodata_zone_2_and_3, algo, algo_params
):
    """
    Test interpolation of multiple areas defined by KML.

    The KML areas cover:
    - Medium zone [20:25, 35:40] (25 cells)
    - Lower zone [40:44, 22:26] (16 cells)
    The small upper left zone [5:8, 5:8] should remain unchanged.
    """
    path_i_dtm = dtm_test_file
    path_o_dtm = tmp_path / "interpolated_multiple_areas.dtm.nc"

    # Perform interpolation with multiple area constraints
    process = algo(**algo_params)
    logger.info(f"Interpolation algo : {process.__class__.__name__} with params {algo_params}")
    interpolate_dtms([path_i_dtm], [path_o_dtm], process.interpolates, areas=kml_nodata_zone_2_and_3)

    # Verify zones 1 (medium) and 2 (lower) are interpolated
    # Zone 0 (small upper left) should remain unchanged
    _verify_partial_interpolation(
        path_i_dtm,
        path_o_dtm,
        all_nodata_zones,
        interpolated_zones=[1, 2],  # Medium and lower zones
        non_interpolated_zones=[0],  # Small zone only
        expected_interpolated_cells=41,  # 25 + 16 cells
        interpolated_layers=interpolated_layers,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
