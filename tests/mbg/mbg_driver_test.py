#! /usr/bin/env python3
# coding: utf-8

import pytest

from pyat.sounder import sounder_driver_factory
from tests.file_test_installer import get_test_path

MBG_PATH = get_test_path() / "mbg" / "0136_20120607_083636_ShipName_ref.mbg"


def test_generated_read_method():
    """
    Verify method __read_layer and __read_layer_as at the same time by reading layer heading
    """
    with sounder_driver_factory.open_sounder(MBG_PATH) as i_driver:
        # all headings
        assert len(i_driver.read_heading()) == 1176
        # read the first 2
        assert len(i_driver.read_heading(to_index=2)) == 2
        # read the last 2
        assert len(i_driver.read_heading(-2)) == 2
        # read one by one
        assert len(i_driver.read_heading(0, 1)) == 1
        assert len(i_driver.read_heading(-1)) == 1
        assert len(i_driver.read_heading(1175, 1176)) == len(i_driver.read_heading(-1))


def test_heading():
    """
    Verify method read_heading
    """
    with sounder_driver_factory.open_sounder(MBG_PATH) as i_driver:
        # read first beams
        headings = i_driver.read_heading(0)
        assert (headings[0, 0], headings[0, 1]) == (92.01, 92.01)
        # read last beams
        headings = i_driver.read_heading(-1)
        assert (headings[0, 0], headings[0, 1]) == (277.5, 277.5)
        # Heading negative. -29615 (8C51) in file is 359.21°
        headings = i_driver.read_heading(1053)
        assert (headings[0, 0], headings[0, 1]) == (359.21, 359.21)


def test_across_distance():
    """
    Verify methods read_across_distances, read_across_distance and read_distance_scale
    The expected values must be the same as those computed by Java class BeamLatLonLayerLoader
    """
    with sounder_driver_factory.open_sounder(MBG_PATH) as i_driver:
        # Expected values picked up from java
        java_across_0_0 = -474.16
        java_across_0_1 = 452.54

        # Redefined methods of SounderDriver
        across = i_driver.read_across_distances(0, 1)  # read values of first swath
        assert (across[0, 0], across[0, -1]) == (java_across_0_0, java_across_0_1)

        # Generated methods in MbgDriver to read raw data
        raw_across = i_driver.read_across_distance(0)
        assert (raw_across[0, 0], raw_across[0, -1]) == (-23708, 22627)
        distance_scale = i_driver.read_distance_scale(0)
        assert (distance_scale[0, 0], distance_scale[0, 1]) == (
            java_across_0_0 / raw_across[0, 0],
            java_across_0_1 / raw_across[0, -1],
        )


def test_across_angles():
    """
    Verify method read_across_angles
    The expected values must be the same as those computed by Java class BeamLatLonLayerLoader
    """
    with sounder_driver_factory.open_sounder(MBG_PATH) as i_driver:
        angles = i_driver.read_across_angles(0, 1)  # read values of first swath
        assert (angles[0, 0], angles[0, -1]) == (-47.7, 50.0)


def test_along_distance():
    """
    Verify methods read_along_distance and read_distance_scale
    The expected values must be the same as those computed by Java class BeamLatLonLayerLoader
    """
    with sounder_driver_factory.open_sounder(MBG_PATH) as i_driver:
        # Expected values picked up from java
        java_along_0_0 = -45.92
        java_along_0_1 = 41.22

        # Generated methods in MbgDriver to read raw data
        raw_along = i_driver.read_along_distance(0)
        assert (raw_along[0, 0], raw_along[0, -1]) == (-2296, 2061)
        distance_scale = i_driver.read_distance_scale(0)
        assert (distance_scale[0, 0], distance_scale[0, 1]) == (
            java_along_0_0 / raw_along[0, 0],
            java_along_0_1 / raw_along[0, -1],
        )


def test_iter_beam_positions():
    """
    Verify methods iter_beam_positions
    The expected values must be the same as those computed by Java class BeamLatLonLayerLoader
    """
    with sounder_driver_factory.open_sounder(MBG_PATH) as i_driver:
        # Expected values picked up from java
        longitudes, latitudes = next(i_driver.iter_beam_positions(1))
        assert longitudes[0, 0] == pytest.approx(-1.5656930349700293)
        assert latitudes[0, 0] == pytest.approx(43.6770568621721)
        assert longitudes[0, -1] == pytest.approx(-1.5650162059094737)
        assert latitudes[0, -1] == pytest.approx(43.66869379095998)


def test_read_reflectivities():
    """
    Verify method read_reflectivities and read_reflectivity
    The expected values must be the same as those computed by Java class SounderDataContainerVariablesAdapter
    """
    with sounder_driver_factory.open_sounder(MBG_PATH) as i_driver:
        # Redefined methods of SounderDriver
        backscatter = i_driver.read_reflectivities(0, 1)  # read values of first swath
        assert (backscatter[0, 0], backscatter[0, -1]) == (-45.5, -44.0)
        # Generated methods in MbgDriver
        backscatter = i_driver.read_reflectivity(0)
        assert (backscatter[0, 0], backscatter[0, -1]) == (-45.5, -44.0)


def test_read_fcs_depths():
    """
    Verify methods read_fcs_depths and read_depth
    The expected values must be the same as those computed by Java class SounderDataContainerVariablesAdapter.getDepth
    """
    with sounder_driver_factory.open_sounder(MBG_PATH) as i_driver:
        # Redefined methods of SounderDriver
        depths = i_driver.read_fcs_depths(0, 1)  # read values of first swath
        assert (depths[0, 0], depths[0, -1]) == (444.41, 390.98)
        # Generated methods in MbgDriver
        depths = i_driver.read_depth(0)
        assert (depths[0, 0], depths[0, -1]) == (444.41, 390.98)


def test_read_scs_depths():
    """
    Verify methods read_fcs_depths and read_depth
    The expected values must be the same as those computed by Java class SounderDataContainerVariablesAdapter
    """
    with sounder_driver_factory.open_sounder(MBG_PATH) as i_driver:
        # Check some depths of valid beam in first swath
        depths = i_driver.read_scs_depths(0, 1)
        assert depths[0, -1] == pytest.approx(390.07, rel=0.01)
        assert depths[0, 118] == pytest.approx(440.80, rel=0.01)


def test_read_validity_flags():
    """
    Verify methods read_validity_flags
    The expected values must be the same as those computed by Java class SounderDataContainerVariablesAdapter
    """
    with sounder_driver_factory.open_sounder(MBG_PATH) as i_driver:
        # Redefined methods of SounderDriver
        # First valid beam : 118
        flags = i_driver.read_validity_flags(0, 1)  # read values of first swath
        assert (flags[0, 0], flags[0, 117], flags[0, 118], flags[0, -1]) == (False, False, True, False)
