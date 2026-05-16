#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile as tmp

import pytest

from pyat.function.evaluate_csv_grid import ExtentEvaluator, GeoboxEvaluator


def make_csv() -> str:
    """
    Creates a plain CSV file with 4 lon/lat
    """
    # CSV file in a XYZ format with 4 lines : NW, NE, SW, SE
    csv_content = """
                    -9.6838  ;52.4911; -100.0
                    -9.6828  ;52.4911; -110.0
                    -9.6838  ;52.4901; -120.0
                    -9.6828  ;52.4901; -130.0
        """
    path_csv = tmp.mktemp(suffix=".csv")
    with open(path_csv, "w") as csv_file:
        csv_file.write(csv_content.replace(" ", ""))
    return path_csv


def test_nominal_geobox_evaluator():
    """
    Evaluates the geobox of a CSV, ie min/max longitudes and latitudes of all points
    """
    path_csv = make_csv()
    try:
        evaluator = GeoboxEvaluator(i_paths=[path_csv], evaluate_spatial_resolution=True)
        evaluator()

        assert evaluator.spatial_resolution == pytest.approx(0.001)
        assert evaluator.geobox.upper == 52.4911
        assert evaluator.geobox.lower == 52.4901
        assert evaluator.geobox.left == -9.6838
        assert evaluator.geobox.right == -9.6828

    finally:
        os.remove(path_csv)


def test_extent_evaluator_center():
    """
    Evaluates the extent of a CSV, ie min/max longitudes and latitudes of the grid to have all points at the center of the cell
    """
    path_csv = make_csv()
    try:
        evaluator = ExtentEvaluator(i_paths=[path_csv])
        evaluator()

        assert evaluator.spatial_resolution == pytest.approx(0.001)
        assert evaluator.geobox.upper == pytest.approx(52.4911 + 0.0005)  # expected north + spatial_resolution / 2
        assert evaluator.geobox.lower == pytest.approx(52.4901 - 0.0005)  # expected south - spatial_resolution / 2
        assert evaluator.geobox.left == pytest.approx(-9.6838 - 0.0005)  # expected west - spatial_resolution / 2
        assert evaluator.geobox.right == pytest.approx(-9.6828 + 0.0005)  # expected east + spatial_resolution / 2

    finally:
        os.remove(path_csv)


def test_extent_evaluator_upper_left():
    """
    Evaluates the extent of a CSV, ie min/max longitudes and latitudes of the grid to have all points at the upper-left of the cell
    """
    path_csv = make_csv()
    try:
        evaluator = ExtentEvaluator(i_paths=[path_csv], pos_in_cell="upper-left")
        evaluator()

        assert evaluator.spatial_resolution == pytest.approx(0.001)
        assert evaluator.geobox.upper == pytest.approx(52.4911)  # expected north unchanged
        assert evaluator.geobox.lower == pytest.approx(52.4901 - 0.001)  # expected south - spatial_resolution
        assert evaluator.geobox.left == pytest.approx(-9.6838)  # expected west unchanged
        assert evaluator.geobox.right == pytest.approx(-9.6828 + 0.001)  # expected east + spatial_resolution

    finally:
        os.remove(path_csv)


def test_extent_evaluator_lower_left():
    """
    Evaluates the extent of a CSV, ie min/max longitudes and latitudes of the grid to have all points at the lower-left of the cell
    """
    path_csv = make_csv()
    try:
        evaluator = ExtentEvaluator(i_paths=[path_csv], pos_in_cell="lower-left")
        evaluator()

        assert evaluator.spatial_resolution == pytest.approx(0.001)
        assert evaluator.geobox.upper == pytest.approx(52.4911 + 0.001)  # expected north + spatial_resolution
        assert evaluator.geobox.lower == pytest.approx(52.4901)  # expected south unchanged
        assert evaluator.geobox.left == pytest.approx(-9.6838)  # expected west unchanged
        assert evaluator.geobox.right == pytest.approx(-9.6828 + 0.001)  # expected east + spatial_resolution

    finally:
        os.remove(path_csv)


def test_extent_evaluator_upper_right():
    """
    Evaluates the extent of a CSV, ie min/max longitudes and latitudes of the grid to have all points at the upper-right of the cell
    """
    path_csv = make_csv()
    try:
        evaluator = ExtentEvaluator(i_paths=[path_csv], pos_in_cell="upper-right")
        evaluator()

        assert evaluator.spatial_resolution == pytest.approx(0.001)
        assert evaluator.geobox.upper == pytest.approx(52.4911)  # expected north unchanged
        assert evaluator.geobox.lower == pytest.approx(52.4901 - 0.001)  # expected south - spatial_resolution
        assert evaluator.geobox.left == pytest.approx(-9.6838 - 0.001)  # expected west - spatial_resolution
        assert evaluator.geobox.right == pytest.approx(-9.6828)  # expected east + unchanged

    finally:
        os.remove(path_csv)


def test_extent_evaluator_lower_right():
    """
    Evaluates the extent of a CSV, ie min/max longitudes and latitudes of the grid to have all points at the lower-right of the cell
    """
    path_csv = make_csv()
    try:
        evaluator = ExtentEvaluator(i_paths=[path_csv], pos_in_cell="lower-right")
        evaluator()

        assert evaluator.spatial_resolution == pytest.approx(0.001)
        assert evaluator.geobox.upper == pytest.approx(52.4911 + 0.001)  # expected north + spatial_resolution
        assert evaluator.geobox.lower == pytest.approx(52.4901)  # expected south unchanged
        assert evaluator.geobox.left == pytest.approx(-9.6838 - 0.001)  # expected west - spatial_resolution
        assert evaluator.geobox.right == pytest.approx(-9.6828)  # expected east unchanged

    finally:
        os.remove(path_csv)
