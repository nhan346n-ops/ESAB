#! /usr/bin/env python3
# coding: utf-8
import pytest

from pyat.function.evaluate_sounder_spatial_resolution import SpatialResolutionEvaluator
from tests.file_test_installer import get_test_path

MBG_PATH = get_test_path() / "mbg" / "0136_20120607_083636_ShipName_ref.mbg"


def test_evaluate_spatial_resolution_mbg():
    """
    Evaluates the resolution spatial of a MBG
    """
    evaluator = SpatialResolutionEvaluator(i_paths=[MBG_PATH])
    report = evaluator()
    assert report["result"]["meter"] == pytest.approx(11)
    assert report["result"]["degree"] == pytest.approx(1.363979e-4, abs=1e-7)
