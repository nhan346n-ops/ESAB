#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile as tmp
from os import PathLike
from pathlib import Path

import numpy as np
import pytest

import pyat.dtm.dtm_standard_constants as DTM
import pyat.dtm.analyse.dtm_statistics as dtm_statistics
import tests.generator.dtm_generator as dtm_generator


def make_dtm(temp_dir: PathLike) -> PathLike:
    """
    Generates a DTM with elevations and backscatter
    """
    elevations = np.array([[-1, -2, -3], [-4, np.nan, -5], [-6, -7, -8]])
    backscatter = np.array([[-10, 11, 15], [-30, np.nan, -31], [21, 22, 23]])
    value_count = np.array([[10, 10, 15], [20, np.nan, 20], [30, 35, 35]])
    dtm_path = dtm_generator.make_dtm_with_data(
        (3.0, 3.1),
        (4.0, 4.1),
        {DTM.ELEVATION_NAME: elevations, DTM.VALUE_COUNT: value_count, DTM.BACKSCATTER: backscatter},
        temp_dir,
    )
    return Path(dtm_path)


def test_nominal_dtm_statistics():
    """
    Nominal test dtm_statistics.
    Calculates statistics on layer elevation and backscatter
    """
    with tmp.TemporaryDirectory() as temp_dir:
        path_i_dtm = make_dtm(temp_dir)
        args = dtm_statistics.StatArgs(
            i_paths=[path_i_dtm],
            output_dir=temp_dir,
            histogram_bins=2,
            confidence_level=50,
            confidence_interval_min=-15.0,
            confidence_interval_max=21.25,
            confidence_interval_1_sigma=True,
            confidence_interval_2_sigma=True,
        )
        all_metrics = dtm_statistics.computes_with_statArgs(args)

        # Check elevations
        metrics = [metrics for metrics in all_metrics.metrics if metrics.layer == DTM.ELEVATION_NAME][0]
        assert metrics.mean == -4.5
        assert metrics.std == pytest.approx(2.449489)
        assert metrics.median == -4.5  # Median
        # Confidence stats not avalaible for elevations
        assert np.all(
            np.isnan(
                [
                    metrics.min_confidence_interval,
                    metrics.max_confidence_interval,
                    metrics.confidence_level_on_interval,
                    metrics.confidence_level_1_sigma,
                    metrics.confidence_level_2_sigma,
                ]
            )
        )

        # Check backscatter
        metrics = [metrics for metrics in all_metrics.metrics if metrics.layer == DTM.BACKSCATTER][0]
        assert metrics.mean == 2.625
        assert metrics.std == pytest.approx(23.008926)
        assert metrics.median == 13.0
        assert metrics.min_confidence_interval == -15.0 and metrics.max_confidence_interval == 21.25
        assert metrics.confidence_level_on_interval == 50
        assert metrics.confidence_level_1_sigma == 75.0
        assert metrics.confidence_level_2_sigma == 100.0

        # Check csv
        assert Path(temp_dir, "metrics.csv").exists()


def test_dtm_statistics_with_value_count_filtering():
    """
    Same test but cells are filtered on value count.
    """
    with tmp.TemporaryDirectory() as temp_dir:
        path_i_dtm = make_dtm(temp_dir)
        dtm_filename = os.path.basename(path_i_dtm)
        args = dtm_statistics.StatArgs(
            i_paths=[path_i_dtm],
            layers=[DTM.ELEVATION_NAME, DTM.BACKSCATTER],
            output_dir=temp_dir,
            min_valid_sounds=15,
        )
        all_metrics = dtm_statistics.computes_with_statArgs(args)

        # Check backscatter
        # filtered backscatter : [[ --, --, 15], [-30, np.nan, -31], [21, 22, 23]])
        # filtered value_count : [[ --, --, 15], [ 20, np.nan,  20], [30, 35, 35]])
        metrics = [metrics for metrics in all_metrics.metrics if metrics.layer == DTM.BACKSCATTER][0]

        assert metrics.mean == pytest.approx((15.0 - 30.0 - 31.0 + 21.0 + 22.0 + 23.0) / 6.0, abs=1e-6)
        assert metrics.median == 18.0
