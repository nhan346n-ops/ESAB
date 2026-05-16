#! /usr/bin/env python3
# coding: utf-8

import tempfile as tmp

import numpy as np

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.analyse.dtm_expected_stdev as es
import pyat.dtm.dtm_standard_constants as DTM
import tests.generator.dtm_generator as dtm_generator


def make_dtm(temp_dir: str) -> str:
    """
    Generates a DTM
    """
    elevations = np.ma.array(
        [
            [-54.70191, -54.437305, -54.125137, -53.907986],
            [-55.05824, -54.758415, -54.486416, -54.19614],
            [-55.11922, -54.827374, -54.536915, -54.25526],
        ]
    )
    value_count = np.ma.array([[3, 197, 326, 354], [482, 559, 555, 546], [508, 570, 633, 625]])
    max_across_distance = np.ma.array(
        [
            [-275.54135, -130.72578, -80.13309, -60.2301],
            [-40.13771, -20.15288, 20.64378, 40.66545],
            [60.93497, 80.60858, 130.03212, 275.828766],
        ],
    )

    return dtm_generator.make_dtm_with_data(
        (-30.0, 40.0),
        (-29.5, 39.5),
        {
            DTM.ELEVATION_NAME: elevations,
            DTM.VALUE_COUNT: value_count,
            DTM.MAX_ACROSS_DISTANCE: max_across_distance,
        },
        temp_dir,
    )


def make_csv(temp_dir: str) -> str:
    """
    Generates a DTM
    """
    csv_content = """   Angle ; Expected stdev
                        -70 ; 1
                        -60 ; 0,7
                        -50 ; 0,4
                        -40 ; 0,3
                        -30 ; 0,2
                        -20 ; 0,2
                        -10 ; 0,2
                        0 ; 0,2
                        10 ; 0,2
                        20 ; 0,2
                        30 ; 0,2
                        40 ; 0,3
                        50 ; 0,4
                        60 ; 0,7
                        70 ; 1
        """
    path_csv = tmp.mktemp(dir=temp_dir, suffix=".csv")
    with open(path_csv, "wt", encoding="utf8") as csv_file:
        csv_file.write(csv_content.replace(" ", ""))
    return path_csv


def test_computes_expected_stdev():
    """
    Test function load_stdev_in_csv, _computes_beam_angle and _interpolate_stdev.
    """

    with tmp.TemporaryDirectory() as temp_dir:
        path_csv = make_csv(temp_dir)
        path_dtm = make_dtm(temp_dir)

        # Test CSV loading
        angle_stdev = es.load_stdev_in_csv(path_csv)
        assert angle_stdev.angle[0] == -70.0
        assert angle_stdev.stdev[0] == 1.0

        # Interpolates stdev
        with dtm_driver.open_dtm(path_dtm) as dtm:
            beam_angle = es._computes_beam_angle(dtm)
            assert np.allclose(
                beam_angle,
                [[78.77, 67.39, 55.96, 48.17], [36.09, 20.20, -20.75, -36.88], [-47.87, -55.78, -67.24, -78.87]],
                atol=1e-2,
            )

            interpolated_stdev = es._interpolate_stdev(beam_angle, angle_stdev)
            assert np.allclose(
                interpolated_stdev,
                [
                    [1.0, 0.92, 0.58, 0.38],  # Angle is 78.77, stdev is 1.0 according to the CSV file...
                    [0.26, 0.2, 0.2, 0.27],  # Angle is 36.09 (between 30 an 40), stdev is between 0.2 and 0.3
                    [0.38, 0.57, 0.92, 1.0],  # Angle is -47.87, stdev is between 0.3 and 0.4
                ],
                atol=1e-2,
            )
