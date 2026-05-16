#! /usr/bin/env python3
# coding: utf-8

import tempfile as tmp

import numpy as np

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.analyse.dtm_quality_indicator as qi
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
    filtered_count = np.ma.masked_equal([[1, 13, 2, 70], [18, -1, -1, 11], [-1, -1, -1, -1]], -1)
    interpolation_flag = np.ma.masked_equal([[127, 127, 127, 127], [127, 127, 1, 127], [127, 127, 127, 127]], 127)
    max_across_distance = np.ma.array(
        [
            [-275.54135, -252.72578, -237.13309, -224.2301],
            [-227.13771, -202.15288, -184.64378, -169.66545],
            [-182.93497, -154.60858, -132.03212, -114.828766],
        ],
    )
    max_across_angle = np.ma.array(
        [
            [-73.62824, -72.22006, -71.25231, -70.29183],
            [-70.06607, -67.9011, -66.07077, -64.38563],
            [-65.36893, -61.70848, -57.816456, -54.45451],
        ]
    )
    stdev = np.ma.array(
        [
            [0.22910747, 0.16387638, 0.1964032, 0.16086923],
            [0.15546678, 0.508233, 0.2829808, 0.18421604],
            [0.09375, 0.1158781, 0.1158781, 0.107119605],
        ]
    )

    return dtm_generator.make_dtm_with_data(
        (-30.0, 40.0),
        (-29.5, 39.5),
        {
            DTM.ELEVATION_NAME: elevations,
            DTM.VALUE_COUNT: value_count,
            DTM.FILTERED_COUNT: filtered_count,
            DTM.MAX_ACROSS_DISTANCE: max_across_distance,
            DTM.MAX_ACCROSS_ANGLE: max_across_angle,
            DTM.STDEV: stdev,
            DTM.INTERPOLATION_FLAG: interpolation_flag,
        },
        temp_dir,
    )


def test_computes_all_flags():
    """
    Test function computes_angle_of_incidence.
    """
    with tmp.TemporaryDirectory() as temp_dir:
        path_i_dtm = make_dtm(temp_dir)
        args = qi.QualityIndicatorArgs(i_paths=[path_i_dtm])
        with dtm_driver.open_dtm(path_i_dtm) as i_driver:
            layers = qi.load_layers(i_driver)
            flags = qi.computes_all_flags(layers, args)

            # Check angle of incidence
            assert flags.angle_of_incidence[2, 0] == 1  # -65.36893 > 65°
            assert flags.angle_of_incidence[2, 1] == 0  # -61.70848 < 65°

            # Check flag sufficient nb sound
            assert flags.sufficient_nb_sound[0, 0] == 8  # Only 4 sounds
            assert flags.sufficient_nb_sound[0, 1] == 0  # 210 sounds

            # Check rate of invalidated sounds
            # Mean is 4.4%
            assert flags.rate_of_invalidated_sounds[0, 2] == 0  # Only 0.6% invalidated
            assert flags.rate_of_invalidated_sounds[0, 3] == 2  # 16.5%, more than mean

            # Check compared_stdev
            assert flags.compared_stdev[1, 0] == 0  #
            #  Stdev at [1, 1] is 0.508233
            # Mean neighbourhood stdev is 0.16916761
            # Computed stdev with neighbouring elevations is 0.3122861
            # (0.508233 - 0.16916761) / 0.3122861 == 1.0857525 (> 1)
            assert flags.compared_stdev[1, 1] == 4

            # Check interpolation
            assert flags.interpolation[1, 1] == 0
            assert flags.interpolation[1, 2] == 16
