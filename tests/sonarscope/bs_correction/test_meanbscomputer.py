import numpy as np
from pyat.sonarscope.bs_correction import mean_bs_utils


def test_mean_reduction():
    """test means reduction recomputing means from several files to 1 curve"""
    two_tx_values = np.array(([10, 20, 30], [2, 4, 6]), dtype=float)
    two_tx_count = np.array(([1, 1, 1], [2, 2, 2]), dtype=float)

    # check with duplicate values
    mean, count = mean_bs_utils.compute_means(
        means_per_file=[two_tx_values, two_tx_values], count_per_file=[two_tx_count, two_tx_count]
    )
    assert np.array_equal(count, 2 * two_tx_count)
    assert np.array_equal(two_tx_values, mean)

    two_tx_values_2 = np.array(([10, 0, 10], [2, 1, 0]), dtype=float)
    two_tx_count_2 = np.array(([1, 1, 1], [1, 1, 1]), dtype=float)

    mean, count = mean_bs_utils.compute_means(
        means_per_file=[two_tx_values, two_tx_values_2], count_per_file=[two_tx_count, two_tx_count_2]
    )

    assert np.array_equal(count, two_tx_count + two_tx_count_2)
    assert np.array_equal(mean, np.array(([10, 10, 20], [2, 3, 4]), dtype=float))
