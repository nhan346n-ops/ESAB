from pyat.sensor.phins_repeater_driver import read_phins_repeater_as_df
from tests.file_test_installer import get_test_path

PHINS_repeater_reference_file_1 = get_test_path() / "phins_repeater" / "PHINS_REPEATER_2024-06-06_17-04-56_test.txt"
PHINS_repeater_reference_file_2 = get_test_path() / "phins_repeater" / "PHINS_REPEATER_07042025_042227.log"


def test_phins_repeater_file_1_reading():
    """
    Verify phins repeater driver with test file from pyat_test_file
    """

    phins_data = read_phins_repeater_as_df(str(PHINS_repeater_reference_file_1))

    assert phins_data.shape == (13, 53)  # 13 data cycles with 53 data field decoded
    assert phins_data.index.dtype == "datetime64[ns]"  # index is datetime64[ns] type ...
    assert phins_data.index.is_monotonic_increasing  # ... and monotonically increasing, thus managing day change


def test_phins_repeater_file_2_reading():
    """
    Verify phins repeater driver with test file from pyat_test_file
    """

    phins_data = read_phins_repeater_as_df(str(PHINS_repeater_reference_file_2))

    assert phins_data.shape == (9, 39)  # 9 data cycles with 39 data field decoded
    assert phins_data.index.dtype == "datetime64[ns]"  # index is datetime64[ns] type ...
    assert phins_data.index.is_monotonic_increasing  # ... and monotonically increasing, thus managing day change
