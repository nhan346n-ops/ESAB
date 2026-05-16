import os.path
import tempfile as tmp

import numpy as np

from pyat.sonarscope.bs_correction.mean_bs_model import BackscatterCurveByIncidence, BackscatterCurveByTransmission
from pyat.sonarscope.bs_correction.stats_computer import MeanBSModel
from pyat.sonarscope.model import sounder_lib
from pyat.sonarscope.model.sounder_mode.all_EM2040_mode import KeyModeAllEM2040
from pyat.sonarscope.model.sounder_mode.all_kongsberg_mode import KeyModeAllGeneric
from pyat.sonarscope.model.sounder_mode.common_mode import KeyModeCommon


def test_mode_2040_none():
    """Check that __eq__ is well generated"""

    mode1 = KeyModeAllEM2040(
        frequency_mode=None, swath_mode=1, swath_index=0, sector_count=8, scanning_mode=False, pulse_length_mode=100
    )
    assert not mode1.is_valid()
    serial = mode1.mode_to_json()
    mode2 = KeyModeAllEM2040.mode_from_json(serial)
    assert not mode2.is_valid()
    mode_valid = KeyModeAllEM2040(
        frequency_mode=100000, swath_mode=1, swath_index=0, sector_count=8, scanning_mode=False, pulse_length_mode=100
    )
    assert mode_valid


def test_mode_2040_eq():
    """Check that __eq__ is well generated"""

    mode1 = KeyModeAllEM2040(
        frequency_mode=200000, swath_mode=1, swath_index=0, sector_count=8, scanning_mode=False, pulse_length_mode=100
    )
    mode2 = KeyModeAllEM2040(
        frequency_mode=200000, swath_mode=1, swath_index=0, sector_count=8, scanning_mode=False, pulse_length_mode=100
    )
    mode3 = KeyModeAllEM2040(
        frequency_mode=200000, swath_mode=1, swath_index=0, sector_count=8, scanning_mode=True, pulse_length_mode=100
    )
    mode4 = KeyModeAllEM2040(
        frequency_mode=100000, swath_mode=1, swath_index=0, sector_count=8, scanning_mode=False, pulse_length_mode=100
    )
    assert mode1 == mode2
    assert mode3 != mode2
    assert mode4 != mode2


def test_mode_2040_types():
    """Check if types are well managed even through serialization to json"""
    mode1 = KeyModeAllEM2040(
        frequency_mode=200000, swath_mode=1, swath_index=0, sector_count=8, scanning_mode=False, pulse_length_mode=100
    )
    serial = mode1.mode_to_json()
    mode2 = KeyModeAllEM2040.mode_from_json(serial)
    assert mode2 == mode1
    assert isinstance(mode1.frequency_mode, float)
    assert isinstance(mode1.swath_mode, int)
    assert isinstance(mode1.sector_count, int)
    assert isinstance(mode1.scanning_mode, bool)
    assert isinstance(mode1.pulse_length_mode, int)


def test_mode_2040_hash():
    """Check that hash is generated and can be used in dict"""
    mode1 = KeyModeAllEM2040(
        frequency_mode=200000, swath_mode=1, swath_index=0, sector_count=8, scanning_mode=False, pulse_length_mode=100
    )
    mode2 = KeyModeAllEM2040(
        frequency_mode=200000, swath_mode=2, swath_index=0, sector_count=8, scanning_mode=False, pulse_length_mode=100
    )

    dict_mode = {mode1: 1, mode2: 2}

    new_mode1 = KeyModeAllEM2040(
        frequency_mode=200000, swath_mode=1, swath_index=0, sector_count=8, scanning_mode=False, pulse_length_mode=100
    )
    new_mode2 = KeyModeAllEM2040(
        frequency_mode=200000, swath_mode=2, swath_index=0, sector_count=8, scanning_mode=False, pulse_length_mode=100
    )

    assert new_mode1 in dict_mode
    assert dict_mode[new_mode1] == 1
    assert dict_mode[new_mode2] == 2


def test_serialization_2040():
    """Create a curve model, serialize it, deserialize it and compare values"""
    mode1 = KeyModeAllEM2040(
        frequency_mode=200000, swath_mode=1, swath_index=0, sector_count=8, scanning_mode=False, pulse_length_mode=100
    )
    mode2 = KeyModeAllEM2040(
        frequency_mode=100000, swath_mode=1, swath_index=0, sector_count=8, scanning_mode=False, pulse_length_mode=100
    )
    bin_centers = np.arange(0, 10)
    mode_curves = {
        mode1: (
            BackscatterCurveByIncidence.build(
                mean_values=np.random.random(bin_centers.shape[0]),
                count=np.full(fill_value=1, shape=(bin_centers.shape[0])),
                bin_centers=bin_centers,
                origin=None,
            ),
            BackscatterCurveByTransmission.build(
                rx_antenna_count=1,
                tx_beam_count=1,
                mean_values=np.random.random((1, 1, bin_centers.shape[0])),
                mean_residual_values=np.random.random((1, 1, bin_centers.shape[0])),
                count=np.full(fill_value=1, shape=(1, 1, bin_centers.shape[0])),
                bin_centers=bin_centers,
                origin=None,
            ),
        ),
        mode2: (
            BackscatterCurveByIncidence.build(
                mean_values=np.random.random(bin_centers.shape[0]),
                count=np.full(fill_value=2, shape=(bin_centers.shape[0])),
                bin_centers=bin_centers,
                origin=None,
            ),
            BackscatterCurveByTransmission.build(
                rx_antenna_count=1,
                tx_beam_count=1,
                mean_values=np.random.random((1, 1, bin_centers.shape[0])),
                mean_residual_values=np.random.random((1, 1, bin_centers.shape[0])),
                count=np.full(fill_value=2, shape=(1, 1, bin_centers.shape[0])),
                bin_centers=bin_centers,
                origin=None,
            ),
        ),
    }
    curves = MeanBSModel(sounder_type=sounder_lib.SounderType.EM2040_ALL, mode_curves=mode_curves)

    filepath = tmp.mktemp(suffix=".nc")
    try:
        # Write file to netcdf
        curves.save_to_netcdf(output_file=filepath)

        # Read file from netcdf
        curves_read = curves.read_from_netcdf(input_file=filepath)

        assert curves == curves_read

        # release resources linked to filepath
        curves_read = None
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_mode_common_eq():
    """Check that __eq__ is well generated"""

    mode1 = KeyModeCommon()
    mode2 = KeyModeCommon()
    assert mode1 == mode2


def test_mode_kongsberg_eq():
    """Check that __eq__ is well generated for kongsberg mode KeyModeAllGeneric"""
    mode1 = KeyModeAllGeneric(
        ping_mode=1, pulse_form=1, swath_mode=1, swath_index=0, sector_count=8, center_frequency=(200000, 200000)
    )
    # swath index is not used in __eq__ method
    mode2 = KeyModeAllGeneric(
        ping_mode=1, pulse_form=1, swath_mode=1, swath_index=1, sector_count=8, center_frequency=(200000, 200000)
    )
    assert mode1 == mode2
    # use center_frequency as swath differanciator
    mode3 = KeyModeAllGeneric(
        ping_mode=1, pulse_form=1, swath_mode=1, swath_index=0, sector_count=8, center_frequency=(190000, 210000)
    )
    assert mode3 != mode1
    # if center_frequency is None, use swath_index as differanciator
    mode4 = KeyModeAllGeneric(
        ping_mode=1, pulse_form=1, swath_mode=1, swath_index=0, sector_count=8, center_frequency=None
    )
    assert mode4 == mode1
    assert mode4 != mode2
    assert mode4 == mode3
