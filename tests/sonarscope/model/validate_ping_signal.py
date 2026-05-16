import sonar_netcdf.sonar_groups as sg

import pyat.sonarscope.model.signal.ping_signal as ps
from pyat.xsf import xsf_driver


def test_multiple_read():
    """read variables in two differents groups and check that they are retrieved in ping/time model"""

    # Test file is not avalaible for Jenkins, need to find another way to do the tests
    input_file = r"C:\data\datasets\Backscatter\Compensation\THALIA_ESSDEC2019\EM2040\XSF\reduced\0000_20190427_071709_Bassop.xsf.nc"
    xsf = xsf_driver.XsfDriver(file_path=input_file)
    try:
        xsf.open()
    except KeyError:
        # we expect a key error due to almost empty xsf file
        pass
    BEAM_GROUP_NAME = sg.BeamGroup1Grp.get_group_path(ident="Beam_group1")

    model = ps.PingSignal(xsf_dataset=xsf)
    model.read(
        [
            sg.BathymetryGrp.DETECTION_Z_VNAME,
            sg.BeamGroup1Grp.PLATFORM_HEADING_VNAME,
            sg.BeamGroup1Grp.PLATFORM_ROLL_VNAME,
            sg.BeamGroup1Grp.PLATFORM_PITCH_VNAME,
            sg.BeamGroup1Grp.PLATFORM_LONGITUDE_VNAME,
            sg.BeamGroup1Grp.PLATFORM_LATITUDE_VNAME,
            sg.BeamGroup1Grp.PING_TIME_VNAME,
            sg.BeamGroup1Grp.SOUND_SPEED_AT_TRANSDUCER_VNAME,
            sg.BeamGroup1Grp.TRANSMIT_TYPE_VNAME,
            sg.BeamGroup1Grp.SAMPLE_COUNT_VNAME,
        ]
    )
    assert sg.BeamGroup1Grp.PLATFORM_HEADING_VNAME in model
    assert sg.BeamGroup1Grp.PLATFORM_ROLL_VNAME in model
    assert sg.BeamGroup1Grp.PLATFORM_PITCH_VNAME in model
    assert sg.BeamGroup1Grp.PLATFORM_LONGITUDE_VNAME in model
    assert sg.BeamGroup1Grp.PLATFORM_LATITUDE_VNAME in model
    assert sg.BeamGroup1Grp.PING_TIME_VNAME in model
    assert sg.BeamGroup1Grp.SOUND_SPEED_AT_TRANSDUCER_VNAME in model
    assert sg.BeamGroup1Grp.TRANSMIT_TYPE_VNAME in model
    assert sg.BeamGroup1Grp.SAMPLE_COUNT_VNAME in model

    model.read(
        [
            sg.BathymetryGrp.DETECTION_Z_VNAME,
            sg.BathymetryGrp.STATUS_VNAME,
            sg.BathymetryGrp.DETECTION_QUALITY_FACTOR_VNAME,
        ],
        ignore_unknown_variables=True,
    )
    assert sg.BathymetryGrp.DETECTION_Z_VNAME in model
    assert sg.BathymetryGrp.STATUS_VNAME in model

    # assert sg.BathymetryGrp.DETECTION_QUALITY_FACTOR_VNAME  in model missing in file example

    assert True
