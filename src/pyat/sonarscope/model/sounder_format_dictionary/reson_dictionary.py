"""A dictionary of known variables for reson files"""

import netCDF4 as nc
import numpy as np
import sonar_netcdf.sonar_groups as sg

import pyat.sonarscope.model.sounder_format_dictionary.common_dictionary as common
import pyat.utils.pyat_logger as log
from pyat.sensor.nmea import GPSQualityIndicator
from pyat.sonarscope.common import xarray_utils as ut
from pyat.sonarscope.model.constants import DEFAULT_BEAM_GROUP_IDENT, DefaultGroups
from pyat.sonarscope.model.constants import VariableKeys as k
from pyat.utils.netcdf import get_default_fillvalue

logger = log.logging.getLogger(__file__)


class IndicativeSwathPerPing(common.VariableInterface):
    """In case of .s7k data an indicative number of ping per swath is computed"""

    def get_dimensions(self, nc_dataset: nc.Dataset):
        # we have the same dimension as ping_time
        ping_time = nc_dataset[sg.BeamGroup1Grp.PING_TIME(ident=DEFAULT_BEAM_GROUP_IDENT)]
        return ut.get_dimensions(ping_time)

    def get_values(self, nc_dataset: nc.Dataset):
        """read variable values"""
        # we have got a runtime dual swath mode variable
        multiping_sequence = nc_dataset[sg.BathymetryGrp.MULTIPING_SEQUENCE(ident=DEFAULT_BEAM_GROUP_IDENT)]
        ping_time = nc_dataset[sg.BeamGroup1Grp.PING_TIME(ident=DEFAULT_BEAM_GROUP_IDENT)]
        max_multiping_sequence = np.amax(multiping_sequence[:])
        # multiping sequence equals 0 when disable
        fill_value = get_default_fillvalue(multiping_sequence.dtype)
        values = np.full_like(ping_time, fill_value=fill_value)
        # if enabled, number of swath is the max sequence number
        values[multiping_sequence[:] > 0] = max_multiping_sequence
        values[multiping_sequence[:] == 0] = 1  # if multiping_sequence is 0 we assume single swath
        values[multiping_sequence[:] < 0] = 0  # if value is negative, we set to zero
        return values

    def get_attributes(self, nc_dataset: nc.Dataset):
        return {"long_name": "Indicative number of swaths per ping"}

    def get_fill_value(self, nc_dataset: nc.Dataset):
        return 0


class RunTimeVariables(common.VariablesContainer, common.VariablesDictionary.RunTimeVariables):
    pass


class PositionVariables(common.VariablesContainer, common.VariablesDictionary.PositionVariables):
    def __init__(self, root_dataset: nc.Dataset):
        super().__init__()

        # First check for preferred position subgroup
        preferred_position = root_dataset[DefaultGroups.BEAM_GROUP_NAME].preferred_position
        # now retrieve the name of the sensor
        sensor_ids = root_dataset[sg.PlatformGrp.get_group_path()].variables[sg.PlatformGrp.POSITION_IDS_VNAME]
        sensor_id = sensor_ids[preferred_position]
        # check sensor id
        # sensor id are not always well set, we use default value if an error is in file
        # use netcdf api to ensure that group really exist
        if sensor_id not in root_dataset[sg.PositionGrp.get_group_path()].groups:
            logger.warning(
                f"Position sensor id {sensor_id} is not found in subgroups of {sg.PositionGrp.get_group_path()} \
                              ({root_dataset[sg.PositionGrp.get_group_path()].groups})"
            )
            # we use the first group found as default sensor_id
            sensor_id = next(iter(root_dataset[sg.PositionGrp.get_group_path()].groups))
            logger.warning(f"Use {sensor_id} as default sensor_id")

        instance = sg.PositionSubGroupVendorSpecificGrp()
        self.variables[k.POSITION_SENSOR_QUALITY_INDICATOR] = common.createXsfVariable(
            variable_name=sg.PositionSubGroupVendorSpecificGrp.POSITIONING_METHOD_VNAME,
            instance=instance,
            group=sg.PositionSubGroupVendorSpecificGrp,
            ident=sensor_id,
            fill_value=GPSQualityIndicator.NOT_AVAILABLE.value,
        )

        instance = sg.PositionSubGroup()
        self.variables[k.POSITION_SENSOR_TIME] = common.createXsfVariable(
            variable_name=sg.PositionSubGroup.TIME_VNAME,
            instance=instance,
            group=sg.PositionSubGroup,
            ident=sensor_id,
        )


class PingTimeVariables(common.VariablesContainer, common.VariablesDictionary.PingTimeVariables):
    def __init__(self, beam_group: str = DEFAULT_BEAM_GROUP_IDENT):
        """Build a key value dictionary referencing kmall variables from a given beam group of a xsf"""
        super().__init__()

        instance = sg.BeamGroup1Grp()
        group = sg.BeamGroup1Grp
        self.variables[k.PING_TIME] = common.createXsfVariable(
            variable_name=sg.BeamGroup1Grp.PING_TIME_VNAME, instance=instance, group=group, ident=beam_group
        )
        self.variables[k.PLATFORM_ROLL] = common.createXsfVariable(
            variable_name=sg.BeamGroup1Grp.PLATFORM_ROLL_VNAME, instance=instance, group=group, ident=beam_group
        )
        self.variables[k.PLATFORM_PITCH] = common.createXsfVariable(
            variable_name=sg.BeamGroup1Grp.PLATFORM_PITCH_VNAME, instance=instance, group=group, ident=beam_group
        )
        self.variables[k.PLATFORM_HEADING] = common.createXsfVariable(
            variable_name=sg.BeamGroup1Grp.PLATFORM_HEADING_VNAME, instance=instance, group=group, ident=beam_group
        )
        self.variables[k.PLATFORM_LONGITUDE] = common.createXsfVariable(
            variable_name=sg.BeamGroup1Grp.PLATFORM_LONGITUDE_VNAME, instance=instance, group=group, ident=beam_group
        )
        self.variables[k.PLATFORM_LATITUDE] = common.createXsfVariable(
            variable_name=sg.BeamGroup1Grp.PLATFORM_LATITUDE_VNAME, instance=instance, group=group, ident=beam_group
        )
        self.variables[k.SOUND_SPEED_AT_TRANSDUCER] = common.createXsfVariable(
            variable_name=sg.BeamGroup1Grp.SOUND_SPEED_AT_TRANSDUCER_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )

        instance = sg.BeamGroup1VendorSpecificGrp()
        group = sg.BeamGroup1VendorSpecificGrp

        self.variables[k.TRANSMIT_TYPE] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.TX_PULSE_TYPE_IDENTIFIER_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )

        self.variables[k.PULSE_FORM] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.TX_PULSE_ENVELOPE_IDENTIFIER_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.PULSE_LENGTH_EFFECTIVE] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.TX_PULSE_WIDTH_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.DETECTION_SAMPLING_FREQ] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.SAMPLE_RATE_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.TX_BEAMWIDTH] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.PROJECTOR_BEAM_MINUS3DB_BEAM_WIDTH_VERTICAL_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.RX_BEAMWIDTH] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.RECEIVE_BEAM_WIDTH_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.GAIN_SELECTION] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.GAIN_SELECTION_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.RANGE_SELECTION] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.RANGE_SELECTION_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.POWER_SELECTION] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.POWER_SELECTION_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.MEAN_ABS_COEFF] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.ABSORPTION_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )

        instance = sg.BathymetryGrp()
        group = sg.BathymetryGrp

        self.variables[k.MULTIPING_SEQUENCE] = common.createXsfVariable(
            variable_name=sg.BathymetryGrp.MULTIPING_SEQUENCE_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.DETECTION_Z] = common.createXsfVariable(
            variable_name=sg.BathymetryGrp.DETECTION_Z_VNAME, instance=instance, group=group, ident=beam_group
        )
        self.variables[k.DETECTION_QUALITY_FACTOR] = common.createXsfVariable(
            variable_name=sg.BathymetryGrp.DETECTION_QUALITY_FACTOR_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.BATHYMETRY_STATUS] = common.createXsfVariable(
            variable_name=sg.BathymetryGrp.STATUS_VNAME, instance=instance, group=group, ident=beam_group
        )

        instance = sg.BathymetryVendorSpecificGrp()
        group = sg.BathymetryVendorSpecificGrp

        self.variables[k.FREQUENCY_MODE] = common.createXsfVariable(
            variable_name=sg.BathymetryVendorSpecificGrp.DETECTION_PING_FREQUENCY_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )


class PingDetectionVariables(common.VariablesContainer, common.VariablesDictionary.PingDetectionVariables):
    def __init__(self, beam_group: str = DEFAULT_BEAM_GROUP_IDENT):
        """Build a key value dictionary referencing all variables from a given beam group of a xsf"""
        super().__init__()
        instance = sg.BeamGroup1Grp()
        group = sg.BeamGroup1Grp
        self.variables[k.TX_TILT_ANGLE_REF_VERTICAL] = common.createXsfVariable(
            variable_name=sg.BeamGroup1Grp.TX_BEAM_ROTATION_THETA_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )

        instance = sg.BathymetryGrp()
        group = sg.BathymetryGrp
        self.variables[k.DETECTION_BACKSCATTER] = common.createXsfVariable(
            variable_name=sg.BathymetryGrp.DETECTION_BACKSCATTER_R_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.DETECTION_BACKSCATTER_CALIBRATION] = common.createXsfVariable(
            variable_name=sg.BathymetryGrp.DETECTION_BACKSCATTER_CALIBRATION_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.DETECTION_BACKSCATTER_WITHOUT_COMP] = common.createXsfVariable(
            variable_name=sg.BathymetryGrp.DETECTION_BACKSCATTER_R_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )

        self.variables[k.DETECTION_BEAM_POINTING_ANGLE] = common.createXsfVariable(
            variable_name=sg.BathymetryGrp.DETECTION_BEAM_POINTING_ANGLE_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.DETECTION_TX_BEAM_INDEX] = common.createXsfVariable(
            variable_name=sg.BathymetryGrp.DETECTION_TX_BEAM_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.DETECTION_RX_TRANSDUCER_INDEX] = common.createXsfVariable(
            variable_name=sg.BathymetryGrp.DETECTION_RX_TRANSDUCER_INDEX_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.DETECTION_LONGITUDE] = common.createXsfVariable(
            variable_name=sg.BathymetryGrp.DETECTION_LONGITUDE_VNAME, instance=instance, group=group, ident=beam_group
        )
        self.variables[k.DETECTION_LATITUDE] = common.createXsfVariable(
            variable_name=sg.BathymetryGrp.DETECTION_LATITUDE_VNAME, instance=instance, group=group, ident=beam_group
        )

        instance = sg.BathymetryVendorSpecificGrp()
        group = sg.BathymetryVendorSpecificGrp
        self.variables[k.DETECTION_RANGE_SAMPLE] = common.createXsfVariable(
            variable_name=sg.BathymetryVendorSpecificGrp.SNIPPET_BOTTOM_DETECTION_SAMPLE_NUMBER_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )


class ComputedPingVariables(common.VariablesContainer, common.VariablesDictionary.ComputedPingVariables):
    def __init__(self):
        super().__init__()
        self.variables[k.INTERPINGS_DISTANCE] = common.InterpingVariable()
        self.variables[k.WC_PRESENCE] = common.WCPresence()
        self.variables[k.SWATH_PER_PING] = IndicativeSwathPerPing()


class ComputedPingDetectionVariables(
    common.VariablesContainer, common.VariablesDictionary.ComputedPingDetectionVariables
):
    def __init__(self):
        super().__init__()
        self.variables[k.DETECTION_BS_SNIPPETS_MEAN_WITHOUT_LAMBERT_COMP] = (
            common.DetectionBackscatterSnippetMeanValues()
        )
        self.variables[k.DETECTION_BEAM_POINTING_ANGLE_REF_VERTICAL] = common.DetectionPointingAngleVertical()
        self.variables[k.DETECTION_BEAM_POINTING_ANGLE_REF_PLATFORM] = common.DetectionPointingAnglePlatform()
        self.variables[k.DETECTION_INCIDENCE_ANGLE] = common.DetectionIncidenceAngle()
