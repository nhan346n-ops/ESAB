"""A dictionary of known variables for kmall files"""

import netCDF4 as nc
import numpy as np
import sonar_netcdf.sonar_groups as sg
from scipy.interpolate import interp1d

import pyat.sonarscope.model.sounder_format_dictionary.common_dictionary as common
import pyat.utils.pyat_logger as log
from pyat.sensor.nmea import GPSQualityIndicator
from pyat.sonarscope.bs_correction import kongsberg_correction
from pyat.sonarscope.common import xarray_utils as ut
from pyat.sonarscope.common.xsf_utils import get_detection_antenna_coords
from pyat.sonarscope.model.constants import DEFAULT_BEAM_GROUP_IDENT, DefaultGroups, VariableDim
from pyat.sonarscope.model.constants import VariableKeys as k

logger = log.logging.getLogger(__file__)


class DetectionRange(common.VariableInterface):
    """Return detection range in number of samples"""

    def get_dimensions(self, nc_dataset):
        return ut.get_dimensions(
            nc_dataset[sg.BathymetryGrp.DETECTION_TWO_WAY_TRAVEL_TIME(ident=DEFAULT_BEAM_GROUP_IDENT)]
        )

    def get_values(self, nc_dataset):
        detection_two_way_travel_time = nc_dataset[
            sg.BathymetryGrp.DETECTION_TWO_WAY_TRAVEL_TIME(ident=DEFAULT_BEAM_GROUP_IDENT)
        ][:]
        sampling_freq = nc_dataset[
            sg.BeamGroup1VendorSpecificGrp.SEABED_IMAGE_SAMPLE_RATE(ident=DEFAULT_BEAM_GROUP_IDENT)
        ][:]

        # compute indices to transform ping_antenna variable to ping_detection variable
        detection_antenna_coords = get_detection_antenna_coords(nc_dataset)

        detection_sampling_freq = sampling_freq[detection_antenna_coords]
        detection_range = detection_two_way_travel_time[:] * detection_sampling_freq[:]

        return detection_range

    def get_attributes(self, nc_dataset):
        return {"long_name": "Detection range in number of samples", "units": "sample"}

    def get_fill_value(self, nc_dataset):
        return np.nan


class DetectionRangeToNormalIncidence(common.VariableInterface):
    """Return detection range in number of samples"""

    def get_dimensions(self, nc_dataset):
        return ut.get_dimensions(
            nc_dataset[
                sg.BathymetryVendorSpecificGrp.BACKSCATTER_NORMAL_INCIDENCE_LEVEL(ident=DEFAULT_BEAM_GROUP_IDENT)
            ]
        )

    def get_values(self, nc_dataset):
        detection_range = DetectionRange().get_values(nc_dataset)

        # Estimate range to normal incidence
        rx_antenna_index = nc_dataset[sg.BathymetryGrp.DETECTION_RX_TRANSDUCER_INDEX(ident=DEFAULT_BEAM_GROUP_IDENT)][:]
        # keep antenna index in valid range
        rx_antenna_index[rx_antenna_index < 0] = 0
        rxAntennaCount = self.get_dimensions(nc_dataset)[VariableDim.RX_ANTENNA_DIM].size
        pingCount = self.get_dimensions(nc_dataset)[VariableDim.PING_DIM].size

        range_to_normal_incidence = np.full(shape=(pingCount, rxAntennaCount), fill_value=np.nan)
        for antenna_index in np.arange(rxAntennaCount):
            range_to_normal_incidence[:, antenna_index] = np.nanmin(
                detection_range[:].data, axis=1, initial=np.inf, where=rx_antenna_index == antenna_index
            )
        return range_to_normal_incidence

    def get_attributes(self, nc_dataset):
        return {"long_name": "Detection range at normal incidence angle in number of samples", "units": "sample"}

    def get_fill_value(self, nc_dataset):
        return np.nan


class TvgLawCrossOverAngle(common.VariableInterface):
    """Return tvg law crossover angle in degrees"""

    def get_dimensions(self, nc_dataset):
        return ut.get_dimensions(nc_dataset[sg.BathymetryGrp.DETECTION_BACKSCATTER_R(ident=DEFAULT_BEAM_GROUP_IDENT)])

    def get_values(self, nc_dataset):
        runtime_time = nc_dataset[sg.RuntimeGrp.TIME()]
        ping_time = nc_dataset[sg.BeamGroup1Grp.PING_TIME(ident=DEFAULT_BEAM_GROUP_IDENT)]
        crossover_angle_runtime = nc_dataset[sg.RuntimeGrp.TVG_LAW_CROSSOVER_ANGLE()]
        if crossover_angle_runtime[:].shape[0] > 0:
            interpolator = interp1d(
                runtime_time[:],
                crossover_angle_runtime[:],
                kind="previous",
                bounds_error=False,
                fill_value=(crossover_angle_runtime[0], crossover_angle_runtime[-1]),
            )
            ping_time_data = ping_time[:].data
            # interpolator does not handle masked array
            crossover_angle_pingtime = interpolator(ping_time_data)
        else:
            crossover_angle_pingtime = np.full_like(ping_time, fill_value=0.0)

        detection_count = self.get_dimensions(nc_dataset)[VariableDim.DETECTION_DIM].size
        crossover_angle = np.repeat(crossover_angle_pingtime[:, None], detection_count, axis=1)
        return crossover_angle

    def get_attributes(self, nc_dataset):
        return {"long_name": "Tvg law cross over angle", "units": "degree"}

    def get_fill_value(self, nc_dataset):
        return np.nan


class DetectionBackscatterValues(common.VariableInterface):
    """Return snippet mean values by beam without compensations"""

    def __init__(self, use_snippets: bool = False, uncomp_specular: bool = False, uncomp_lambert: bool = False):
        """Initialize variable with optinanal compensations
        @param use_snippets : True to recompute backscatter from snippets
        @param uncomp_specular : True to remove specular compensation
        @param uncomp_lambert : True to remove Lamberts law compensation
        """
        self.use_snippets = use_snippets
        self.uncomp_specular = uncomp_specular
        self.uncomp_lambert = uncomp_lambert

    def get_dimensions(self, nc_dataset):
        return ut.get_dimensions(nc_dataset[sg.BathymetryGrp.DETECTION_BACKSCATTER_R(ident=DEFAULT_BEAM_GROUP_IDENT)])

    def get_values(self, nc_dataset):
        if self.use_snippets:
            values = common.DetectionBackscatterSnippetMeanValues().get_values(nc_dataset)
        else:
            values = nc_dataset[sg.BathymetryGrp.DETECTION_BACKSCATTER_R(ident=DEFAULT_BEAM_GROUP_IDENT)][:]
            # Kmall has always compensated detection_backscatter_r for now.
            # TODO : use lambertsLawApplied variable to check if compensation is really applied

        if values is not None:
            detection_range = DetectionRange().get_values(nc_dataset)
            detection_antenna_coords = get_detection_antenna_coords(nc_dataset)
            # Estimate range to normal incidence
            range_to_normal_incidence = DetectionRangeToNormalIncidence().get_values(nc_dataset)[
                detection_antenna_coords
            ]

            if self.uncomp_specular:
                # Simrad_correctionSpeculaire.m
                BSN = nc_dataset[
                    sg.BathymetryVendorSpecificGrp.BACKSCATTER_NORMAL_INCIDENCE_LEVEL(ident=DEFAULT_BEAM_GROUP_IDENT)
                ][:][detection_antenna_coords]
                BSO = nc_dataset[
                    sg.BathymetryVendorSpecificGrp.BACKSCATTER_OBLIQUE_INCIDENCE_LEVEL(ident=DEFAULT_BEAM_GROUP_IDENT)
                ][:][detection_antenna_coords]

                tvg_law_crossover_angle = TvgLawCrossOverAngle().get_values(nc_dataset)[:]

                specular_corr = kongsberg_correction.specular_correction(
                    detection_range=detection_range,
                    range_to_normal_incidence=range_to_normal_incidence,
                    backscatter_normal_incidence_level=BSN,
                    backscatter_oblique_incidence_level=BSO,
                    tvg_law_crossover_angle=tvg_law_crossover_angle,
                )
                values = values - specular_corr

            if self.uncomp_lambert:
                # Sonar_Lambert_KM.m
                # remove Sonar_Lambert_KM
                lambert_corr = kongsberg_correction.lambert_correction(
                    detection_range=detection_range, range_to_normal_incidence=range_to_normal_incidence
                )
                values = values - lambert_corr
        return values

    def get_attributes(self, nc_dataset):
        return {"long_name": "Backscatter snippets mean value without specular", "units": "dB"}

    def get_fill_value(self, nc_dataset):
        return np.nan


class RunTimeVariables(common.VariablesContainer, common.VariablesDictionary.RunTimeVariables):
    def __init__(self):
        """Build a key value dictionary referencing all variables from a given beam group of a xsf"""
        super().__init__()
        instance = sg.RuntimeGrp()
        self.variables[k.RUNTIME_TIME] = common.createXsfVariable(
            variable_name=sg.RuntimeGrp.TIME_VNAME, instance=instance, group=sg.RuntimeGrp
        )
        self.variables[k.TVG_LAW_CROSSOVER_ANGLE] = common.createXsfVariable(
            variable_name=sg.RuntimeGrp.TVG_LAW_CROSSOVER_ANGLE_VNAME, instance=instance, group=sg.RuntimeGrp
        )


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
            variable_name=sg.PositionSubGroupVendorSpecificGrp.SENSOR_QUALITY_INDICATOR_VNAME,
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
        self.variables[k.TRANSMIT_TYPE] = common.createXsfVariable(
            variable_name=sg.BeamGroup1Grp.TRANSMIT_TYPE_VNAME, instance=instance, group=group, ident=beam_group
        )
        # self.variables[k.SAMPLE_COUNT] = common.createXsfVariable(
        #     variable_name=sg.BeamGroup1Grp.SAMPLE_COUNT_VNAME, instance=instance, group=group, ident=beam_group
        # )

        instance = sg.BeamGroup1VendorSpecificGrp()
        group = sg.BeamGroup1VendorSpecificGrp

        self.variables[k.TX_SECTOR_COUNT] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.TX_SECTOR_COUNT_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.FREQUENCY_MODE] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.FREQUENCY_MODE_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.SWATH_PER_PING] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.SWATH_PER_PING_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.PING_MODE] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.DEPTH_MODE_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )

        # PULSE_LENGTH_MODE doesn't exist in kmall
        self.variables[k.PULSE_FORM] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.PULSE_FORM_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.TX_BEAMWIDTH] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.TRANSMIT_ARRAY_SIZE_USED_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.RX_BEAMWIDTH] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.RECEIVE_ARRAY_SIZE_USED_VNAME,
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
        self.variables[k.MEAN_ABS_COEFF] = common.createXsfVariable(
            variable_name=sg.BathymetryGrp.DETECTION_MEAN_ABSORPTION_COEFFICIENT_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )

        instance = sg.BathymetryVendorSpecificGrp()
        group = sg.BathymetryVendorSpecificGrp
        self.variables[k.BACKSCATTER_OBLIQUE_INCIDENCE_LEVEL] = common.createXsfVariable(
            variable_name=sg.BathymetryVendorSpecificGrp.BACKSCATTER_OBLIQUE_INCIDENCE_LEVEL_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.BACKSCATTER_NORMAL_INCIDENCE_LEVEL] = common.createXsfVariable(
            variable_name=sg.BathymetryVendorSpecificGrp.BACKSCATTER_NORMAL_INCIDENCE_LEVEL_VNAME,
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

        instance = sg.BeamGroup1VendorSpecificGrp()
        group = sg.BeamGroup1VendorSpecificGrp
        self.variables[k.PULSE_LENGTH_EFFECTIVE] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.EFFECTIVESIGNALLENGTH_SEC_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.PULSE_LENGTH_REALTIME] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.EFFECTIVESIGNALLENGTH_SEC_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.DETECTION_SAMPLING_FREQ] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.SEABED_IMAGE_SAMPLE_RATE_VNAME,
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


class ComputedPingVariables(common.VariablesContainer, common.VariablesDictionary.ComputedPingVariables):
    def __init__(self):
        super().__init__()
        self.variables[k.INTERPINGS_DISTANCE] = common.InterpingVariable()
        self.variables[k.WC_PRESENCE] = common.WCPresence()


class ComputedPingDetectionVariables(
    common.VariablesContainer, common.VariablesDictionary.ComputedPingDetectionVariables
):
    def __init__(self):
        super().__init__()

        self.variables[k.DETECTION_BEAM_POINTING_ANGLE_REF_VERTICAL] = common.DetectionPointingAngleVertical()
        self.variables[k.DETECTION_BEAM_POINTING_ANGLE_REF_PLATFORM] = common.DetectionPointingAnglePlatform()

        self.variables[k.DETECTION_INCIDENCE_ANGLE] = common.DetectionIncidenceAngle()
        self.variables[k.DETECTION_RANGE_SAMPLE] = DetectionRange()
        self.variables[k.RANGE_TO_NORMAL_INCIDENCE] = DetectionRangeToNormalIncidence()
        self.variables[k.DETECTION_BACKSCATTER_WITHOUT_COMP] = DetectionBackscatterValues(
            use_snippets=False, uncomp_specular=True, uncomp_lambert=True
        )
        self.variables[k.DETECTION_BS_SNIPPETS_MEAN] = common.DetectionBackscatterSnippetMeanValues()
        self.variables[k.DETECTION_BS_SNIPPETS_MEAN_WITHOUT_SPECULAR_COMP] = DetectionBackscatterValues(
            use_snippets=True, uncomp_specular=True, uncomp_lambert=False
        )
        self.variables[k.DETECTION_BS_SNIPPETS_MEAN_WITHOUT_LAMBERT_COMP] = DetectionBackscatterValues(
            use_snippets=True, uncomp_specular=True, uncomp_lambert=True
        )
        self.variables[k.TVG_LAW_CROSSOVER_ANGLE] = TvgLawCrossOverAngle()
