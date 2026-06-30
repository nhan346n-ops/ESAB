"""A dictionary of known variables for all files"""

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
from pyat.sonarscope.model.constants import DEFAULT_BEAM_GROUP_IDENT, DefaultGroups
from pyat.sonarscope.model.constants import VariableKeys as k
from pyat.utils.netcdf import get_default_fillvalue
from pyat.utils.netcdf_utils import get_variable


class RuntimeTime(common.VariableInterface):
    """In case of .all data runtime time is retrieved from ping sequence number (raw count)"""

    def get_dimensions(self, nc_dataset: nc.Dataset):
        # we have the same dimension as runtime_time
        time = nc_dataset[sg.RuntimeGrp.TIME()]
        return ut.get_dimensions(time)

    def get_values(self, nc_dataset: nc.Dataset):
        """read variable values"""
        # we search for matching runtime_raw_count and ping_raw_count
        runtime_time = get_variable(nc_dataset, sg.RuntimeGrp.TIME())
        runtime_values = runtime_time[:]
        runtime_raw_count = get_variable(nc_dataset, sg.RuntimeGrp.RUNTIME_RAW_COUNT())
        ping_raw_count = get_variable(
            nc_dataset, sg.BeamGroup1VendorSpecificGrp.PING_RAW_COUNT(ident=DEFAULT_BEAM_GROUP_IDENT)
        )
        ping_time = get_variable(nc_dataset, sg.BeamGroup1Grp.PING_TIME(ident=DEFAULT_BEAM_GROUP_IDENT))

        if runtime_raw_count and ping_raw_count and ping_time is not None:
            # ping_raw_count_unwrap is sorted in ascending order
            ping_indices = np.searchsorted(ping_time, runtime_values)
            # force last runtime_time value to be associated to last ping_time value to avoid extrapolation issues after last ping
            ping_indices[ping_indices == len(ping_time)] = len(ping_time)
            # for each ping index, we search for matching runtime_raw_count in the neighboorhood of ping_time
            for i, raw_count in enumerate(runtime_raw_count):
                lower_index = max(0, ping_indices[i] - 5)
                upper_index = min(len(ping_time), ping_indices[i] + 5)
                # compute offsets as uint16 then cast to int16 to handle wrap around (raw count are uint16 that can wrap around from 65535 to 0)
                ping_count_offsets = (
                    ping_raw_count[lower_index:upper_index].astype(np.uint16) - np.uint16(raw_count)
                ).astype(np.int16)
                # find first ping with raw count offset >= 0 (ie raw count is smaller than ping raw count, so we are just before the ping)
                local_index = np.searchsorted(ping_count_offsets, 0)
                # if no ping with raw count offset >= 0 is found, we keep runtime value (can happen if runtime value is after last ping)
                if local_index == len(ping_count_offsets):
                    continue
                runtime_values[i] = ping_time[lower_index + local_index]
        return runtime_values

    def get_attributes(self, nc_dataset: nc.Dataset):
        return {"long_name": "Runtime time"}

    def get_fill_value(self, nc_dataset: nc.Dataset):
        return 0


class IndicativeSwathPerPing(common.VariableInterface):
    """In case of .all data an indicative number of ping per swath is computed"""

    def get_dimensions(self, nc_dataset: nc.Dataset):
        # we have the same dimension as ping_time
        ping_time = nc_dataset[sg.BeamGroup1Grp.PING_TIME(ident=DEFAULT_BEAM_GROUP_IDENT)]
        return ut.get_dimensions(ping_time)

    def get_values(self, nc_dataset: nc.Dataset):
        """read variable values"""
        # we have got a runtime dual swath mode variable
        dual_swath_mode = nc_dataset[sg.RuntimeGrp.DUAL_SWATH_MODE()]
        runtime_time = RuntimeTime().get_values(nc_dataset)
        # ge fully monotonic runtime_time and keep only last runtime change
        runtime_mask = np.append((np.diff(runtime_time) > 0), True)
        ping_time = nc_dataset[sg.BeamGroup1Grp.PING_TIME(ident=DEFAULT_BEAM_GROUP_IDENT)]
        fill_value = get_default_fillvalue(dual_swath_mode.dtype)
        interpolator = interp1d(
            runtime_time[runtime_mask],
            dual_swath_mode[runtime_mask],
            kind="previous",
            bounds_error=False,
            fill_value=(fill_value, dual_swath_mode[-1]),  # extrapolate after last value
        )
        ping_time_data = ping_time[:]
        # interpolator does not handle masked array
        if np.ma.isMaskedArray(ping_time_data):
            ping_time_data = np.array(ping_time_data)
        values = interpolator(ping_time_data)

        values[values > 0] = 2  # if mode is set to 1 or 2 we assume dual swath
        values[values == 0] = 1  # if mode is 0 we assume single swath
        values[values < 0] = 0  # if value is negative, we set to zero
        return values

    def get_attributes(self, nc_dataset: nc.Dataset):
        return {"long_name": "Indicative number of swaths per ping"}

    def get_fill_value(self, nc_dataset: nc.Dataset):
        return 0


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
        return {"long_name": "Number of samples to detection", "units": "samples"}

    def get_fill_value(self, nc_dataset):
        return np.nan


class DetectionBackscatterValues(common.VariableInterface):
    """Return snippet mean values by beam without compensation"""

    def __init__(self, use_snippets: bool = False, uncomp_specular: bool = False, uncomp_lambert: bool = False):
        """Initialize variable with optional compensations
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
            # all has always uncompensated detection_backscatter_r for now.
            # TODO : use detection_backscatter_compensation variable to check if compensation is really applied
        if self.use_snippets and (values is not None):
            detection_range = DetectionRange().get_values(nc_dataset)
            detection_antenna_coords = get_detection_antenna_coords(nc_dataset)
            Rn = nc_dataset[sg.BathymetryVendorSpecificGrp.RANGE_TO_NORMAL_INCIDENCE(ident=DEFAULT_BEAM_GROUP_IDENT)][
                :
            ][detection_antenna_coords]

            if self.uncomp_specular:
                # Simrad_correctionSpeculaire.m
                BSN = nc_dataset[
                    sg.BathymetryVendorSpecificGrp.BACKSCATTER_NORMAL_INCIDENCE_LEVEL(ident=DEFAULT_BEAM_GROUP_IDENT)
                ][:][detection_antenna_coords]
                BSO = nc_dataset[
                    sg.BathymetryVendorSpecificGrp.BACKSCATTER_OBLIQUE_INCIDENCE_LEVEL(ident=DEFAULT_BEAM_GROUP_IDENT)
                ][:][detection_antenna_coords]
                TVGCrossOver = nc_dataset[
                    sg.BathymetryVendorSpecificGrp.TVG_LAW_CROSS_OVER_ANGLE(ident=DEFAULT_BEAM_GROUP_IDENT)
                ][:][detection_antenna_coords]

                specular_corr = kongsberg_correction.specular_correction(
                    detection_range=detection_range,
                    range_to_normal_incidence=Rn,
                    backscatter_normal_incidence_level=BSN,
                    backscatter_oblique_incidence_level=BSO,
                    tvg_law_crossover_angle=TVGCrossOver,
                )

                values = values - specular_corr
            if self.uncomp_lambert:
                # Sonar_Lambert_KM.m
                # remove Sonar_Lambert_KM

                lambert_corr = kongsberg_correction.lambert_correction(
                    detection_range=detection_range, range_to_normal_incidence=Rn
                )
                values = values - lambert_corr

        return values

    def get_attributes(self, nc_dataset):
        return {"long_name": "Backscatter snippets mean value without specular", "units": "dB"}

    def get_fill_value(self, nc_dataset):
        return np.nan


class PulseLengthEffective(common.VariableInterface):
    """
    Return computed pulse length effective
    Code inspired from create_signalsFromRawRange.m (sonarscope)
    """

    def get_dimensions(self, nc_dataset):
        return ut.get_dimensions(nc_dataset[sg.BeamGroup1Grp.TRANSMIT_DURATION_NOMINAL(ident=DEFAULT_BEAM_GROUP_IDENT)])

    def get_values(self, nc_dataset):
        pulse_length_nominal = nc_dataset[sg.BeamGroup1Grp.TRANSMIT_DURATION_NOMINAL(ident=DEFAULT_BEAM_GROUP_IDENT)][:]
        transmit_type = nc_dataset[sg.BeamGroup1Grp.TRANSMIT_TYPE(ident=DEFAULT_BEAM_GROUP_IDENT)][:]

        pulse_length_effective = np.copy(pulse_length_nominal)

        # CW(0)/LFM(1)/HFM(2)
        cw_mask = transmit_type == 0

        # CW
        # pulse_length_nominal = 0.7 * total_pulse_length
        # pulse_length_effective = 0.375 * total_pulse_length
        # pulse_length_nominal = 1/BDW
        # FM
        # pulse_length_nominal = pulse_length_effective = 1/BDW

        pulse_length_effective[cw_mask] = (
            pulse_length_nominal[cw_mask] / 0.7  # total PL
        ) * 0.375  # Mail kjell.echholt.nilsen@km.kongsberg.com du 28/09/2017

        sounder_model_number = nc_dataset[sg.PlatformVendorSpecificGrp.get_group_path()].kongsbergModelNumber
        if sounder_model_number == 850:  # ME70
            pulse_length_effective = pulse_length_nominal * 0.7

        return pulse_length_effective

    def get_attributes(self, nc_dataset):
        return {"long_name": "Effective pulse length by sector", "units": "s"}

    def get_fill_value(self, nc_dataset):
        return np.nan


class PulseLengthRealtime(common.VariableInterface):
    """
    Return pulse length used for insonified area correction in realtime by the sounder
    Code inspired from create_signalsFromRawRange.m (sonarscope)
    """

    def get_dimensions(self, nc_dataset):
        return ut.get_dimensions(nc_dataset[sg.BeamGroup1Grp.TRANSMIT_DURATION_NOMINAL(ident=DEFAULT_BEAM_GROUP_IDENT)])

    def get_values(self, nc_dataset):
        pulse_length_effective = PulseLengthEffective().get_values(nc_dataset)
        pulse_length_nominal = nc_dataset[sg.BeamGroup1Grp.TRANSMIT_DURATION_NOMINAL(ident=DEFAULT_BEAM_GROUP_IDENT)][:]
        transmit_type = nc_dataset[sg.BeamGroup1Grp.TRANSMIT_TYPE(ident=DEFAULT_BEAM_GROUP_IDENT)][:]
        # CW(0)/LFM(1)/HFM(2)
        cw_mask = transmit_type == 0
        # check pulse length with runtime value in CW mode
        pulse_length_runtime = nc_dataset[sg.RuntimeGrp.TX_PULSE_LENGTH()]

        sounder_model_number = nc_dataset[sg.PlatformVendorSpecificGrp.get_group_path()].kongsbergModelNumber
        if np.any(cw_mask) and pulse_length_runtime[:].shape[0] > 0:
            runtime_time = RuntimeTime().get_values(nc_dataset)
            # ge fully monotonic runtime_time to keep only last runtime change
            runtime_mask = np.append((np.diff(runtime_time) > 0), True)

            ping_time = nc_dataset[sg.BeamGroup1Grp.PING_TIME(ident=DEFAULT_BEAM_GROUP_IDENT)]
            interpolator = interp1d(
                runtime_time[runtime_mask],
                pulse_length_runtime[runtime_mask],
                kind="previous",
                bounds_error=False,
                fill_value=(pulse_length_runtime[0], pulse_length_runtime[-1]),
            )
            ping_time_data = ping_time[:].data
            # interpolator does not handle masked array
            pulse_length_pingtime = interpolator(ping_time_data)

            # compare pulse_length_pingtime with pulse_length_effective
            # check first CW values are close
            pulse_length_runtime = np.repeat(pulse_length_pingtime[:, None], cw_mask.shape[1], axis=1)
            pl_effective_cw = pulse_length_effective[cw_mask]
            pl_runtime_cw = pulse_length_runtime[cw_mask]
            pl_nominal_cw = pulse_length_nominal[cw_mask]
            if np.any(np.isclose(pl_runtime_cw, pl_effective_cw, rtol=0.01)):
                # First check effective beacause runtime value can match effective from a sector and nominal from another
                log.info("Pulse length from runtime datagram fits effective pulse length.")
            elif np.any(np.isclose(pl_runtime_cw, pl_nominal_cw, rtol=0.01)):
                log.info("Pulse length from runtime datagram fits nominal pulse length.")
            else:
                log.warning("Runtime pulse length doesn't fit any sector")
            log.info(
                f"First Runtime value = {pl_runtime_cw[0]}. Effective value = {pl_effective_cw[0]}. Nominal value = {pl_nominal_cw[0]}."
            )
            if sounder_model_number in [122, 302, 710]:  # EM122
                pulse_length_effective[cw_mask] = pulse_length_nominal[cw_mask]
                log.info(
                    f"Pulse length is always nominal for EM{sounder_model_number}. Using nominal pulse length for Kongsberg Insonified Area correction."
                )
            else:
                log.info("Using Runtime value for Kongsberg Insonified Area correction.")
                pulse_length_effective[cw_mask] = pulse_length_runtime[cw_mask]

        return pulse_length_effective

    def get_attributes(self, nc_dataset):
        return {"long_name": "Effective pulse length by sector (realtime)", "units": "s"}

    def get_fill_value(self, nc_dataset):
        return np.nan


class RunTimeVariables(common.VariablesContainer, common.VariablesDictionary.RunTimeVariables):
    def __init__(self):
        """Build a key value dictionary referencing all variables from a given beam group of a xsf"""
        super().__init__()
        instance = sg.RuntimeGrp()
        self.variables[k.RUNTIME_TIME] = RuntimeTime()
        self.variables[k.FREQUENCY_MODE] = common.createXsfVariable(
            variable_name=sg.RuntimeGrp.FREQUENCY_MODE_VNAME, instance=instance, group=sg.RuntimeGrp
        )
        self.variables[k.PING_MODE] = common.createXsfVariable(
            variable_name=sg.RuntimeGrp.PING_MODE_VNAME, instance=instance, group=sg.RuntimeGrp
        )
        self.variables[k.PULSE_FORM] = common.createXsfVariable(
            variable_name=sg.RuntimeGrp.TX_PULSE_FORM_VNAME, instance=instance, group=sg.RuntimeGrp
        )
        self.variables[k.PULSE_LENGTH_MODE] = common.createXsfVariable(
            variable_name=sg.RuntimeGrp.PULSE_LENGTH_MODE_VNAME, instance=instance, group=sg.RuntimeGrp
        )
        self.variables[k.PULSE_LENGTH_EFFECTIVE] = common.createXsfVariable(
            variable_name=sg.RuntimeGrp.TX_PULSE_LENGTH_VNAME, instance=instance, group=sg.RuntimeGrp
        )
        self.variables[k.DUAL_SWATH_MODE] = common.createXsfVariable(
            variable_name=sg.RuntimeGrp.DUAL_SWATH_MODE_VNAME, instance=instance, group=sg.RuntimeGrp
        )
        self.variables[k.RX_BEAMWIDTH] = common.createXsfVariable(
            variable_name=sg.RuntimeGrp.RECEIVER_BEAMWIDTH_VNAME, instance=instance, group=sg.RuntimeGrp
        )
        self.variables[k.TX_BEAMWIDTH] = common.createXsfVariable(
            variable_name=sg.RuntimeGrp.TX_BEAMWIDTH_VNAME, instance=instance, group=sg.RuntimeGrp
        )
        self.variables[k.RECEIVER_FIXED_GAIN] = common.createXsfVariable(
            variable_name=sg.RuntimeGrp.RECEIVER_FIXED_GAIN_VNAME, instance=instance, group=sg.RuntimeGrp
        )
        self.variables[k.TX_POWER] = common.createXsfVariable(
            variable_name=sg.RuntimeGrp.TX_POWER_RE_MAXIMUM_VNAME, instance=instance, group=sg.RuntimeGrp
        )


class PositionVariables(common.VariablesContainer, common.VariablesDictionary.PositionVariables):
    def __init__(self, root_dataset: nc.Dataset):
        super().__init__()

        # First check for preferred position subgroup
        preferred_position = root_dataset[DefaultGroups.BEAM_GROUP_NAME].preferred_position
        # now retrieve the name of the sensor
        sensor_ids = root_dataset[sg.PlatformGrp.get_group_path()].variables[sg.PlatformGrp.POSITION_IDS_VNAME]
        sensor_id = sensor_ids[preferred_position]
        instance = sg.PositionSubGroupVendorSpecificGrp()
        # check sensor id
        # sensor id are not always well set, we use default value if an error is in file
        # use netcdf api to ensure that group really exist
        if sensor_id not in root_dataset[sg.PositionGrp.get_group_path()].groups:
            log.warning(
                f"Position sensor id {sensor_id} is not found in subgroups of {sg.PositionGrp.get_group_path()} \
                              ({root_dataset[sg.PositionGrp.get_group_path()].groups})"
            )
            # we use the first group found as default sensor_id
            sensor_id = next(iter(root_dataset[sg.PositionGrp.get_group_path()].groups))
            log.warning(f"Use {sensor_id} as default sensor_id")

        self.variables[k.POSITION_SENSOR_QUALITY_INDICATOR] = common.createXsfVariable(
            variable_name=sg.PositionSubGroupVendorSpecificGrp.SENSOR_QUALITY_INDICATOR_VNAME,
            instance=instance,
            group=sg.PositionSubGroupVendorSpecificGrp,
            ident=sensor_id,
            fill_value=GPSQualityIndicator.NOT_AVAILABLE.value,
        )
        self.variables[k.POSITION_SENSOR_TIME] = common.createXsfVariable(
            variable_name=sg.PositionSubGroup.TIME_VNAME,
            instance=sg.PositionSubGroup(),
            group=sg.PositionSubGroup,
            ident=sensor_id,
        )


class PingTimeVariables(common.VariablesContainer, common.VariablesDictionary.PingTimeVariables):
    def __init__(self, beam_group: str = DEFAULT_BEAM_GROUP_IDENT):
        """Build a key value dictionary referencing all variables from a given beam group of a xsf"""
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
        self.variables[k.MEAN_ABS_COEFF] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.MEAN_ABS_COEFF_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.TX_SECTOR_COUNT] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.TX_SECTOR_COUNT_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.CENTER_FREQUENCY] = common.createXsfVariable(
            variable_name=sg.BeamGroup1VendorSpecificGrp.CENTER_FREQUENCY_VNAME,
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
        self.variables[k.RANGE_TO_NORMAL_INCIDENCE] = common.createXsfVariable(
            variable_name=sg.BathymetryVendorSpecificGrp.RANGE_TO_NORMAL_INCIDENCE_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.TVG_LAW_CROSSOVER_ANGLE] = common.createXsfVariable(
            variable_name=sg.BathymetryVendorSpecificGrp.TVG_LAW_CROSS_OVER_ANGLE_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )
        self.variables[k.DETECTION_SAMPLING_FREQ] = common.createXsfVariable(
            variable_name=sg.BathymetryVendorSpecificGrp.DETECTION_SAMPLING_FREQ_VNAME,
            instance=instance,
            group=group,
            ident=beam_group,
        )


class ComputedPingVariables(common.VariablesContainer, common.VariablesDictionary.ComputedPingVariables):
    def __init__(self):
        super().__init__()
        self.variables[k.INTERPINGS_DISTANCE] = common.InterpingVariable()
        self.variables[k.SWATH_PER_PING] = IndicativeSwathPerPing()
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
        self.variables[k.PULSE_LENGTH_EFFECTIVE] = PulseLengthEffective()
        self.variables[k.PULSE_LENGTH_REALTIME] = PulseLengthRealtime()

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
