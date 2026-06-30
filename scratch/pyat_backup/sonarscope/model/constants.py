import sonar_netcdf.sonar_groups as sg

DEFAULT_BEAM_GROUP_IDENT = "Beam_group1"
SONAR_GROUP_NAME = sg.SonarGrp.get_group_path()
BEAM_GROUP_NAME = sg.BeamGroup1Grp.get_group_path(ident=DEFAULT_BEAM_GROUP_IDENT)
BATHY_GROUP_NAME = sg.BathymetryGrp.get_group_path(ident=DEFAULT_BEAM_GROUP_IDENT)
BATHY_VENDOR_GROUP_NAME = sg.BathymetryVendorSpecificGrp.get_group_path(ident=DEFAULT_BEAM_GROUP_IDENT)
BEAM_VENDOR_GROUP_NAME = sg.BeamGroup1VendorSpecificGrp.get_group_path(ident=DEFAULT_BEAM_GROUP_IDENT)
PLATFORM_VENDOR_GROUP_NAME = sg.PlatformVendorSpecificGrp.get_group_path()


class Naming:
    DETECTION_PREFIX = "detection_"


# List of keys for known variables as string values
class DefaultGroups:
    DEFAULT_BEAM_GROUP_IDENT = "Beam_group1"
    SONAR_GROUP_NAME = sg.SonarGrp.get_group_path()
    BEAM_GROUP_NAME = sg.BeamGroup1Grp.get_group_path(ident=DEFAULT_BEAM_GROUP_IDENT)
    BATHY_GROUP_NAME = sg.BathymetryGrp.get_group_path(ident=DEFAULT_BEAM_GROUP_IDENT)
    BATHY_VENDOR_GROUP_NAME = sg.BathymetryVendorSpecificGrp.get_group_path(ident=DEFAULT_BEAM_GROUP_IDENT)
    BEAM_VENDOR_GROUP_NAME = sg.BeamGroup1VendorSpecificGrp.get_group_path(ident=DEFAULT_BEAM_GROUP_IDENT)


class VariableDim:
    PING_DIM = "ping_time"
    DETECTION_DIM = "detection"
    RX_ANTENNA_DIM = "rxAntenna"
    TX_BEAM_DIM = "tx_beam"


class VariableKeys:
    INTERPINGS_DISTANCE = "interping_distance"
    WC_PRESENCE = "wc presence"

    SWATH_PER_PING = "swath_per_ping"
    MULTIPING_SEQUENCE = "multiping_sequence"

    PING_TIME = "ping_time"
    PLATFORM_ROLL = "platform_roll"
    PLATFORM_PITCH = "platform_pitch"
    PLATFORM_HEADING = "platform_heading"
    PLATFORM_LONGITUDE = "platform_longitude"
    PLATFORM_LATITUDE = "platform_latitude"
    SOUND_SPEED_AT_TRANSDUCER = "sound_speed_at_transducer"
    DETECTION_Z = "detection_z"
    DETECTION_LONGITUDE = "detection_longitude"
    DETECTION_LATITUDE = "detection_latitude"
    DETECTION_QUALITY_FACTOR = "detection_quality_factor"  # can be missing
    BATHYMETRY_STATUS = "status"
    #  see ref l250 : this is a computed variable if not defined in dataset
    MEAN_ABS_COEFF = "mean_abs_coeff"
    RANGE_TO_NORMAL_INCIDENCE = "range_to_normal_incidence"
    BACKSCATTER_NORMAL_INCIDENCE_LEVEL = "backscatter_normal_incidence_level"
    BACKSCATTER_OBLIQUE_INCIDENCE_LEVEL = "backscatter_oblique_incidence_level"
    TVG_LAW_CROSSOVER_ANGLE = "tvg_law_crossover_angle"

    TRANSMIT_TYPE = "transmit_type"
    SAMPLE_COUNT = "sample_count"

    FREQUENCY_MODE = "frequency_mode"
    PING_MODE = "ping_mode"  # DEPTH MODE in kmall
    PULSE_LENGTH_MODE = "pulse_length_mode"  # DEPTH MODE in kmall
    PULSE_LENGTH_EFFECTIVE = "pulse_length_effective"
    PULSE_LENGTH_REALTIME = "pulse_length_realtime"  # pulse used for IA correction in Real time
    PULSE_FORM = "pulse_form"
    RX_BEAMWIDTH = "rx_beamwidth"
    TX_BEAMWIDTH = "tx_beamwidth"

    ### Kongsberg specific
    RECEIVER_FIXED_GAIN = "receiver_fixed_gain"
    TX_POWER = "tx_power"
    CENTER_FREQUENCY = "center_frequency"

    ### Reson specific
    GAIN_SELECTION = "gain_selection"
    RANGE_SELECTION = "range_selection"
    POWER_SELECTION = "power_selection"
    ###

    DUAL_SWATH_MODE = "dual_swath_mode"
    RUNTIME_TIME = "runtime_time"
    TX_SECTOR_COUNT = "tx_sector_count"

    POSITION_SENSOR_QUALITY_INDICATOR = "position_sensor_quality_indicator"
    POSITION_SENSOR_TIME = "position_sensor_time"

    DETECTION_BACKSCATTER = "detection_backscatter_r"
    DETECTION_BACKSCATTER_WITHOUT_COMP = "detection_backscatter_without_comp"

    DETECTION_BACKSCATTER_CALIBRATION = "detection_backscatter_calibration"

    DETECTION_BEAM_POINTING_ANGLE = "detection_beam_pointing_angle"
    DETECTION_BEAM_POINTING_ANGLE_REF_VERTICAL = "detection_beam_pointing_angle_ref_vertical"
    DETECTION_BEAM_POINTING_ANGLE_REF_PLATFORM = "detection_beam_pointing_angle_ref_platform"
    TX_TILT_ANGLE_REF_VERTICAL = "tx_tilt_angle_ref_vertical"

    DETECTION_INCIDENCE_ANGLE = "detection_incidence_angle_ref_vertical"
    DETECTION_TX_BEAM_INDEX = "detection_tx_beam"
    DETECTION_RX_TRANSDUCER_INDEX = "detection_rx_transducer_index"
    DETECTION_RANGE_SAMPLE = "detection_range_sample"
    DETECTION_SAMPLING_FREQ = "detection_sampling_freq"

    DETECTION_BS_SNIPPETS_MEAN = "detection_bs_snippets_mean"
    DETECTION_BS_SNIPPETS_MEAN_WITHOUT_SPECULAR_COMP = "detection_bs_snippets_mean_without_specular_comp"
    DETECTION_BS_SNIPPETS_MEAN_WITHOUT_LAMBERT_COMP = "detection_bs_snippets_mean_without_lambert_comp"
