#! /usr/bin/env python3
# coding: utf-8

import math
import re
from contextlib import contextmanager
from typing import Dict, Generator, Iterable, Optional, Tuple

import netCDF4 as nc
import numba
import numpy as np
from pyproj import crs

import pyat.utils.coordinates_system_utils as cs_util
import pyat.xsf.xsf_driver as xd
from pyat.mbg.mbg_sounder_desc import mbg_code_to_desc
from pyat.sounder import sounder_driver
from pyat.utils import numpy_utils
from pyat.utils.coords import compute_norms_and_radii
from pyat.utils.nc_encoding import open_nc_file

# properties
InstallParameters = "mbInstallParameters"
TX_ANTENNA = "mbTxAntennaLeverArm"
# Dimensions
ANTENNA_NBR = "mbAntennaNbr"
BEAM_NBR = "mbBeamNbr"
COMMENT_LENGTH = "mbCommentLength"
CYCLE_NBR = "mbCycleNbr"
HISTORY_REC_NBR = "mbHistoryRecNbr"
NAME_LENGTH = "mbNameLength"
VELOCITY_PROFIL_NBR = "mbVelocityProfilNbr"
# Layers
A_FLAG = "mbAFlag"
ABSCISSA = "mbAbscissa"
ABSORPTION_COEFFICIENT = "mbAbsorptionCoefficient"
ACROSS_BEAM_ANGLE = "mbAcrossBeamAngle"
ACROSS_DISTANCE = "mbAcrossDistance"
ACROSS_SLOPE = "mbAcrossSlope"
ALONG_DISTANCE = "mbAlongDistance"
ALONG_SLOPE = "mbAlongSlope"
ANTENNA = "mbAntenna"
AZIMUT_BEAM_ANGLE = "mbAzimutBeamAngle"
B_FLAG = "mbBFlag"
B_S_P_STATUS = "mbBSPStatus"
BEAM = "mbBeam"
BEAM_SPACING = "mbBeamSpacing"
C_FLAG = "mbCFlag"
C_QUALITY = "mbCQuality"
COMPENSATION_LAYER_MODE = "mbCompensationLayerMode"
CYCLE = "mbCycle"
DATE = "mbDate"
DEPTH = "mbDepth"
DISTANCE_SCALE = "mbDistanceScale"
DUROTONG_SPEED = "mbDurotongSpeed"
DYNAMIC_DRAUGHT = "mbDynamicDraught"
FILTER_IDENTIFIER = "mbFilterIdentifier"
FREQUENCY = "mbFrequency"
HEADING = "mbHeading"
HI_LO_ABSORPTION_RATIO = "mbHiLoAbsorptionRatio"
HIST_AUTOR = "mbHistAutor"
HIST_CODE = "mbHistCode"
HIST_COMMENT = "mbHistComment"
HIST_DATE = "mbHistDate"
HIST_MODULE = "mbHistModule"
HIST_TIME = "mbHistTime"
INTERLACING = "mbInterlacing"
MAX_PORT_COVERAGE = "mbMaxPortCoverage"
MAX_PORT_WIDTH = "mbMaxPortWidth"
MAX_STARBOARD_COVERAGE = "mbMaxStarboardCoverage"
MAX_STARBOARD_WIDTH = "mbMaxStarboardWidth"
OPERATOR_STATION_STATUS = "mbOperatorStationStatus"
ORDINATE = "mbOrdinate"
PARAM_MAXIMUM_DEPTH = "mbParamMaximumDepth"
PARAM_MINIMUM_DEPTH = "mbParamMinimumDepth"
PITCH = "mbPitch"
PROCESSING_UNIT_STATUS = "mbProcessingUnitStatus"
QUALITY = "mbQuality"
RANGE = "mbRange"
RECEIVE_BANDWIDTH = "mbReceiveBandwidth"
RECEIVE_BEAMWIDTH = "mbReceiveBeamwidth"
RECEIVER_FIXED_GAIN = "mbReceiverFixedGain"
RECEPTION_HEAVE = "mbReceptionHeave"
REFERENCE_DEPTH = "mbReferenceDepth"
REFLECTIVITY = "mbReflectivity"
ROLL = "mbRoll"
S_FLAG = "mbSFlag"
S_LENGTH_OF_DETECTION = "mbSLengthOfDetection"
S_QUALITY = "mbSQuality"
SAMPLING_RATE = "mbSamplingRate"
SONAR_FREQUENCY = "mbSonarFrequency"
SONAR_STATUS = "mbSonarStatus"
SOUND_VELOCITY = "mbSoundVelocity"
SOUNDER_MODE = "mbSounderMode"
SOUNDING_BIAS = "mbSoundingBias"
T_V_G_LAW_CROSSOVER_ANGLE = "mbTVGLawCrossoverAngle"
TIDE = "mbTide"
TIME = "mbTime"
TRANS_VELOCITY_SOURCE = "mbTransVelocitySource"
TRANSMISSION_HEAVE = "mbTransmissionHeave"
TRANSMIT_BEAMWIDTH = "mbTransmitBeamwidth"
TRANSMIT_POWER_RE_MAX = "mbTransmitPowerReMax"
TRANSMIT_PULSE_LENGTH = "mbTransmitPulseLength"
VEL_PROFIL_DATE = "mbVelProfilDate"
VEL_PROFIL_IDX = "mbVelProfilIdx"
VEL_PROFIL_REF = "mbVelProfilRef"
VEL_PROFIL_TIME = "mbVelProfilTime"
VERTICAL_DEPTH = "mbVerticalDepth"
YAW_PITCH_STAB_MODE = "mbYawPitchStabMode"

# Correction flags
AUTOMATIC_CLEANING = "mbAutomaticCleaning"
MANUAL_CLEANING = "mbManualCleaning"
POSITION_CORRECTION = "mbPositionCorrection"
VELOCITY_CORRECTION = "mbVelocityCorrection"
BIAS_CORRECTION = "mbBiasCorrection"
TIDE_CORRECTION = "mbTideCorrection"
DRAUGHT_CORRECTION = "mbSoundingCorrection"
IM_REFLECTIVITY_ORIGIN = "mbImReflectivityOrigin"

# A (antanna) flag values
A_FLAG_INVALID = -1
A_FLAG_MISSING = 0
A_FLAG_VALID = 1

# B (beam) flag values
B_FLAG_INVALID = -1
B_FLAG_MISSING = 0
B_FLAG_VALID = 2

# C (cycle=ping) flag values
C_FLAG_INVALID_ACQUIS = -3
C_FLAG_INVALID_AUTO = -2
C_FLAG_INVALID_OPERATOR = -1
C_FLAG_MISSING = 0
C_FLAG_VALID = 2
C_FLAG_INVALID_VALIDATED = 4
C_FLAG_MODIFIED = 5

# S (sounding) flag values
S_FLAG_INVALID_ACQUIS = -3
S_FLAG_INVALID_AUTO = -2
S_FLAG_INVALID_OPERATOR = -1
S_FLAG_MISSING = 0
S_FLAG_DOUBTFUL = 1
S_FLAG_VALID = 2


# pylint: disable=too-many-lines
class MbgDriver(sounder_driver.SounderDriver):
    @property
    def dataset(self) -> nc.Dataset:
        return self._dataset

    def __init__(self, file_path: str):
        super().__init__(file_path)

        self._dataset = None

        # Keep this layers in memory
        self._antennas: Optional[np.ndarray] = None
        self._fcs_depths: Optional[np.ndarray] = None
        self._scs_depths: Optional[np.ndarray] = None
        self._distance_scales: Optional[np.ndarray] = None
        self._reflectivities: Optional[np.ndarray] = None

    def open(self, mode: str = "r") -> nc.Dataset:
        """
        Open the file and return the resulting Dataset
        Implementation of SounderDriver abstract method
        """
        self._dataset = open_nc_file(self.sounder_file.file_path, mode=mode)

        self.sounder_file.south = self.dataset.mbSouthLatitude
        self.sounder_file.north = self.dataset.mbNorthLatitude
        self.sounder_file.west = self.dataset.mbWestLongitude
        self.sounder_file.east = self.dataset.mbEastLongitude
        self.sounder_file.swath_count = self.dataset.dimensions[CYCLE_NBR].size
        self.sounder_file.beam_count = self.dataset.dimensions[BEAM_NBR].size
        self.sounder_file.antenna_count = self.dataset.dimensions[ANTENNA_NBR].size

        return self.dataset

    def close(self) -> None:
        """
        Close the dataset if opened
        Implementation of SounderDriver abstract method
        """
        if self.dataset and self.dataset.isopen():
            self.dataset.close()
        self._dataset = None
        self._antennas = None
        self._distance_scales = None
        self._reflectivities = None

    def read_validity_flags(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of validity flags
        Implementation of SounderDriver abstract method
        """
        result = np.empty(shape=(to_swath - from_swath, self.sounder_file.beam_count), dtype=bool)

        antennas = self.read_antenna()
        c_flags = self.read_c_flag()
        b_flags = self.read_b_flag()
        a_flags = self.read_a_flag()
        s_flags = self.read_s_flag(from_swath, to_swath)

        MbgDriver.__compute_validity_flags(from_swath, to_swath, antennas, c_flags, b_flags, a_flags, s_flags, result)
        return result

    def translate_flags_to_xsf_status_and_details(
        self, from_swath: int, to_swath: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Translate MBG flags array to XSF status and status details
        """
        out_result_status = np.zeros(shape=(to_swath - from_swath, self.sounder_file.beam_count))
        out_result_details = np.zeros_like(out_result_status)
        antennas = self.read_antenna()
        c_flags = self.read_c_flag()
        b_flags = self.read_b_flag()
        a_flags = self.read_a_flag()
        s_flags = self.read_s_flag()
        MbgDriver.__flags_to_xsf_status_and_details(
            from_swath,
            to_swath,
            antennas,
            c_flags,
            b_flags,
            a_flags,
            s_flags,
            out_result_status,
            out_result_details,
        )

        return out_result_status, out_result_details

    def read_fcs_depths(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of depths. Shape is (to_swath - from_swath, beam_count)
        Depths are projected in Coordinates system transformations FCS (Fixed Coordinate System)
        For a MBG, this exactly the Depth layer
        Implementation of SounderDriver abstract method
        """
        if self._fcs_depths is None:
            self._fcs_depths = numpy_utils.to_memmap(self.read_depth())
        return self._fcs_depths[from_swath:to_swath]

    def read_scs_depths(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of depths. Shape is (to_swath - from_swath, beam_count)
        Depths are projected in Coordinates system transformations SCS (Surface Coordinate System)
        Implementation of SounderDriver abstract method
        """
        if self._scs_depths is None:
            antennas = self.read_antenna()
            vertical_offsets = self.__read_platform_vertical_offsets_by_antenna()
            self._scs_depths = numpy_utils.to_memmap(self.read_depth())
            tides = self.read_tide()
            draughts = self.read_dynamic_draught()
            MbgDriver.__adjust_depths(self._scs_depths, antennas, vertical_offsets, tides, draughts)
        return self._scs_depths[from_swath:to_swath]

    def read_reflectivities(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of Reflectivity values of all antennas
        Implementation of SounderDriver abstract method
        """
        if self._reflectivities is None:
            self._reflectivities = numpy_utils.to_memmap(self.read_reflectivity())
        return self._reflectivities[from_swath:to_swath]

    def read_across_distances(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of across distance. Shape is (to_swath - from_swath, beam_count)
        Implementation of SounderDriver abstract method
        """
        result = self.read_across_distance(from_swath, to_swath).astype(float)
        scales = self.__read_distance_scales(from_swath, to_swath)
        antennas = self.read_antenna()
        MbgDriver.__multiply_distances_by_scales(result, scales, antennas)
        return result

    def read_across_angles(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of across angles. Shape is (to_swath - from_swath, beam_count)
        Implementation of SounderDriver abstract method
        """
        return self.read_across_beam_angle(from_swath, to_swath)

    def iter_beam_positions(
        self, swath_count_by_iter: int, first_swath: int = 0, valid_only: bool = False
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Implementation of SounderDriver abstract method
        """
        return BeamPositionIterator(self, swath_count_by_iter, first_swath, valid_only=valid_only)

    def read_platform_longitudes(self) -> np.ndarray:
        """
        Implementation of SounderDriver abstract method
        """
        values = self.read_abscissa()
        if self.sounder_file.antenna_count > 1:
            # if we have two antennas, returns only 1st column (NB: values from 1st are duplicated into 2nd column)
            values = values[:, 0]
        return values.reshape(-1)  # enforce 1-dimensionality

    def read_platform_latitudes(self) -> np.ndarray:
        """
        Implementation of SounderDriver abstract method
        """
        values = self.read_ordinate()
        if self.sounder_file.antenna_count > 1:
            # if we have two antennas, returns only 1st column (NB: values from 1st are duplicated into 2nd column)
            values = values[:, 0]
        return values.reshape(-1)  # enforce 1-dimensionality

    def read_platform_headings(self) -> np.ndarray:
        """
        Implementation of SounderDriver abstract method
        """
        values = self.read_heading()
        if self.sounder_file.antenna_count > 1:
            # if we have two antennas, returns only 1st column (NB: values from 1st are duplicated into 2nd column)
            values = values[:, 0]
        return values.reshape(-1)  # enforce 1-dimensionality

    def read_ping_times(self) -> np.ndarray:
        """
        Implementation of SounderDriver abstract method
        """
        # Get ping time as timestamp[s]
        values = self.read_date_time()
        if self.sounder_file.antenna_count > 1:
            # if we have two antennas, returns only 1st column (NB: values from 1st are duplicated into 2nd column)
            values = values[:, 0]
        # Convert to milliseconds and then to datetime64 with millisecond precision and enforce 1-dimensionality
        return (values.reshape(-1) * 1000).astype("datetime64[ms]")

    def get_swath_indexes_from_time(self, start_time: np.datetime64, stop_time: np.datetime64) -> np.ndarray:
        """
        Get swath indexes corresponding to given start and stop time
        """
        ping_times = self.read_ping_times()

        return np.nonzero((ping_times >= start_time) & (ping_times <= stop_time))[0]

    def read_platform_vertical_offsets(self) -> np.ndarray:
        """
        Implementation of SounderDriver abstract method
        """
        values = self.__read_platform_vertical_offsets_by_antenna()
        if self.sounder_file.antenna_count > 1:
            # if we have two antennas, returns only 1st column (NB: values from 1st are duplicated into 2nd column)
            values = values[:, 0]
        return values.reshape(-1)  # enforce 1-dimensionality

    def read_sounder_desc(self) -> Tuple[str, int]:
        """Read sounder description and serial number"""
        code = self.dataset.getncattr("mbSounder")
        serialNumber = self.dataset.getncattr("mbSerialNumber")
        sounder_name = "Unknown"
        if code in mbg_code_to_desc:
            sounder_name = mbg_code_to_desc[code]
        return sounder_name, serialNumber

    def __read_platform_vertical_offsets_by_antenna(self) -> np.ndarray:
        """
        return the numpy array of computed read_platform vertical offsets.
        """
        antenna_vcs_coords = self.__read_tx_antenna_coordinates()
        pitchs = self.read_pitch()
        rolls = self.read_roll()
        transducter_depths = self.read_reference_depth()

        result = np.zeros(pitchs.shape, dtype=float)
        for i_swath in range(result.shape[0]):
            for antenna in range(result.shape[1]):
                antenna_scs_coords = cs_util.transform_vcs_to_scs(
                    np.deg2rad(pitchs[i_swath, antenna]), np.deg2rad(rolls[i_swath, antenna]), antenna_vcs_coords
                )
                result[i_swath, antenna] = antenna_scs_coords[2] - transducter_depths[i_swath, antenna]

        return result

    def __read_tx_antenna_coordinates(self) -> np.ndarray:

        if TX_ANTENNA in self.dataset.__dict__:
            return self.dataset.__dict__[TX_ANTENNA]

        result = np.zeros(3, dtype=np.float64)
        installParameters = self.__read_install_parameters()
        if all(key in installParameters for key in ["S1X", "S1Y", "S1Z"]):
            result[0] = installParameters["S1X"]
            result[1] = installParameters["S1Y"]
            result[2] = installParameters["S1Z"]
        return result

    def __read_install_parameters(self) -> Dict[str, float]:
        if InstallParameters in self.dataset.__dict__:
            install_params = self.dataset.__dict__[InstallParameters]
            split_regex = r"""
                (?P<key>[\w]+)=
                (?P<value>[-+]?(?:(?:\d*\.\d+)|(?: \d+ \.?))(?:[Ee][+-]?\d+)?)
                ($|,|;)
            """
            regex = re.compile(split_regex, re.VERBOSE)
            return {match.group("key"): float(match.group("value")) for match in regex.finditer(install_params)}

        return {}

    def read_along_distances(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of along distance. Shape is (to_swath - from_swath, beam_count)
        """
        result = self.read_along_distance(from_swath, to_swath).astype(float)
        scales = self.__read_distance_scales(from_swath, to_swath)
        antennas = self.read_antenna()
        MbgDriver.__multiply_distances_by_scales(result, scales, antennas)
        return result

    @staticmethod
    @numba.njit("void(float64[:,:], int8[:], float64[:,:], float64[:,:], float64[:,:])", cache=True, fastmath=True)
    def __adjust_depths(
        depths: np.ndarray,
        antennas: np.ndarray,
        vertical_offsets: np.ndarray,
        tides: np.ndarray,
        draughts: np.ndarray,
    ):
        for i_swath in range(depths.shape[0]):
            for i_beam in range(depths.shape[1]):
                antenna = max(antennas[i_beam], 0)
                depths[i_swath, i_beam] = (
                    depths[i_swath, i_beam]
                    + vertical_offsets[i_swath, antenna]
                    + tides[i_swath, antenna]  # Tides are positive in a MBG
                    + draughts[i_swath, antenna]  # Draughts are positive in a MBG
                )

    @staticmethod
    @numba.njit("void(float64[:,:], float64[:,:], int8[:])", cache=True, fastmath=True)
    def __multiply_distances_by_scales(
        distances: np.ndarray,
        scales: np.ndarray,
        antennas: np.ndarray,
    ):
        for i_swath in range(distances.shape[0]):
            for i_beam in range(distances.shape[1]):
                antenna = max(antennas[i_beam], 0)
                factor = scales[i_swath, antenna]
                distances[i_swath, i_beam] = distances[i_swath, i_beam] * factor

    @staticmethod
    @numba.njit(cache=True, fastmath=True, parallel=True)
    def __compute_validity_flags(
        from_swath: int,
        to_swath: int,
        antennas: np.ndarray,
        c_flags: np.ndarray,
        b_flags: np.ndarray,
        a_flags: np.ndarray,
        s_flags: np.ndarray,
        out_result_flags: np.ndarray,
    ) -> None:
        """
        return the numpy array of validity flags
        """

        for i_beam in numba.prange(out_result_flags.shape[1]):
            antenna = max(antennas[i_beam], 0)
            b_flag = b_flags[i_beam]
            a_flag = a_flags[antenna]
            for i_swath in range(from_swath, to_swath):
                c_flag = c_flags[i_swath, antenna]
                s_flag = s_flags[i_swath - from_swath, i_beam]
                out_result_flags[i_swath - from_swath, i_beam] = (
                    c_flag >= C_FLAG_VALID
                    and b_flag >= B_FLAG_VALID
                    and a_flag >= A_FLAG_VALID
                    and s_flag >= S_FLAG_VALID
                )

    @staticmethod
    @numba.njit(cache=True, fastmath=True, parallel=True)
    def __flags_to_xsf_status_and_details(
        from_swath: int,
        to_swath: int,
        antennas: np.ndarray,
        c_flags: np.ndarray,
        b_flags: np.ndarray,
        a_flags: np.ndarray,
        s_flags: np.ndarray,
        out_result_status: np.ndarray,
        out_result_details: np.ndarray,
    ) -> None:
        """
        return the numpy array of validity flags
        """

        for i_beam in numba.prange(out_result_status.shape[1]):
            antenna = max(antennas[i_beam], 0)
            b_flag = b_flags[i_beam]
            a_flag = a_flags[antenna]
            for i_swath in range(from_swath, to_swath):
                c_flag = c_flags[i_swath, antenna]
                s_flag = s_flags[i_swath, i_beam]

                status = xd.STATUS_VALID
                status_details = xd.STATUS_DETAIL_UNKNOWN
                if c_flag < C_FLAG_VALID:
                    status |= xd.STATUS_INVALID_SWATH
                if b_flag < B_FLAG_VALID:
                    status |= xd.STATUS_INVALID_SOUNDER_ROW
                if a_flag < A_FLAG_VALID:
                    status |= xd.STATUS_REJECTED

                if s_flag == S_FLAG_DOUBTFUL:
                    status |= xd.STATUS_REJECTED
                    status_details = xd.STATUS_DETAIL_DOUBTFUL
                elif s_flag == S_FLAG_INVALID_OPERATOR:
                    status |= xd.STATUS_REJECTED
                    status_details = xd.STATUS_DETAIL_MANUAL
                elif s_flag == S_FLAG_INVALID_ACQUIS:
                    status |= xd.STATUS_INVALID_ACQUIS
                elif s_flag == S_FLAG_INVALID_AUTO:
                    status |= xd.STATUS_REJECTED
                    status_details = xd.STATUS_DETAIL_AUTO
                elif s_flag == S_FLAG_MISSING:
                    status |= xd.STATUS_INVALID_CONVERSION

                out_result_status[i_swath - from_swath, i_beam] = status
                out_result_details[i_swath - from_swath, i_beam] = status_details

    def __read_distance_scales(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of all distance_scale factors for all antennas
        """
        if self._distance_scales is None:
            self._distance_scales = numpy_utils.to_memmap(self.read_distance_scale())
        return self._distance_scales[from_swath:to_swath]

    def _compute_norm_and_radius(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        return the norm and the radius array (shape is swath_count / antenna_count)
        """
        ellipsoid = crs.Ellipsoid.from_name("WGS 84")
        eccentricity2 = 1.0 - (ellipsoid.semi_minor_metre / ellipsoid.semi_major_metre) ** 2
        ordinates = self.read_ordinate()

        return compute_norms_and_radii(
            lats=ordinates, semi_major_axis=ellipsoid.semi_major_metre, eccentricity2=eccentricity2
        )

    def has_manual_cleaning(self) -> bool:
        return self._has_correction(MANUAL_CLEANING)

    def has_automatic_cleaning(self) -> bool:
        return self._has_correction(AUTOMATIC_CLEANING)

    def has_im_reflectivity_origin(self) -> bool:
        return self._has_correction(IM_REFLECTIVITY_ORIGIN)

    def has_position_correction(self) -> bool:
        return self._has_correction(POSITION_CORRECTION)

    def has_bias_correction(self) -> bool:
        return self._has_correction(BIAS_CORRECTION)

    def has_velocity_correction(self) -> bool:
        return self._has_correction(VELOCITY_CORRECTION)

    def has_draught_correction(self) -> bool:
        return self._has_correction(DRAUGHT_CORRECTION)

    def has_tide_correction(self) -> bool:
        return self._has_correction(TIDE_CORRECTION)

    def _has_correction(self, correction_name) -> bool:
        if correction_name in self.dataset.ncattrs():
            flag = self.dataset.getncattr(correction_name)
            if len(flag) > 0:
                return flag[0] == "\x01"
        return False

    def read_date_time(self) -> np.ndarray:
        """
        return the numpy array of UTC date/time. Shape is (to_swath - from_swath, antenna_count)
        """
        date = self.read_date()  # Julian date
        time = self.read_time()
        return (date - 2440588) * 24 * 3600 + (time / 1000)

    def read_detection_longitude(self) -> np.ndarray | None:
        """
        return the numpy array of longitude of the detection.
        """
        longitudes, _ = next(self.iter_beam_positions(self.sounder_file.swath_count))
        return longitudes

    def read_detection_latitude(self) -> np.ndarray | None:
        """
        return the numpy array of latitude of the detection.
        """
        _, latitudes = next(self.iter_beam_positions(self.sounder_file.swath_count))
        return latitudes

    def read_detection_quality_factor(self) -> np.ndarray | None:
        """
        return the numpy array of the estimated standard deviation as % of the detected depth.
        """
        return self.read_quality()

    def read_detection_tx_beam(self) -> np.ndarray | None:
        """
        return the numpy array of the detection transmit beam index. NOT AVALAIBLE FOR MBG
        """
        return None

    def read_detection_type(self) -> np.ndarray | None:
        """
        return the numpy array of the type of detection.
        """
        s_quality = self.read_s_quality()
        return np.where(s_quality <= 127, 1, 2)  # 1 = AMPLITUDE, 2 = PHASE

    def read_multiping_sequence(self) -> np.ndarray | None:
        """
        return the numpy array of the multiping sequence identifier.
        """
        return self.read_frequency()

    def read_multiping_center_frequency(self) -> np.ndarray | None:
        """
        return the numpy array of the center frequency in transmitted pulse. NOT AVALAIBLE FOR MBG
        """
        return None

    def read_detection_ping_frequency(self) -> np.ndarray | None:
        """
        return the numpy array of the detection ping frequencies.
        """
        return self.read_sonar_frequency()

    #    ____ ____ _  _ ____ ____ ____ ___ ____ ___     ____ _  _ _  _ ____ ___ _ ____ _  _ ____
    #    | __ |___ |\ | |___ |__/ |__|  |  |___ |  \    |___ |  | |\ | |     |  | |  | |\ | [__
    #    |__] |___ | \| |___ |  \ |  |  |  |___ |__/    |    |__| | \| |___  |  | |__| | \| ___]
    #

    def read_a_flag(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbAFlag as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(A_FLAG, np.int8, None, from_index, to_index)

    def read_abscissa(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbAbscissa as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(ABSCISSA, from_index, to_index)

    def read_absorption_coefficient(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbAbsorptionCoefficient as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(ABSORPTION_COEFFICIENT, np.uint16, None, from_index, to_index)

    def read_across_beam_angle(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbAcrossBeamAngle as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(ACROSS_BEAM_ANGLE, from_index, to_index)

    def read_across_distance(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbAcrossDistance as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(ACROSS_DISTANCE, from_index, to_index)

    def read_across_slope(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbAcrossSlope as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(ACROSS_SLOPE, from_index, to_index)

    def read_along_distance(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbAlongDistance as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(ALONG_DISTANCE, from_index, to_index)

    def read_along_slope(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbAlongSlope as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(ALONG_SLOPE, from_index, to_index)

    def read_antenna(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbAntenna as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(ANTENNA, np.int8, None, from_index, to_index)

    def read_azimut_beam_angle(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbAzimutBeamAngle as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(AZIMUT_BEAM_ANGLE, np.uint16, float, from_index, to_index)

    def read_b_flag(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbBFlag as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(B_FLAG, np.int8, None, from_index, to_index)

    def read_b_s_p_status(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbBSPStatus as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(B_S_P_STATUS, np.int8, None, from_index, to_index)

    def read_beam(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbBeam as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(BEAM, np.uint16, None, from_index, to_index)

    def read_beam_spacing(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbBeamSpacing as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(BEAM_SPACING, np.int8, None, from_index, to_index)

    def read_c_flag(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbCFlag as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(C_FLAG, np.int8, None, from_index, to_index)

    def read_c_quality(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbCQuality as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(C_QUALITY, np.int8, None, from_index, to_index)

    def read_compensation_layer_mode(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbCompensationLayerMode as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(COMPENSATION_LAYER_MODE, np.int8, None, from_index, to_index)

    def read_cycle(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbCycle as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(CYCLE, from_index, to_index)

    def read_date(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbDate as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(DATE, from_index, to_index)

    def read_depth(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbDepth as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(DEPTH, from_index, to_index)

    def read_distance_scale(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbDistanceScale as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(DISTANCE_SCALE, np.int8, float, from_index, to_index)

    def read_durotong_speed(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbDurotongSpeed as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(DUROTONG_SPEED, np.uint16, None, from_index, to_index)

    def read_dynamic_draught(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbDynamicDraught as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(DYNAMIC_DRAUGHT, from_index, to_index)

    def read_filter_identifier(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbFilterIdentifier as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(FILTER_IDENTIFIER, np.int8, None, from_index, to_index)

    def read_frequency(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbFrequency as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(FREQUENCY, np.int8, None, from_index, to_index)

    def read_heading(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbHeading as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(HEADING, np.uint16, float, from_index, to_index)

    def read_hi_lo_absorption_ratio(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbHiLoAbsorptionRatio as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(HI_LO_ABSORPTION_RATIO, np.int8, None, from_index, to_index)

    def read_hist_autor(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbHistAutor as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(HIST_AUTOR, np.int8, None, from_index, to_index)

    def read_hist_code(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbHistCode as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(HIST_CODE, np.int8, None, from_index, to_index)

    def read_hist_comment(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbHistComment as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(HIST_COMMENT, np.int8, None, from_index, to_index)

    def read_hist_date(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbHistDate as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(HIST_DATE, from_index, to_index)

    def read_hist_module(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbHistModule as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(HIST_MODULE, np.int8, None, from_index, to_index)

    def read_hist_time(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbHistTime as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(HIST_TIME, from_index, to_index)

    def read_interlacing(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbInterlacing as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(INTERLACING, np.int8, None, from_index, to_index)

    def read_max_port_coverage(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbMaxPortCoverage as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(MAX_PORT_COVERAGE, np.int8, None, from_index, to_index)

    def read_max_port_width(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbMaxPortWidth as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(MAX_PORT_WIDTH, np.uint16, None, from_index, to_index)

    def read_max_starboard_coverage(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbMaxStarboardCoverage as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(MAX_STARBOARD_COVERAGE, np.int8, None, from_index, to_index)

    def read_max_starboard_width(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbMaxStarboardWidth as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(MAX_STARBOARD_WIDTH, np.uint16, None, from_index, to_index)

    def read_operator_station_status(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbOperatorStationStatus as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(OPERATOR_STATION_STATUS, np.int8, None, from_index, to_index)

    def read_ordinate(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbOrdinate as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(ORDINATE, from_index, to_index)

    def read_param_maximum_depth(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbParamMaximumDepth as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(PARAM_MAXIMUM_DEPTH, np.uint16, None, from_index, to_index)

    def read_param_minimum_depth(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbParamMinimumDepth as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(PARAM_MINIMUM_DEPTH, np.uint16, None, from_index, to_index)

    def read_pitch(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbPitch as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(PITCH, from_index, to_index)

    def read_processing_unit_status(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbProcessingUnitStatus as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(PROCESSING_UNIT_STATUS, np.int8, None, from_index, to_index)

    def read_quality(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbQuality as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(QUALITY, from_index, to_index)

    def read_range(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbRange as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(RANGE, from_index, to_index)

    def read_receive_bandwidth(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbReceiveBandwidth as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(RECEIVE_BANDWIDTH, np.int8, None, from_index, to_index)

    def read_receive_beamwidth(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbReceiveBeamwidth as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(RECEIVE_BEAMWIDTH, np.int8, None, from_index, to_index)

    def read_receiver_fixed_gain(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbReceiverFixedGain as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(RECEIVER_FIXED_GAIN, np.int8, None, from_index, to_index)

    def read_reception_heave(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbReceptionHeave as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(RECEPTION_HEAVE, np.int8, float, from_index, to_index)

    def read_reference_depth(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbReferenceDepth as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(REFERENCE_DEPTH, from_index, to_index)

    def read_reflectivity(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbReflectivity as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(REFLECTIVITY, np.int8, float, from_index, to_index)

    def read_roll(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbRoll as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(ROLL, from_index, to_index)

    def read_s_flag(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbSFlag as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(S_FLAG, np.int8, None, from_index, to_index)

    def read_s_length_of_detection(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbSLengthOfDetection as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(S_LENGTH_OF_DETECTION, np.int8, None, from_index, to_index)

    def read_s_quality(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbSQuality as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(S_QUALITY, np.int8, None, from_index, to_index)

    def read_sampling_rate(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbSamplingRate as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(SAMPLING_RATE, np.uint16, None, from_index, to_index)

    def read_sonar_frequency(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbSonarFrequency as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(SONAR_FREQUENCY, from_index, to_index)

    def read_sonar_status(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbSonarStatus as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(SONAR_STATUS, np.int8, None, from_index, to_index)

    def read_sound_velocity(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbSoundVelocity as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(SOUND_VELOCITY, from_index, to_index)

    def read_sounder_mode(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbSounderMode as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(SOUNDER_MODE, np.int8, None, from_index, to_index)

    def read_sounding_bias(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbSoundingBias as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(SOUNDING_BIAS, from_index, to_index)

    def read_t_v_g_law_crossover_angle(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbTVGLawCrossoverAngle as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(T_V_G_LAW_CROSSOVER_ANGLE, np.int8, None, from_index, to_index)

    def read_tide(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbTide as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(TIDE, from_index, to_index)

    def read_time(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbTime as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(TIME, from_index, to_index)

    def read_trans_velocity_source(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbTransVelocitySource as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(TRANS_VELOCITY_SOURCE, np.int8, None, from_index, to_index)

    def read_transmission_heave(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbTransmissionHeave as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(TRANSMISSION_HEAVE, from_index, to_index)

    def read_transmit_beamwidth(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbTransmitBeamwidth as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(TRANSMIT_BEAMWIDTH, np.uint16, None, from_index, to_index)

    def read_transmit_power_re_max(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbTransmitPowerReMax as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(TRANSMIT_POWER_RE_MAX, np.int8, None, from_index, to_index)

    def read_transmit_pulse_length(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbTransmitPulseLength as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(TRANSMIT_PULSE_LENGTH, np.uint16, None, from_index, to_index)

    def read_vel_profil_date(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbVelProfilDate as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(VEL_PROFIL_DATE, from_index, to_index)

    def read_vel_profil_idx(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbVelProfilIdx as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(VEL_PROFIL_IDX, from_index, to_index)

    def read_vel_profil_ref(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbVelProfilRef as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(VEL_PROFIL_REF, np.int8, None, from_index, to_index)

    def read_vel_profil_time(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbVelProfilTime as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(VEL_PROFIL_TIME, from_index, to_index)

    def read_vertical_depth(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbVerticalDepth as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer(VERTICAL_DEPTH, from_index, to_index)

    def read_yaw_pitch_stab_mode(self, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the variable mbYawPitchStabMode as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        return self.__read_layer_as(YAW_PITCH_STAB_MODE, np.int8, None, from_index, to_index)

    def __apply_offset_and_scale(self, variable: nc.Variable, data: np.ndarray) -> None:
        """
        Apply the offset and scale if present
        Generated with nc_driver_skeleton_generator.py
        """
        if "scale_factor" in variable.ncattrs():
            np.multiply(data, variable.scale_factor, out=data)
        if "add_offset" in variable.ncattrs():
            np.add(data, variable.add_offset, out=data)

    def __read_layer(self, layer_name: str, from_index: int = None, to_index: int = None) -> np.ndarray:
        """
        return the data of the specified variable as a numpy array.
        Generated with nc_driver_skeleton_generator.py
        """
        variable = self.dataset[layer_name]
        variable.set_auto_mask(False)
        if from_index is None and to_index is None:
            return self.dataset[layer_name][:]
        if from_index is not None and to_index is None:
            return self.dataset[layer_name][from_index:]
        if from_index is None and to_index is not None:
            return self.dataset[layer_name][:to_index]
        return self.dataset[layer_name][from_index:to_index]

    def __read_layer_as(
        self,
        layer_name: str,
        from_numpy_dtype=np.int8,
        to_numpy_dtype=None,
        from_index: int = None,
        to_index: int = None,
    ) -> np.ndarray:
        """
        return the data of the specified variable as a numpy array of a specific type.
        Generated with nc_driver_skeleton_generator.py
        """
        variable = self.dataset[layer_name]
        variable.set_auto_maskandscale(False)

        data = self.__read_layer(layer_name, from_index, to_index)
        result = np.frombuffer(data, dtype=from_numpy_dtype).reshape(data.shape)

        if to_numpy_dtype is not None:
            result = result.astype(to_numpy_dtype)

        self.__apply_offset_and_scale(variable, result)

        return result


class BeamPositionIterator:
    def __init__(self, driver: MbgDriver, swath_count_by_iter: int, first_swath: int, valid_only: bool = False):
        self.driver = driver
        self.swath_count_by_iter = swath_count_by_iter
        self.swath = first_swath
        self.valid_only = valid_only

        self.norm, self.radius = self.driver._compute_norm_and_radius()
        self.headings = self.driver.read_heading()
        self.longitudes = self.driver.read_abscissa()
        self.latitudes = self.driver.read_ordinate()

        # Reusable buffers for __next__ returned values
        self._result_lon: Optional[np.ndarray] = None
        self._result_lat: Optional[np.ndarray] = None

    def __iter__(self):
        return self

    @staticmethod
    @numba.njit(
        "void(float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], int8[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])",
        cache=True,
        fastmath=False,
    )
    def __compute_lon_lat(
        norm: np.ndarray,
        radius: np.ndarray,
        headings: np.ndarray,
        along_distances: np.ndarray,
        across_distances: np.ndarray,
        antennas: np.ndarray,
        longitudes: np.ndarray,
        latitudes: np.ndarray,
        out_long: np.ndarray,
        out_lat: np.ndarray,
    ):
        for i_swath in range(out_long.shape[0]):
            for i_beam in range(out_long.shape[1]):
                antenna = max(antennas[i_beam], 0)
                cos_heading = math.cos(math.radians(headings[i_swath, antenna]))
                sin_heading = math.sin(math.radians(headings[i_swath, antenna]))
                out_lat[i_swath, i_beam] = (
                    latitudes[i_swath, antenna]
                    + math.degrees(
                        along_distances[i_swath, i_beam] * cos_heading - across_distances[i_swath, i_beam] * sin_heading
                    )
                    / radius[i_swath, antenna]
                )

                out_long[i_swath, i_beam] = longitudes[i_swath, antenna] + math.degrees(
                    along_distances[i_swath, i_beam] * sin_heading + across_distances[i_swath, i_beam] * cos_heading
                ) / norm[i_swath, antenna] / math.cos(math.radians(latitudes[i_swath, antenna]))

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        # stop ?
        if self.swath >= self.driver.sounder_file.swath_count:
            self._result_lon = self._result_lat = None
            raise StopIteration()

        last_swath = min(self.swath + self.swath_count_by_iter, self.driver.sounder_file.swath_count)

        # Initialize buffers
        if self._result_lon is None or self._result_lon.shape[0] != last_swath - self.swath:
            shape = (last_swath - self.swath, self.driver.sounder_file.beam_count)
            self._result_lon = np.empty(shape, dtype=float)
            self._result_lat = np.empty(shape, dtype=float)

        antennas = self.driver.read_antenna()
        across_distances = self.driver.read_across_distances(self.swath, last_swath)
        along_distances = self.driver.read_along_distances(self.swath, last_swath)
        BeamPositionIterator.__compute_lon_lat(
            self.norm[self.swath : last_swath],
            self.radius[self.swath : last_swath],
            self.headings[self.swath : last_swath],
            along_distances,
            across_distances,
            antennas,
            self.longitudes[self.swath : last_swath],
            self.latitudes[self.swath : last_swath],
            self._result_lon,
            self._result_lat,
        )
        if self.valid_only:
            # if asked, mask invalid soundings positions
            is_valid = self.driver.read_validity_flags(self.swath, last_swath)
            self._result_lon[~is_valid] = np.nan
            self._result_lat[~is_valid] = np.nan
        self.swath = last_swath
        return self._result_lon, self._result_lat


@contextmanager
def open_mbg(file_path: str, mode: str = "r") -> Generator[MbgDriver, None, None]:
    """
    Define a With Statement Context Managers for a MbgDriver
    Allow opening a MbgDriver in a With Statement
    """
    driver = MbgDriver(file_path)
    driver.open(mode)
    try:
        yield driver
    finally:
        driver.close()
