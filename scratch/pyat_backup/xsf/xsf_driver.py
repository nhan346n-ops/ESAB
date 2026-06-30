#! /usr/bin/env python3
# coding: utf-8
import datetime
import json
from contextlib import contextmanager
from typing import Generator, Iterable, Optional, Tuple

import netCDF4 as nc
import numba
import numpy as np
import numpy.core.umath as npmath
import sonar_netcdf.sonar_groups as constants

from pyat.sounder import sounder_driver
from pyat.utils import netcdf_utils, numpy_utils, signal
from pyat.utils.nc_encoding import open_nc_file
from pyat.utils.path_utils import ext_of_fname

BEAM_GROUP_NAME = "Beam_group1"

PING_TIME = constants.BeamGroup1Grp.PING_TIME(BEAM_GROUP_NAME)
PLATFORM_VERTICAL_OFFSET = constants.BeamGroup1Grp.PLATFORM_VERTICAL_OFFSET(BEAM_GROUP_NAME)
WATERLINE_TO_CHART_DATUM = constants.BeamGroup1Grp.WATERLINE_TO_CHART_DATUM(BEAM_GROUP_NAME)
PLATFORM_LONGITUDE = constants.BeamGroup1Grp.PLATFORM_LONGITUDE(BEAM_GROUP_NAME)
PLATFORM_LATITUDE = constants.BeamGroup1Grp.PLATFORM_LATITUDE(BEAM_GROUP_NAME)
PLATFORM_HEADING = constants.BeamGroup1Grp.PLATFORM_HEADING(BEAM_GROUP_NAME)
PLATFORM_PITCH = constants.BeamGroup1Grp.PLATFORM_PITCH(BEAM_GROUP_NAME)
PLATFORM_ROLL = constants.BeamGroup1Grp.PLATFORM_ROLL(BEAM_GROUP_NAME)
TX_TRANSDUCER_DEPTH = constants.BeamGroup1Grp.TX_TRANSDUCER_DEPTH(BEAM_GROUP_NAME)

DETECTION_X = constants.BathymetryGrp.DETECTION_X(BEAM_GROUP_NAME)
DETECTION_Y = constants.BathymetryGrp.DETECTION_Y(BEAM_GROUP_NAME)
DETECTION_Z = constants.BathymetryGrp.DETECTION_Z(BEAM_GROUP_NAME)
STATUS = constants.BathymetryGrp.STATUS(BEAM_GROUP_NAME)
STATUS_DETAIL = constants.BathymetryGrp.STATUS_DETAIL(BEAM_GROUP_NAME)
DETECTION_BACKSCATTER_R = constants.BathymetryGrp.DETECTION_BACKSCATTER_R(BEAM_GROUP_NAME)
DETECTION_LONGITUDE = constants.BathymetryGrp.DETECTION_LONGITUDE(BEAM_GROUP_NAME)
DETECTION_LATITUDE = constants.BathymetryGrp.DETECTION_LATITUDE(BEAM_GROUP_NAME)
DETECTION_BEAM_POINTING_ANGLE = constants.BathymetryGrp.DETECTION_BEAM_POINTING_ANGLE(BEAM_GROUP_NAME)
DETECTION_QUALITY_FACTOR = constants.BathymetryGrp.DETECTION_QUALITY_FACTOR(BEAM_GROUP_NAME)
DETECTION_TX_BEAM = constants.BathymetryGrp.DETECTION_TX_BEAM(BEAM_GROUP_NAME)
DETECTION_TYPE = constants.BathymetryGrp.DETECTION_TYPE(BEAM_GROUP_NAME)
DETECTION_RX_TRANSDUCER_INDEX = constants.BathymetryGrp.DETECTION_RX_TRANSDUCER_INDEX(BEAM_GROUP_NAME)

MULTIPING_SEQUENCE = constants.BathymetryGrp.MULTIPING_SEQUENCE(BEAM_GROUP_NAME)
CENTER_FREQUENCY = constants.BeamGroup1VendorSpecificGrp.CENTER_FREQUENCY(BEAM_GROUP_NAME)
DETECTION_PING_FREQUENCY = constants.BathymetryVendorSpecificGrp.DETECTION_PING_FREQUENCY(BEAM_GROUP_NAME)

POSITION_OFFSET_X = constants.PlatformGrp.POSITION_OFFSET_X()
POSITION_OFFSET_Y = constants.PlatformGrp.POSITION_OFFSET_Y()
POSITION_OFFSET_Z = constants.PlatformGrp.POSITION_OFFSET_Z()

TRANSDUCER_OFFSET_X = constants.PlatformGrp.TRANSDUCER_OFFSET_X()
TRANSDUCER_OFFSET_Y = constants.PlatformGrp.TRANSDUCER_OFFSET_Y()
TRANSDUCER_OFFSET_Z = constants.PlatformGrp.TRANSDUCER_OFFSET_Z()

WATERLEVEL = constants.PlatformGrp.WATER_LEVEL()
DELTA_DRAUGHT = constants.DynamicDraughtGrp.DELTA_DRAUGHT()
TIDE_INDICATIVE = constants.TideGrp.TIDE_INDICATIVE()

# ATTRIBUTES CONSTANTS
ATT_HISTORY = "history"
ATT_PROCESSING_STATUS = "processing_status"

# Processing status flags
ATT_PROCESSING_STATUS_FLAG_ON = 1
ATT_PROCESSING_STATUS_FLAG_OFF = 0

# processing status fields
ATT_PROCESSING_STATUS_VELOCITY_CORRECTION = "velocityCorrection"
ATT_PROCESSING_STATUS_MANUAL_CLEANING = "manualCleaning"
ATT_PROCESSING_STATUS_AUTOMATIC_CLEANING = "automaticCleaning"
ATT_PROCESSING_STATUS_BIAS_CORRECTION = "biasCorrection"
ATT_PROCESSING_STATUS_TIDE_CORRECTION = "tideCorrection"
ATT_PROCESSING_STATUS_POSITION_CORRECTION = "positionCorrection"
ATT_PROCESSING_STATUS_DRAUGHT_CORRECTION = "draughtCorrection"
ATT_PROCESSING_STATUS_BACKSCATTER_CORRECTION = "backscatterCorrection"
# specific status field containing list of bias correctors already applied
ATT_PROCESSING_STATUS_BIAS_CORRECTORS = "biasCorrectors"

# bitmask for status
STATUS_VALID = 0x0
STATUS_REJECTED = 0x1
STATUS_INVALID_ACQUIS = 0x2
STATUS_INVALID_CONVERSION = 0x4
STATUS_INVALID_SWATH = 0x8
STATUS_INVALID_SOUNDER_ROW = 0x10

# values for status detail
STATUS_DETAIL_UNKNOWN = 0
STATUS_DETAIL_AUTO = 1
STATUS_DETAIL_DOUBTFUL = 2
STATUS_DETAIL_MANUAL = 3


class XsfDriver(sounder_driver.SounderDriver):
    @property
    def dataset(self) -> nc.Dataset:
        return self._dataset

    def __init__(self, file_path: str):
        super().__init__(file_path)

        self._dataset = None

        # Keep this layers in memory
        self._fcs_depths: Optional[np.ndarray] = None
        self._scs_depths: Optional[np.ndarray] = None
        self._across_angles: Optional[np.ndarray] = None

    def open(self, mode: str = "r") -> nc.Dataset:
        """
        Open the file and return the resulting Dataset
        """
        if self._dataset is not None:
            # already opened
            return self._dataset
        self._dataset = open_nc_file(self.sounder_file.file_path, mode=mode)
        if not str(self.dataset.file_format).startswith("NETCDF4"):
            self.dataset.close()
            raise ValueError(
                f"The format of the file {self.sounder_file.file_path} must be NETCDF4 (instead of {self.dataset.file_format})."
            )
        # TODO improve/refactor XSF to allow to load sonarnetcdf without bathymetry
        try:
            shape = self[DETECTION_Z].shape
            self.sounder_file.swath_count = shape[0]
            self.sounder_file.beam_count = shape[1]
        except AttributeError as e:
            raise ValueError(f"Bad XSF format of the file {self.sounder_file.file_path}. Unable to parse it.") from e
        except KeyError as e:
            raise ValueError(f"No WC beam in {self.sounder_file.file_path}. ") from e

        return self.dataset

    def close(self) -> None:
        """Close the dataset if opened"""
        if self.dataset and self.dataset.isopen():
            self.dataset.close()
        self._dataset = None

    def __getitem__(self, layer_name: str) -> nc.Variable:
        """return the layer called layer_name"""
        result = self.dataset[layer_name]
        result.set_auto_mask(False)
        return result

    def get_layer(self, layer_path: str) -> Optional[nc.Variable]:
        """return the nc variable designated by the path layer_path"""
        return netcdf_utils.get_variable(i_dataset=self.dataset, variable_path=layer_path)

    def get_provenance_ext(self) -> str | None:
        """
        return file extension of first provenance file or None file list is empty
        """
        # use netcdf api to ensure that group really exist
        if constants.ProvenanceGrp.SOURCE_FILENAMES_VNAME in self[constants.ProvenanceGrp.get_group_path()].variables:
            filenames = self[constants.ProvenanceGrp.SOURCE_FILENAMES()][:]
            # check extension
            if len(filenames) > 0:
                return ext_of_fname(filenames[0])
        return None

    def read_validity_flags(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of validity flags
        """
        return np.logical_not(self[STATUS][from_swath:to_swath, :])

    def read_fcs_depths(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of depths. Shape is (to_swath - from_swath, beam_count)
        Depths are projected in Coordinates system transformations FCS (Fixed Coordinate System)
        """
        if self._fcs_depths is None:
            vertical_offsets = self[PLATFORM_VERTICAL_OFFSET][:]
            waterline_to_chart_datum = self[WATERLINE_TO_CHART_DATUM][:]
            self._fcs_depths = numpy_utils.to_memmap(self[DETECTION_Z][:])
            XsfDriver.__adjust_depths(self._fcs_depths, vertical_offsets, waterline_to_chart_datum)
        return self._fcs_depths[from_swath:to_swath]

    def read_scs_depths(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of depths. Shape is (to_swath - from_swath, beam_count)
        Depths are projected in Coordinates system transformations SCS (Surface Coordinate System)
        For a XSF, this exactly the DETECTION_Z layer
        """
        if self._scs_depths is None:
            self._scs_depths = numpy_utils.to_memmap(self[DETECTION_Z][:])
        return self._scs_depths[from_swath:to_swath]

    @staticmethod
    @numba.njit("void(float32[:,:], float32[:], float32[:])", cache=True, fastmath=True)
    def __adjust_depths(depths: np.ndarray, vertical_offsets: np.ndarray, waterline_to_chart_datum: np.ndarray):
        for i_swath in range(depths.shape[0]):
            for i_beam in range(depths.shape[1]):
                depths[i_swath, i_beam] = (
                    depths[i_swath, i_beam] - vertical_offsets[i_swath] - waterline_to_chart_datum[i_swath]
                )

    def read_reflectivities(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of Reflectivity values of all antennas
        """
        try:
            bs_var = self[DETECTION_BACKSCATTER_R]
            # convert to dB if unit is magnitude
            if "units" in bs_var.ncattrs() and bs_var.getncattr("units") != "dB":
                return signal.amplitude_to_db(bs_var[from_swath:to_swath].astype(np.float64))
            else:
                return bs_var[from_swath:to_swath]
        except IndexError:
            # Detection backscatter is a mandatory variable, but some files historically have been found without it
            return np.full((to_swath - from_swath, int(self.sounder_file.beam_count)), np.nan)

    def read_across_distances(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of across distance. Shape is (to_swath - from_swath, beam_count)
        """
        return self[DETECTION_Y][from_swath:to_swath]

    def read_vertical_distances(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of vertical distance. Shape is (to_swath - from_swath, beam_count)
        """
        return self[DETECTION_Z][from_swath:to_swath]

    def read_transducer_depth(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of vertical distance. Shape is (to_swath - from_swath, beam_count)
        """
        return self[TX_TRANSDUCER_DEPTH][from_swath:to_swath]

    def read_sound_speed_at_transducer(self) -> np.ndarray:
        """
        returns the numpy float64 array of sound speed at transducer. shape is (swath_count,).
        """
        return self[constants.BeamGroup1Grp.SOUND_SPEED_AT_TRANSDUCER(BEAM_GROUP_NAME)][:].astype(np.float64)

    def read_sound_speed_profiles_times(self) -> np.ndarray:
        """
        returns the numpy datetime64[ns] array of sound speed profiles times. shape is (n_profiles,).
        """
        return self[constants.SoundSpeedProfileGrp.PROFILE_TIME()][:].astype("datetime64[ns]")

    def read_sound_speed_profile(self, ssp_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns the sound speed profile at given index, as a numpy float64 array.
        :param profile_index: index of the sound speed profile to read
        :return: tuple of (depths, sound speeds) as numpy arrays
        """
        sound_speeds = self[constants.SoundSpeedProfileGrp.SOUND_SPEED()][ssp_idx]
        depths = self[constants.SoundSpeedProfileGrp.SAMPLE_DEPTH()][ssp_idx]
        # Keep only unique SVP values
        repeated_values_ix = np.where(np.diff(sound_speeds) == 0)[0]
        depths = np.delete(depths, repeated_values_ix)
        sound_speeds = np.delete(sound_speeds, repeated_values_ix)

        return depths.astype(np.float64), sound_speeds.astype(np.float64)

    def get_ssp_idx(self) -> np.ndarray:
        """
        returns the index of the closest preceding sound speed profile time for each ping time.
        :return: numpy array of sound speed profile indices, shape is (swath_count,).
        """
        ping_time = self.read_ping_times()
        svp_time = self.read_sound_speed_profiles_times()
        ssp_idx = np.searchsorted(svp_time, ping_time) - 1
        ssp_idx[ssp_idx < 0] = 0  # if ping time is before first ssp time, use first ssp
        return ssp_idx

    def read_across_angles(self, from_swath: int, to_swath: int) -> np.ndarray:
        """
        return the numpy array of across angles. Shape is (to_swath - from_swath, beam_count)
        Implementation of SounderDriver abstract method
        """
        if self._across_angles is None:
            transducer_offset_y = self[TRANSDUCER_OFFSET_Y][:]
            transducer_offset_z = self[TRANSDUCER_OFFSET_Z][:]
            transducer_index = self[DETECTION_RX_TRANSDUCER_INDEX][:]
            invalid = transducer_index < 0
            transducer_index[invalid] = 0
            rx_offset_y = np.array([transducer_offset_y[idx] for idx in transducer_index])
            rx_offset_z = np.array([transducer_offset_z[idx] for idx in transducer_index])
            rx_detection_y = self[DETECTION_Y][:] - rx_offset_y
            rx_detection_z = self[DETECTION_Z][:] - rx_offset_z
            self._across_angles = numpy_utils.to_memmap(npmath.rad2deg(npmath.arctan2(rx_detection_y, rx_detection_z)))
            self._across_angles[invalid] = np.nan
        return self._across_angles[from_swath:to_swath]

    def read_platform_longitudes(self) -> np.ndarray:
        """
        Implementation of SounderDriver abstract method
        """
        return self[PLATFORM_LONGITUDE][:]

    def read_platform_latitudes(self) -> np.ndarray:
        """
        Implementation of SounderDriver abstract method
        """
        return self[PLATFORM_LATITUDE][:]

    def read_platform_headings(self) -> np.ndarray:
        """
        Implementation of SounderDriver abstract method
        """
        return self[PLATFORM_HEADING][:]

    def read_ping_times(self) -> np.ndarray:
        """
        Returns ping time as datetime64[ns] array.
        Implementation of SounderDriver abstract method
        """
        return self[PING_TIME][:].astype("datetime64[ns]")

    def read_platform_vertical_offsets(self) -> np.ndarray:
        """
        Implementation of SounderDriver abstract method
        """
        return self[PLATFORM_VERTICAL_OFFSET][:]

    def get_waterlevel(self) -> float:
        """
        return the nominal waterlevel or 0 if not available
        """
        try:
            return float(self[WATERLEVEL][0])
        except Exception:
            return 0.0

    def get_preferred_position_subgroup_id(self) -> str:
        """
        return preferred platform position subgroup or first one if not found
        """
        preferred_position = self[constants.BeamGroup1Grp.get_group_path(BEAM_GROUP_NAME)].preferred_position
        position_id = None
        # now retrieve the name of the sensor
        # use netcdf api to ensure that group really exist
        if constants.PlatformGrp.POSITION_IDS_VNAME in self[constants.PlatformGrp.get_group_path()].variables:
            position_ids = self[constants.PlatformGrp.POSITION_IDS()][:]
            position_id = position_ids[preferred_position]

        # check position_id
        # position ids are not always well set, we use default value if an error is in file
        # use netcdf api to ensure that group really exist
        if position_id not in self[constants.PositionGrp.get_group_path()].groups:
            # we use the first group found as default position_id
            position_id = next(iter(self[constants.PositionGrp.get_group_path()].groups))
        return position_id

    def read_position_longitudes(self) -> np.ndarray:
        """
        Returns preferred position sensor longitude as float32
        """
        return self[constants.PositionSubGroup.LONGITUDE(self.get_preferred_position_subgroup_id())][:]

    def read_position_latitudes(self) -> np.ndarray:
        """
        Returns preferred position sensor latitude as float32
        """
        return self[constants.PositionSubGroup.LATITUDE(self.get_preferred_position_subgroup_id())][:]

    def read_position_times(self) -> np.ndarray:
        """
        Returns preferred position sensor time as datetime64 with nanosecond precision
        """
        time = self[constants.PositionSubGroup.TIME(self.get_preferred_position_subgroup_id())]
        return time[:].astype("datetime64[ns]")

    def read_position_nmea(self) -> np.ndarray:
        """
        Returns raw NMEA strings from preferred position sensor
        """
        return self[
            constants.PositionSubGroupVendorSpecificGrp.DATA_RECEIVED_FROM_SENSOR(
                self.get_preferred_position_subgroup_id()
            )
        ][:]

    def read_position_height_above_ellipsoid(self) -> np.ndarray:
        """
        Implementation of SounderDriver abstract method
        """
        return self[
            constants.PositionSubGroup.HEIGHT_ABOVE_REFERENCE_ELLIPSOID(self.get_preferred_position_subgroup_id())
        ][:]

    def read_position_sensor_quality_indicators(self) -> np.ndarray:
        """
        Implementation of SounderDriver abstract method
        """
        return self[
            constants.PositionSubGroupVendorSpecificGrp.SENSOR_QUALITY_INDICATOR(
                self.get_preferred_position_subgroup_id()
            )
        ][:]

    def read_position_offset(self) -> np.ndarray:
        """
        Returns the platform position distance from the platform coordinate system origin to the latitude/longitude sensor origin
        """
        pos_idx = self[constants.BeamGroup1Grp.get_group_path(BEAM_GROUP_NAME)].preferred_position
        return np.asarray(
            (self[POSITION_OFFSET_X][pos_idx], self[POSITION_OFFSET_Y][pos_idx], self[POSITION_OFFSET_Z][pos_idx])
        )

    def get_preferred_attitude_subgroup_id(self) -> str:
        """
        return preferred platform attitude subgroup or first one if not found
        """
        preferred_attitude_idx = self[constants.BeamGroup1Grp.get_group_path(BEAM_GROUP_NAME)].preferred_MRU
        attitude_id = None
        # now retrieve the name of the sensor, use netcdf api to ensure that group really exist
        if constants.PlatformGrp.MRU_IDS_VNAME in self[constants.PlatformGrp.get_group_path()].variables:
            attitude_ids = self[constants.PlatformGrp.MRU_IDS()][:]
            attitude_id = attitude_ids[preferred_attitude_idx]
        # check preferred_MRU really exists
        if attitude_id not in self[constants.AttitudeGrp.get_group_path()].groups:
            # we use the first group found as default attitude_id
            attitude_id = next(iter(self[constants.AttitudeGrp.get_group_path()].groups))
        return attitude_id

    def read_attitude_offset(self) -> np.ndarray:
        """
        Returns vector from the platform coordinate system origin to the preferred MRU origin
        """
        mru_idx = self[constants.BeamGroup1Grp.get_group_path(BEAM_GROUP_NAME)].preferred_MRU
        return np.asarray(
            (
                self[constants.PlatformGrp.MRU_OFFSET_X()][mru_idx],
                self[constants.PlatformGrp.MRU_OFFSET_Y()][mru_idx],
                self[constants.PlatformGrp.MRU_OFFSET_Z()][mru_idx],
            )
        )

    def read_attitude_times(self) -> np.ndarray:
        """
        Returns the numpy array of times from preffered_MRU (nanoseconds since 1970-01-01 00:00:00Z).
        """
        return self[constants.AttitudeSubGroup.TIME(self.get_preferred_attitude_subgroup_id())][:]

    def read_attitude_rolls(self) -> np.ndarray:
        """
        Returns the numpy array of rolls from preffered_MRU.
        """
        return self[constants.AttitudeSubGroup.ROLL(self.get_preferred_attitude_subgroup_id())][:]

    def read_attitude_pitches(self) -> np.ndarray:
        """
        Returns the numpy array of pitches from preffered_MRU.
        """

        return self[constants.AttitudeSubGroup.PITCH(self.get_preferred_attitude_subgroup_id())][:]

    def read_attitude_headings(self) -> np.ndarray:
        """
        Returns the numpy array of headings from preffered_MRU.
        """
        return self[constants.AttitudeSubGroup.HEADING(self.get_preferred_attitude_subgroup_id())][:]

    def read_attitude_vertical_offsets(self) -> np.ndarray:
        """
        Returns the numpy array of vertical offsets from preffered_MRU.
        """
        return self[constants.AttitudeSubGroup.VERTICAL_OFFSET(self.get_preferred_attitude_subgroup_id())][:]

    def get_preferred_depth_subgroup_id(self) -> str | None:
        """
        return preferred platform depth subgroup or first one if not found, or None if no depth sensor
        """
        preferred_depth_idx = self[constants.BeamGroup1Grp.get_group_path(BEAM_GROUP_NAME)].preferred_depth
        depth_id = None
        if preferred_depth_idx >= 0:  # negative value means no depth sensor, hence no depth group
            # now retrieve the name of the sensor, use netcdf api to ensure that group really exist
            if constants.PlatformGrp.DEPTH_IDS_VNAME in self[constants.PlatformGrp.get_group_path()].variables:
                depth_ids = self[constants.PlatformGrp.DEPTH_IDS()][:]
                depth_id = depth_ids[preferred_depth_idx]
            # check preferred_depth really exists
            if depth_id not in self[constants.DepthGrp.get_group_path()].groups:
                # we use the first group found as default depth_id
                depth_id = next(iter(self[constants.DepthGrp.get_group_path()].groups))
        return depth_id

    def read_depth_sensor_offset(self) -> np.ndarray | None:
        """
        Returns distance from the platform coordinate system origin to the depth sensor origin.
        """
        preferred_depth_idx = self[constants.BeamGroup1Grp.get_group_path(BEAM_GROUP_NAME)].preferred_depth
        if preferred_depth_idx >= 0:
            return np.asarray(
                (
                    self[constants.PlatformGrp.DEPTH_OFFSET_X()][preferred_depth_idx],
                    self[constants.PlatformGrp.DEPTH_OFFSET_Y()][preferred_depth_idx],
                    self[constants.PlatformGrp.DEPTH_OFFSET_Z()][preferred_depth_idx],
                )
            )
        else:  # negative value means no depth sensor
            return None

    def read_depth_sensor_times(self) -> np.ndarray | None:
        """
        Returns the numpy array of times from preffered_depth (nanoseconds since 1970-01-01 00:00:00Z).
        """
        preferred_depth_id = self.get_preferred_depth_subgroup_id()
        if preferred_depth_id is not None:
            return self[constants.DepthSubGroup.TIME(preferred_depth_id)][:]
        else:
            return None

    def read_depth_sensor_vertical_offset(self) -> np.ndarray | None:
        """
        Returns the numpy array of depths from preffered_depth.
        """
        preferred_depth_id = self.get_preferred_depth_subgroup_id()
        if preferred_depth_id is not None:
            return self[constants.DepthSubGroup.VERTICAL_OFFSET(preferred_depth_id)][:]
        else:
            return None

    def iter_beam_positions(
        self, swath_count_by_iter: int, first_swath: int = 0
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        return BeamPositionIterator(self, swath_count_by_iter, first_swath)

    def read_detection_longitude(self) -> np.ndarray | None:
        """
        return the numpy array of longitude of the detection.
        """
        return self[DETECTION_LONGITUDE][:]

    def read_detection_latitude(self) -> np.ndarray | None:
        """
        return the numpy array of latitude of the detection.
        """
        return self[DETECTION_LATITUDE][:]

    def read_detection_quality_factor(self) -> np.ndarray | None:
        """
        return the numpy array of the estimated standard deviation as % of the detected depth.
        """
        return self[DETECTION_QUALITY_FACTOR][:]

    def read_detection_tx_beam(self) -> np.ndarray | None:
        """
        return the numpy array of the detection transmit beam index.
        """
        return self[DETECTION_TX_BEAM][:]

    def read_detection_two_way_travel_time(self) -> np.ndarray:
        """
        return the numpy float64 array of the detection two way travel time in seconds.
        """
        return self[constants.BathymetryGrp.DETECTION_TWO_WAY_TRAVEL_TIME(BEAM_GROUP_NAME)][:].astype(np.float64)

    def read_detection_type(self) -> np.ndarray | None:
        """
        return the numpy array of the type of detection.
        """
        return self[DETECTION_TYPE][:]

    def read_multiping_sequence(self) -> np.ndarray | None:
        """
        return the numpy array of the multiping sequence identifier.
        """
        return self[MULTIPING_SEQUENCE][:]

    def read_multiping_center_frequency(self) -> np.ndarray | None:
        """
        return the numpy array of the center frequency in transmitted pulse.
        """
        return self[CENTER_FREQUENCY][:]

    def read_detection_ping_frequency(self) -> np.ndarray | None:
        """
        return the numpy array of the detection ping frequencies.
        """
        return self[DETECTION_PING_FREQUENCY][:]

    def get_rx_transducers(self) -> np.ndarray | None:
        """
        return array of receive transducer indices
        """
        transducers_func = self[constants.PlatformGrp.TRANSDUCER_FUNCTION()][:]
        rx_indices = np.where(transducers_func[:] == 0)
        return rx_indices[0]

    def get_version(self) -> float:
        """
        return xsf_convention_version cast as float
        """
        return float(self.dataset.xsf_convention_version)

    def get_processing_status(self) -> dict:
        """
        return current processing status as dict
        """
        try:
            processing_status_json = self.dataset.getncattr(ATT_PROCESSING_STATUS)
            processing_status = json.loads(processing_status_json)
        except Exception as e:  # json reading went wrong or empty, overwrite processing status
            processing_status = {}

        return processing_status

    def append_history_line(self, history_info: str):
        """
        Append one history line in history variable
        """
        provenance_grp = self[constants.ProvenanceGrp.get_group_path()]
        provenance_history = provenance_grp.getncattr(ATT_HISTORY)
        history = provenance_history if isinstance(provenance_history, list) else [provenance_history]
        timestamped_info = f"{datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')} {history_info}"
        history.append(timestamped_info)
        provenance_grp.setncattr(ATT_HISTORY, history)

    def update_processing_status(self, status_dict: dict[str, any]):
        """
        Update processing status attribute of root group
        """
        # try:
        #     processing_status_json = self.dataset.getncattr(ATT_PROCESSING_STATUS)
        #     processing_status = json.loads(processing_status_json)
        # except Exception as e:  # json reading went wrong or empty, overwrite processing status
        #     processing_status = {}
        processing_status = self.get_processing_status()
        for key, value in status_dict.items():
            processing_status[key] = value
        self.dataset.setncattr(ATT_PROCESSING_STATUS, json.dumps(processing_status))

    def get_bscorr(self) -> str | None:
        """
        return embedded bscorr string from .all generated XSF or None if empty
        """
        # bscorr from .all generated XSF is stored in extra_parameters with content_id 6
        if self.get_layer(constants.ExtraParametersGrp.CONTENT_IDENTIFIER()):
            BSCORR_CONTENTID = 6
            content_id = self[constants.ExtraParametersGrp.CONTENT_IDENTIFIER()][:]
            # find content_id with bscorr
            # content_id is a list of int
            if BSCORR_CONTENTID in content_id:
                bscorr_content = self[constants.ExtraParametersGrp.EXTRA_PARAM_INFORMATION()][
                    np.argwhere(content_id == BSCORR_CONTENTID)[0][0]
                ]
                # two first bytes are the length of the string
                # convert to string
                return bscorr_content[2:]
        # calibration file from .kmall generated XSF is stored in backscatter_calibration group
        elif self.get_layer(constants.BackscatterCalibrationGrp.FILE_CONTENT()):
            # assume there is only one calibration file and return it as string
            return self[constants.BackscatterCalibrationGrp.FILE_CONTENT()][0][:]
        return None


class BeamPositionIterator:
    def __init__(self, driver: XsfDriver, swath_count_by_iter: int, first_swath: int):
        self.driver = driver
        self.swath_count_by_iter = swath_count_by_iter
        self.swath = first_swath

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        # stop ?
        if self.swath >= self.driver.sounder_file.swath_count:
            raise StopIteration()

        last_swath = min(self.swath + self.swath_count_by_iter, self.driver.sounder_file.swath_count)
        result_lon = self.driver[DETECTION_LONGITUDE][self.swath : last_swath, :]
        result_lat = self.driver[DETECTION_LATITUDE][self.swath : last_swath, :]
        self.swath = last_swath
        return result_lon, result_lat


@contextmanager
def open_xsf(file_path: str, mode: str = "r") -> Generator[XsfDriver, None, None]:
    """
    Define a With Statement Context Managers for a XsfDriver
    Allow opening a XsfDriver in a With Statement
    """
    driver = XsfDriver(file_path)
    driver.open(mode)
    try:
        yield driver
    finally:
        driver.close()
