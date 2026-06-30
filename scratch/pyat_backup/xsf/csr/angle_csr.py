# -*- coding: utf-8 -*-
"""
This module handle classes and definitions to handle changes of coordinates systems for angles of beams
Those classes are given as facilities, for real position and computation a vectoriel approach should be preferred
to switch from one csr to another one
"""

from enum import Enum

import netCDF4 as nc
import numpy as np
from scipy.interpolate import interp1d
from sonar_netcdf.sonar_groups import (
    AttitudeSubGroup,
    BathymetryGrp,
    BeamGroup1Grp,
    BeamGroup1VendorSpecificGrp,
    PlatformGrp,
)

from pyat.utils.time_utils import floatsecond_tonano_array


class CsrAngle(Enum):
    SACS = 0
    # Surface Angle Coordinate System, angles are expressed in a
    # coordinate system having its origin positioned at the center of the transducer face.
    # Axis of the crs are parallel with the axis of the SCS coordinate system (ie Z ig parallel with the g vector)
    # Angles have the same sign as angles of pitch, roll, and heave.
    # Roll is positive with port side up, pitch is positive with bow up, and heading/yaw is positive clockwise

    TACS = 1
    # Transducer Angle Coordinate System, angles are expressed in a
    # coordinate system having its origin positioned at the center of the transducer face.
    # Axis of the crs is perpendicular with the face of the transducer
    # (ie axis matches the vessel coordinate system with the installation rotations and offsets applied
    # Angles have the same sign as angles of pitch, roll, and heave.
    # Roll is positive with port side up, pitch is positive with bow up, and heading/yaw is positive clockwise


def rx_beam_angle_to_surface():
    """Convert RxAngle from a Transducer coordinate system to a FACS csr"""
    # xdir = RootGrp.SonarGrp.BeamGroup1Grp.BEAM_DIRECTION_X
    # ydir = RootGrp.SonarGrp.BeamGroup1Grp.BEAM_DIRECTION_Y
    # zdir = RootGrp.SonarGrp.BeamGroup1Grp.BEAM_DIRECTION_Z

    raise NotImplementedError()


def detection_pointing_angle_to_surface_crs(file: nc.Dataset):
    """
    convert bathymetry detection beam pointing angle expected to be referred in transducer csr to
    surface referential csr this include at the time of
    """
    detection_transducer_index = np.array(file[BathymetryGrp.DETECTION_RX_TRANSDUCER_INDEX()])
    transducer_roll_offset = np.array(file[PlatformGrp.TRANSDUCER_ROTATION_X()])
    beam_pointing_angle_transducer = np.array(file[BathymetryGrp.DETECTION_BEAM_POINTING_ANGLE()])

    # switch to 2D array for indexing purpose
    transducer_roll_offset = transducer_roll_offset.reshape(1, -1)
    # create a fake ping index
    ping_number = detection_transducer_index.shape[0]
    detection_ping_index = np.zeros((ping_number, 1), dtype=np.int32)

    # we create an installation rotation offset matrix per ping,detection, each detection being indexed by its ping number
    roll_offset_ping_detection = transducer_roll_offset[detection_ping_index, detection_transducer_index]

    # roll is interpolated at the ping time it shall be interpolated at the detection time
    # compute detection time
    pingtime = np.array(file[BeamGroup1Grp.PING_TIME()])

    # detection two way travel time
    detection_2WTT = np.array(file[BathymetryGrp.DETECTION_TWO_WAY_TRAVEL_TIME()])
    detection_2WTT = floatsecond_tonano_array(detection_2WTT)
    # retrieve the transmit time delay for each tx (per tx index)
    tx_delay = np.array(file[BeamGroup1VendorSpecificGrp.TRANSMIT_TIME_DELAY()])
    tx_delay = floatsecond_tonano_array(tx_delay)
    # we create an detection transmit delay matrix per ping,detection, each detection being indexed by its ping number
    detection_delay = tx_delay[detection_ping_index, detection_transducer_index]

    detection_delay = detection_delay + detection_2WTT
    detection_time = detection_delay + pingtime.reshape(-1, 1)
    # readeable_time_values=nano_todatetime(detection_time)
    # now interpolate roll for each detection time
    # retrieve navigation roll and roll time (high frequency)
    roll = file[AttitudeSubGroup.ROLL("000")]
    time_roll = file[AttitudeSubGroup.TIME("000")]
    # readeable_time_roll_values= nano_todatetime(np.array(time_roll))

    f = interp1d(time_roll, roll, fill_value="extrapolate", kind="linear")
    if np.ma.is_masked(detection_time):
        detection_time = detection_time.filled(np.nan)
    detection_roll = f(detection_time)  # cannot apply function to masked array, so fill masked values with nan

    return beam_pointing_angle_transducer - roll_offset_ping_detection - detection_roll
