import math
from math import atan2, sqrt

import numpy as np
from scipy.spatial.transform import Rotation as R


def get_roll_matrix(roll: float) -> np.ndarray:
    """
    Return the roll transformation matrix
    """
    return R.from_euler("X", roll, degrees=False).as_matrix()


def get_pitch_matrix(pitch: float) -> np.ndarray:
    """
    Return the pitch transformation matrix
    """
    return R.from_euler("Y", pitch, degrees=False).as_matrix()


def get_vcs_attitude(pitch: float, roll: float) -> np.ndarray:
    """
    Return the attitude matrix in Vessel Coordinate System
    """
    return R.from_euler("YX", [pitch, roll], degrees=False).as_matrix()


def transform_vcs_to_scs(pitch: float, roll: float, x_vcs: np.ndarray) -> np.ndarray:
    """
    Transform coordinates from the Vessel Coordinate System to Surface Coordinate System
    """
    return R.from_euler("YX", [pitch, roll], degrees=False).apply(x_vcs)


def rot_vcs_to_fcs(headings: np.ndarray, pitches: np.ndarray, rolls: np.ndarray) -> R:
    """
    VESSEL -> FIXED rotations.
    Returns VCS to FCS rotations given platform attitude time series (heading, pitch, roll) in degrees.
    Uses Tait-Bryan convention (ZYX), and replaces invalid attitude values with 0 to avoid creation of invalid rotations.
    """
    headings = np.nan_to_num(headings, copy=True)
    pitches = np.nan_to_num(pitches, copy=True)
    rolls = np.nan_to_num(rolls, copy=True)
    return R.from_euler("ZYX", np.column_stack([headings, pitches, rolls]), degrees=True)


def rot_vcs_to_scs(pitches: np.ndarray, rolls: np.ndarray) -> R:
    """
    VESSEL -> SURFACE rotations.
    Returns VCS to SCS rotations given platform attitude (pitch, roll) in degrees.
    Uses Tait-Bryan convention (YX), and replaces invalid attitude values with 0 to avoid creation of invalid rotations.
    """
    pitches = np.nan_to_num(pitches, copy=True)
    rolls = np.nan_to_num(rolls, copy=True)
    return R.from_euler("YX", np.column_stack([pitches, rolls]), degrees=True)


def rot_scs_to_fcs(headings: np.ndarray) -> R:
    """
    SURFACE -> FIXED rotations.
    Returns SCS to FCS rotations given platform heading in degrees.
    Replaces invalid attitude values with 0 to avoid creation of invalid rotations.
    Origin of the FCS is fixed somewhere in the nominal sea surface. Be aware that the FCS is defined according to the right hand rule:
        x-axis pointing north,
        y-axis pointing east,
        z-axis pointing down along the g-vector.
    """
    headings = np.nan_to_num(headings, copy=True)
    return R.from_euler("Z", headings, degrees=True)


def correct_mru_yaw_misalignment(
    headings: np.ndarray, pitches: np.ndarray, rolls: np.ndarray, yaw_delta: float
) -> np.ndarray:
    """
    Correct MRU measured angles from MRU misalignment yaw_delta
    returns corrected headings, pitches and rolls as [n, 3] np.ndarray
    """
    # MRU misalignment rotation (intrinsic, about MRU z-axe)
    mru_misalign_rot = R.from_euler("z", yaw_delta, degrees=True)
    mru_recorded_rots = rot_vcs_to_fcs(headings=headings, pitches=pitches, rolls=rolls)
    # Correction : Globale_Rotation * Local_Inverse_Correction
    mru_true_rots = mru_recorded_rots * mru_misalign_rot.inv()

    return mru_true_rots.as_euler("ZYX", degrees=True)


def to_euler_angles(rotation: R):
    """
    Retrieve Tait-Bryan angles from rotation
    Ref: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_(in_3-2-1_sequence)_conversion
    Equivalent to rotation.as_euler("ZYX")
    """

    # transform as quaternion
    q = rotation.as_quat()

    qx, qy, qz, qw = q
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = sqrt(1 + 2 * (qw * qy - qx * qz))
    cosp = sqrt(1 - 2 * (qw * qy - qx * qz))
    pitch = 2 * atan2(sinp, cosp) - math.pi / 2

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = atan2(siny_cosp, cosy_cosp)

    return [roll, pitch, yaw]


def angle_about_axis_in_new_frame(angles, axis_A, R_A2B):
    """
    Retrieve angles about an axis (axis_A) in frame A in frame B, given rotations from frame A to B
    """
    axis_A = axis_A / np.linalg.norm(axis_A)

    # R_axis_A = R.from_rotvec(np.deg2rad(angle) * axis_A)
    R_axis_A = R.from_rotvec([angle * axis_A for angle in np.deg2rad(angles)])
    axis_B = R_A2B.inv().apply(axis_A)

    R_axis_B = R_A2B.inv() * R_axis_A * R_A2B
    rotvec_B = R_axis_B.as_rotvec()

    return np.rad2deg(np.vecdot(rotvec_B, axis_B))


def angle_in_new_frame(theta_axis_A_deg, R_A2B, axis="X", degrees=True):
    """
    Retrieve angle about axis in frame A in frame B, given rotations from frame A to B
    """
    axis_idx = {
        "X": 0,
        "Y": 1,
        "Z": 2,
    }
    if axis not in axis_idx:
        raise ValueError(f"Invalid axis '{axis}'. Must be one of {list(axis_idx.keys())}.")

    R_A = R.from_euler(axis, theta_axis_A_deg, degrees=degrees)
    R_B = R_A2B.inv() * R_A * R_A2B
    return np.rad2deg(R_B.as_rotvec()[:, axis_idx[axis]]) if degrees else R_B.as_rotvec()[:, axis_idx[axis]]


if __name__ == "__main__":
    print(transform_vcs_to_scs(0.63, 1.61, [4.223, 0.01, 1.735]))
