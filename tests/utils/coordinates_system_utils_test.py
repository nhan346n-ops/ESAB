import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import pyat.utils.coordinates_system_utils as cs_utils


@pytest.fixture
def roll():
    return 0.63


@pytest.fixture
def pitch():
    return 1.41


@pytest.fixture
def x_vcs():
    return [4.223, 0.01, 1.735]


def test_roll_matrix(roll: float):
    """
    Return the roll transformation matrix
    """
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    roll_mat = np.array([[1, 0, 0], [0, cos_roll, -sin_roll], [0, sin_roll, cos_roll]])
    assert np.allclose(cs_utils.get_roll_matrix(roll), roll_mat)


def test_pitch_matrix(pitch: float):
    """
    Return the pitch transformation matrix
    """
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    pitch_mat = np.array([[cos_pitch, 0, sin_pitch], [0, 1, 0], [-sin_pitch, 0, cos_pitch]])
    assert np.allclose(cs_utils.get_pitch_matrix(pitch), pitch_mat)


def test_vcs_attitude(pitch: float, roll: float) -> np.ndarray:
    """
    Return the attitude matrix in Vessel Coordinate System
    """
    vcs_att = np.matmul(cs_utils.get_pitch_matrix(pitch), cs_utils.get_roll_matrix(roll))
    assert np.allclose(cs_utils.get_vcs_attitude(pitch=pitch, roll=roll), vcs_att)
    assert np.allclose(cs_utils.to_euler_angles(R.from_matrix(vcs_att)), [roll, pitch, 0])
    assert np.allclose(R.from_matrix(vcs_att).as_euler("ZYX"), [0, pitch, roll])


def test_transform_vcs_to_scs(pitch: float, roll: float, x_vcs: np.ndarray) -> np.ndarray:
    """
    Transform coordinates from the Vessel Coordinate System to Surface Coordinate System
    """
    x_out = np.matmul(cs_utils.get_vcs_attitude(pitch=pitch, roll=roll), x_vcs)
    assert np.allclose(cs_utils.transform_vcs_to_scs(pitch=pitch, roll=roll, x_vcs=x_vcs), x_out)


def test_correct_mru_misalignment(pitch: float, roll: float):
    """
    MRU is misaligned by 90° :
    - measured roll = - true pitch
    - measured pitch = true roll
    """
    assert np.allclose(
        cs_utils.correct_mru_yaw_misalignment([0.0], [pitch], [roll], yaw_delta=90.0), [-90.0, roll, -pitch], atol=0.1
    )
