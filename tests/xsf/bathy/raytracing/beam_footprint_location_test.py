import math

import numpy as np
import numpy.testing as npt

from pyat.xsf.bathy.raytracing import beam_footprint_location as bfl


def _call(fn, *args, **kwargs):
    """Call numba njit-wrapped function using its python implementation if available."""
    f = getattr(fn, "py_func", fn)
    return f(*args, **kwargs)


def test_line_direction_and_parametric():
    v = _call(bfl.get_line_direction, 0.0)
    npt.assert_allclose(v, np.array([1.0, 0.0]))

    v = _call(bfl.get_line_direction, math.pi / 2)
    npt.assert_allclose(v, np.array([0.0, 1.0]), atol=1e-12)

    p = np.array([1.0, 2.0])
    d = np.array([1.0, 0.0])
    q = _call(bfl.get_line_parametric_point, p, d, 3.0)
    npt.assert_allclose(q, np.array([4.0, 2.0]))

    ts = np.array([0.0, 1.0, 2.0])
    pts = _call(bfl.get_line_parametric_points, p, d, ts)
    expected = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
    npt.assert_allclose(pts, expected)


def test_hyperbola_helpers_and_distance():
    a, b = 3.0, 4.0
    c = _call(bfl.hyperbola_c, a, b)
    assert pytest_rel(c, 5.0)

    f = _call(bfl.hyperbola_f, a, b)
    assert pytest_rel(f, a * a / c)

    R = _call(bfl.hyperbola_rotation_matrix, 0.0)
    npt.assert_allclose(R, np.eye(2))

    foci = _call(bfl.hyperbola_foci, a, b, np.array([0.0, 0.0]), 0.0, True)
    # vertical True => foci on y axis at ±c
    npt.assert_allclose(foci[0], np.array([0.0, -c]))
    npt.assert_allclose(foci[1], np.array([0.0, c]))

    # test point_in_local_frame with vertical swap
    p = np.array([1.0, 6.0])
    center = np.array([1.0, 1.0])
    local = _call(bfl.point_in_local_frame, center, 0.0, True, p)
    # R = I, p-center = [0,5], vertical True -> reversed -> [5,0]
    npt.assert_allclose(local, np.array([5.0, 0.0]))

    # construct a point exactly on the hyperbola using parameter t
    t = 0.5
    y_local = b * np.sinh(t)
    x_local = a * np.cosh(t)
    point = np.array([x_local, y_local])

    d = _call(bfl.hyperbola_distance_to, np.array([0.0, 0.0]), a, b, 0.0, False, point)
    # thus,distance should be zero
    assert pytest_rel(d, 0.0, atol=1e-12)


def pytest_rel(value, expected, atol=1e-12):
    return abs(value - expected) <= atol


def test_rotation_and_transform():
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    R = _call(bfl.get_rotation_matrix, A, B)
    npt.assert_allclose(R, np.eye(2))

    # translation and rotation
    t = np.array([1.0, 2.0])
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])  # 90 deg
    T, Tback = _call(bfl.construct_transform_mtx, t, rot)

    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    applied = _call(bfl.apply_transform, T, pts)
    # apply: rotate and then translate
    rotated = (rot @ pts.T).T + rot @ t
    npt.assert_allclose(applied, rotated)

    # back transform should recover original up to numerical tolerance
    recovered = _call(bfl.apply_transform, Tback, applied)
    npt.assert_allclose(recovered, pts, atol=1e-12)


def test_unique_keep_and_closest():
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    uniq = _call(bfl.unique_points, pts)
    # order preserved, expect two unique rows
    npt.assert_allclose(uniq, np.array([[0.0, 0.0], [1.0, 1.0]]))

    mask = np.array([True, False, True])
    kept = _call(bfl.keep_points, pts, mask)
    npt.assert_allclose(kept, np.array([[0.0, 0.0], [0.0, 0.0]]))

    # get_closest_point: simple test where first point is closer
    xy = np.array([[0.0, 0.0], [10.0, 10.0]])
    idx = _call(
        bfl.get_closest_point,
        xy,
        np.array([0.0, 0.0]),
        1.0,
        1.0,
        0.0,
        False,
        np.array([100.0, 100.0]),
        1.0,
        1.0,
        0.0,
        False,
    )
    assert idx == 1  # 0


def test_lines_intersection():
    # intersecting lines
    p1, d1 = np.array([0.0, 0.0]), np.array([0.0, 1.0])  # vertical line x=0
    p2, d2 = np.array([1.0, 2.0]), np.array([1.0, 0.0])  # horizontal line y=2
    xy = _call(bfl.compute_lines_intersection, p1, d1, p2, d2)
    npt.assert_allclose(xy, np.array([0.0, 2.0]))


def test_lines_no_intersection():
    # parallel lines : non-intersecting (should return NaN)
    p3, d3 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    p4, d4 = np.array([0.0, 1.0]), np.array([1.0, 0.0])
    xy2 = _call(bfl.compute_lines_intersection, p3, d3, p4, d4)
    assert np.all(np.isnan(xy2))


def test_line_hyperbola_intersection():
    # line-hyperbola: horizontal line y=0 intersects right branch at x=a
    p, d = np.array([0.0, 0.0]), np.array([1.0, 0.0])  # horizontal line y=0
    pt = _call(bfl.compute_line_hyperbola_intersection, p, d, np.array([0.0, 0.0]), 3.0, 4.0, 0.0, False)
    npt.assert_allclose(pt, np.array([3.0, 0.0]), atol=1e-12)


def test_line_hyperbola_no_intersection():
    # line-hyperbola: vertical line x=0 non-intersecting (should return NaN)
    p, d = np.array([0.0, 0.0]), np.array([0.0, 1.0])  # vertical line x=0
    pt = _call(bfl.compute_line_hyperbola_intersection, p, d, np.array([0.0, 0.0]), 3.0, 4.0, 0.0, False)
    assert np.all(np.isnan(pt))


def test_hyperbolas_no_intersection():
    # two non-intersecting hyperbolas : check that they are indeed not intersecting (should return NaN).
    pt = _call(
        bfl.compute_hyperbolas_intersection,
        np.array([0.0, 0.0]),
        3.0,
        1.0,
        0.0,
        False,
        np.array([0.0, 0.0]),
        3.0,
        1.0,
        0.0,
        True,
    )
    assert np.all(np.isnan(pt))


def test_perpendicular_hyperbolas_intersection():
    # Two identic perpendicular hyperbolas that should intersect on [1.154700538379252, 1.154700538379251]
    # quadratic algorithm
    pt = _call(
        bfl.compute_hyperbolas_intersection,
        np.array([0.0, 0.0]),
        1.0,
        2.0,
        0.0,
        False,
        np.array([0.0, 0.0]),
        1.0,
        2.0,
        0.0,
        True,
    )
    npt.assert_allclose(pt, np.array([1.154700538379252, 1.154700538379251]), atol=1e-12)


def test_general_hyperbolas_intersection():
    # Two hyperbolas that should intersect on [1.3166421172575253, 3.302937286342112]
    # quartic algorithm
    pt = _call(
        bfl.compute_hyperbolas_intersection,
        np.array([2.0, -2.0]),
        1.0,
        5.0,
        0.40,
        False,
        np.array([3.5, -1.5]),
        2.0,
        1.0,
        0.0,
        True,
    )
    npt.assert_allclose(pt, np.array([1.3166421172575253, 3.302937286342112]), atol=1e-12)
