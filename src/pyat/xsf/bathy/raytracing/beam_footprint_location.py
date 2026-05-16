"""
Module providing functions to compute beams footprint location on focal plane,
using analytical intersection of hyperbolas and lines geometries, parallelized using numba.
"""

from typing import Tuple

import numba
import numpy as np

from pyat.utils.polynomial_utils import (
    quadratic_real_roots,
    solve_quadratic,
    solve_quartic,
)


@numba.njit
def get_line_direction(theta: float) -> np.ndarray:
    # theta: angle between line direction and x-axis (radians)
    return np.array([np.cos(theta), np.sin(theta)])


@numba.njit
def get_line_parametric_point(point: np.ndarray, direction: np.ndarray, t: float) -> np.ndarray:
    """Return one point at parameter t"""
    return point + t * direction


@numba.njit
def get_line_parametric_points(point: np.ndarray, direction: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return some points at parameters t"""
    # if t.ndim > 1:
    #     raise ValueError(f"point parameter must be one-dimensional")
    points_on_line = point[:, np.newaxis] + t[np.newaxis, :] * direction[:, np.newaxis]
    return points_on_line.T


@numba.njit
def hyperbola_c(a: float, b: float) -> np.ndarray:
    """
    Computes distance from center to foci, aka linear eccentricity.
    """
    return np.sqrt(a**2 + b**2)


@numba.njit
def hyperbola_f(a: float, b: float) -> np.ndarray:
    """
    Computes distance from center to hyperbola directrix
    """
    return a**2 / hyperbola_c(a, b)


@numba.njit
def hyperbola_foci(a: float, b: float, center: np.ndarray, theta: float, vertical: bool = True) -> np.ndarray:
    """
    Returns the coordinates of the foci of the hyperbola.
    theta: angle between hyperbola focal axis and x-axis (radians)
    vertical: if True, hyperbola focal axis is aligned with y-axis in local frame
    """
    c = hyperbola_c(a, b)
    # In local frame
    if vertical:
        f1_local = np.array([0, -c])
        f2_local = np.array([0, c])
    else:
        f1_local = np.array([-c, 0])
        f2_local = np.array([c, 0])
    R = hyperbola_rotation_matrix(theta)
    f1 = center + R @ f1_local
    f2 = center + R @ f2_local
    return np.stack((f1, f2))


@numba.njit
def hyperbola_rotation_matrix(theta: float) -> np.ndarray:
    """
    Returns the rotation matrix to rotate the hyperbola to the global frame.
    theta: angle between hyperbola focal axis and x-axis (radians)
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


@numba.njit
def point_in_local_frame(hy_center: np.ndarray, theta: float, hy_vertical: bool, point: np.ndarray) -> np.ndarray:
    """
    Returns point coordinates in local hyperbola frame : x-axis aligned with hyperbola focal axis, and centered @ hyperbola center
    theta: angle between hyperbola focal axis and x-axis (radians)
    hy_vertical: if True, hyperbola focal axis is aligned with y-axis in local frame
    """
    R = hyperbola_rotation_matrix(theta)
    p_local = R.T @ (point - hy_center)
    return p_local if not hy_vertical else p_local[::-1]


@numba.njit
def hyperbola_distance_to(
    hy_center: np.ndarray, hy_a: float, hy_b: float, hy_theta: float, hy_vertical: bool, point: np.ndarray
) -> np.ndarray:
    """
    Returns x offset between point and hyperbola, in hyperbola local frame.
    """
    x_local, y_local = point_in_local_frame(hy_center, hy_theta, hy_vertical, point)
    t = np.arcsinh(y_local / hy_b)
    x_check = hy_a * np.cosh(t)
    return np.abs(x_local - x_check)


@numba.njit
def get_rotation_matrix(first_point: np.ndarray, second_point: np.ndarray) -> np.ndarray:
    """
    Return the rotation matrix to rotate the vector AB (1st hyperbolas focii) to the x-axis.
    """
    cos_alpha = (second_point[0] - first_point[0]) / np.linalg.norm(second_point - first_point)
    sin_alpha = (second_point[1] - first_point[1]) / np.linalg.norm(second_point - first_point)
    # rotation matrix
    return np.array([[cos_alpha, sin_alpha], [-sin_alpha, cos_alpha]])


@numba.njit
def construct_transform_mtx(translation: np.ndarray, rotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a transformation matrix in 2D plane that applies translation and then rotation,
    and the back transform matrix
    """
    transform_mat = np.eye(3)
    transform_mat[:-1, :-1] = rotation
    transform_mat[:-1, -1] = rotation @ translation

    back_transform_mat = np.eye(3)
    back_transform_mat[:-1, :-1] = rotation.T
    back_transform_mat[:-1, -1] = -translation

    return transform_mat, back_transform_mat


@numba.njit
def apply_transform(transform_mtx: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Applies transformation defined by transform_mtx to a set of points
    """
    homog = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    transformed = (transform_mtx @ homog.T).T
    return transformed[:, :2]


@numba.njit
def hyperbola_domain_contains(
    hy_center: np.ndarray, a: float, b: float, theta: float, hy_vertical: bool, point: np.ndarray
) -> bool:
    """
    Returns True if the point is on the right hyperbola branch (or near it depending on the tolerance parameter).
    Default tolerance is 1e-2 (1 cm)
    theta: angle between hyperbola focal axis and x-axis (radians)
    hy_vertical: if True, hyperbola focal axis is aligned with y-axis in local frame
    """
    x_local, y_local = point_in_local_frame(hy_center, theta, hy_vertical, point)

    if (x_local / a) < 0:
        # x/a < 0 : point is not on the right hyperbola branch
        # print(f"x/a < 0 : {point} is not on the right hyperbola branch")
        return False
    if np.abs(x_local) < hyperbola_f(a, b):
        # point lies between origin and directrix, outside hyperbola domain
        return False
    # if (x_local / self.a) < tolerance:
    #     # "x < tolerance * a → point is not on hyperbola domain, tolerance < 1"
    #     # print(f"x/a < 1 → {point} is not on hyperbola domain")
    #     return False
    if np.abs(y_local) > np.abs(b / a * x_local):
        # point is outside asymptote domain
        return False
    return True


@numba.njit
def unique_points(points: np.ndarray) -> np.ndarray:
    """
    Returns unique points from point array (N,2).
    Numba compatible replacement for not supported numpy.unique()
    """
    n, m = points.shape
    result = np.empty((n, m))
    count = 0

    for i in range(n):
        is_unique = True
        for j in range(count):
            match = True
            for k in range(m):
                if points[i, k] != result[j, k]:
                    match = False
                    break
            if match:
                is_unique = False
                break
        if is_unique:
            for k in range(m):
                result[count, k] = points[i, k]
            count += 1

    return result[:count]


@numba.njit
def keep_points(points: np.ndarray, to_be_kept: np.ndarray) -> np.ndarray:
    """
    Keep points according boolean array to_be_kept.
    """
    n, m = points.shape
    result = np.empty((n, m), dtype=points.dtype)
    count = 0

    for i in range(n):
        if to_be_kept[i]:
            result[count, :] = points[i, :]
            count += 1

    return result[:count]


@numba.njit
def get_closest_point(
    xy, hy1_center, hy1_a, hy1_b, hy1_theta, hy1_vertical, hy2_center, hy2_a, hy2_b, hy2_theta, hy2_vertical
):
    n = xy.shape[0]
    dists = np.empty(n)

    for i in range(n):
        pnt = xy[i]
        d1 = hyperbola_distance_to(hy1_center, hy1_a, hy1_b, hy1_theta, hy1_vertical, pnt)
        d2 = hyperbola_distance_to(hy2_center, hy2_a, hy2_b, hy2_theta, hy2_vertical, pnt)
        dists[i] = np.sqrt(d1**2 + d2**2)

    return np.argmin(dists)


@numba.njit
def compute_lines_intersection(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray, tol: float = 1e-10
) -> np.ndarray:
    """
    Returns the intersection point with another Line2D, or None if parallel.
    """
    # Construct the system: t*d1 - s*d2 = (p2 - p1)
    # A = np.concatenate((d1, -d2))
    A = np.stack((d1, -d2), axis=1)
    b = p2 - p1
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if np.abs(det) < tol:
        # Parallel or coincident : no intersection
        return np.full(2, np.nan)
    # solve linear system
    t = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det

    return get_line_parametric_point(p1, d1, t)


@numba.njit
def compute_line_hyperbola_intersection(
    p1: np.ndarray, d1: np.ndarray, hy_center: np.ndarray, a: float, b: float, theta: float, hy_vertical: bool
) -> np.ndarray:
    """
    Compute intersection points between a line and an hyperbola branch.
    Returns one intersection point on the hyperbola right branch, or nans if no intersection.

    :param theta: angle between hyperbola focal axis and x-axis (radians)
    :param hy_vertical: if True, hyperbola focal axis is aligned with y-axis in local frame
    """
    # Transform to local hyperbola coordinates
    R = hyperbola_rotation_matrix(theta).T  # Inverse rotation
    p0_local = R @ (p1 - hy_center)
    d_local = R @ d1
    # manage vertical hyperbola
    x0, y0 = p0_local if not hy_vertical else p0_local[::-1]
    dx, dy = d_local if not hy_vertical else d_local[::-1]
    # Solve quadratic: A*t^2 + B*t + C = 0
    A = (dx**2) / a**2 - (dy**2) / b**2
    B = (2 * x0 * dx) / a**2 - (2 * y0 * dy) / b**2
    C = (x0**2) / a**2 - (y0**2) / b**2 - 1
    t = quadratic_real_roots(A.item(), B.item(), C.item())

    # if there is any real roots, returns intersection points
    if np.all(np.isfinite(t)):
        xy = get_line_parametric_points(p1, d1, t)
        for pnt in xy:
            # returns only right branch intersecting point
            if hyperbola_domain_contains(hy_center, a, b, theta, hy_vertical, pnt):
                return pnt
    return np.full(2, np.nan)


@numba.njit
def compute_hyperbolas_intersection(
    hy1_center: np.ndarray,
    hy1_a: float,
    hy1_b: float,
    hy1_theta: float,
    hy1_vertical: bool,
    hy2_center: np.ndarray,
    hy2_a: float,
    hy2_b: float,
    hy2_theta: float,
    hy2_vertical: bool,
) -> np.ndarray:
    """
    Returns the intersection points of two Hyperbolas, using analytical method.
    Solves system of hyperbolic equations using coordinate transformation and quadratic polynomial simplification,
    implementing algorithm from : J. Vesely and S. V. Doan, 2015, doi: 10.1109/RADIOELEK.2015.7129064
    Algorithm has been corrected to account for the case when hyperbolas are perpendicular and symmetric to the x-axis,
    which simplifies the quartic equation to a quadratic one : the paper proposed solution is not working in this case.
    if c=-e => U3=0 and if b=d => U4=0, so eq. 17 of the Vasely et al. paper simplifies to a quadratic equation

    :param hy1_center: hyperbola 1 center coordinates
    :param hy1_a: hyperbola 1 semi-major axis
    :param hy1_b: hyperbola 1 semi-minor axis
    :param hy1_theta: angle between hyperbola 1 focal axis and x-axis (radians)
    :param hy1_vertical: if True, hyperbola 1 focal axis is aligned with y-axis in local frame
    :param hy2_center: hyperbola 2 center coordinates
    :param hy2_a: hyperbola 2 semi-major axis
    :param hy2_b: hyperbola 2 semi-minor axis
    :param hy2_theta: angle between hyperbola 2 focal axis and x-axis (radians)
    :param hy2_vertical: if True, hyperbola 2 focal axis is aligned with y-axis in local frame
    :return: intersection points coordinates array (N,2), or array of nans if no intersection
    """
    # construct transformation that translates first hyperbola focus point to origin and then rotate an angle α
    # about the coordinate origin, where α is the angle between first hyperbola focal axis and x-axis.
    hy1_foci = hyperbola_foci(hy1_a, hy1_b, hy1_center, hy1_theta, hy1_vertical)
    hy2_foci = hyperbola_foci(hy2_a, hy2_b, hy2_center, hy2_theta, hy2_vertical)

    transform_mtx, back_transform_mtx = construct_transform_mtx(
        translation=-hy1_foci[0], rotation=get_rotation_matrix(hy1_foci[0], hy1_foci[1])
    )
    # Transform every foci to new places
    foci_t = apply_transform(transform_mtx, points=np.concatenate((hy1_foci, hy2_foci)))

    # extract parameters
    a = foci_t[1, 0]
    b = foci_t[2, 0]
    c = foci_t[2, 1]
    d = foci_t[3, 0]
    e = foci_t[3, 1]
    xy = np.concatenate(
        (
            solve_quadratic(a, b, c, 2 * hy1_a, 2 * hy2_a),
            solve_quadratic(a, b, c, -2 * hy1_a, 2 * hy2_a),
            solve_quartic(a, b, c, d, e, 2 * hy1_a, 2 * hy2_a),
            solve_quartic(a, b, c, d, e, -2 * hy1_a, 2 * hy2_a),
        )
    )
    # Remove NaN points - Numba-compatible approach
    valid_rows = np.zeros(len(xy), dtype=np.bool_)
    for i in range(len(xy)):
        valid_rows[i] = not (np.isnan(xy[i, 0]) or np.isnan(xy[i, 1]))
    xy = xy[valid_rows]
    # early return if no intersection found
    if xy.size == 0:
        return np.full(2, np.nan)
    # transform back coordinates to the original coordinate system
    xy = apply_transform(back_transform_mtx, points=xy)
    # remove possible duplicates (common solution of the two systems)
    xy = unique_points(xy)
    # keep only intersection points that are on right branche of each hyperbola
    within_both_domain = np.array(
        [
            hyperbola_domain_contains(hy1_center, hy1_a, hy1_b, hy1_theta, hy1_vertical, pnt)
            and hyperbola_domain_contains(hy2_center, hy2_a, hy2_b, hy2_theta, hy2_vertical, pnt)
            for pnt in xy
        ]
    )
    xy = keep_points(xy, within_both_domain)

    if xy.size == 0:
        # intersection point is not on the right hyperbola branch -> no intersection
        return np.full(2, np.nan)
    else:
        # otherwise retain closest point to each hyperbola
        idx = get_closest_point(
            xy, hy1_center, hy1_a, hy1_b, hy1_theta, hy1_vertical, hy2_center, hy2_a, hy2_b, hy2_theta, hy2_vertical
        )
        return xy[idx]


@numba.njit
def intersect_tx_rx_beams_analytical_nb(r, m, tx_steer, rx_steer, array_separation: np.ndarray) -> np.ndarray:
    """
    Intersects Tx and Rx beams, and find an analytical solution, based on hyperbolas intersection on a focal plane
    r: distance from the Tx to the focal plane (m)
    m: Rx arrays orientations relative to Tx, expressed in ARF (radians)
    tx_steer: steering angle of the Tx beam (radians)
    rx_steer: steering angle of the Rx beam (radians)
    array_separation: (x, y, z) separation of the Rx array from the Tx beam center, expressed in ARF (m)
    line_tol: tolerance on asymptote for considering that an hyperbola is effectively a line (default 1e-2)
    """
    # init Tx beam
    a_tx = r * np.tan(tx_steer)
    b_tx = r
    # init Rx beam
    a_rx = (r - array_separation[2]) * np.tan(-rx_steer)
    b_rx = r - array_separation[2]

    if a_tx == 0 and a_rx == 0:  # or np.abs(a_tx / b_tx) < line_tol and np.abs(a_rx / b_rx) < line_tol:
        # Tx  and Rx footprint are lines
        p1, d1 = np.array([a_tx, 0]), get_line_direction(np.pi / 2)
        p2, d2 = np.array(
            [array_separation[0] + a_rx * np.sin(m), array_separation[1] + a_rx * np.cos(m)]
        ), get_line_direction(m)
        xy = compute_lines_intersection(p1, d1, p2, d2)
    elif a_tx == 0:
        # Tx footprint is a line and Rx footprint is an hyperbola
        xy = compute_line_hyperbola_intersection(
            p1=np.array([a_tx, 0]),
            d1=get_line_direction(np.pi / 2),
            hy_center=np.array([array_separation[0], array_separation[1]]),
            a=a_rx,
            b=b_rx,
            theta=m,
            hy_vertical=True,
        )
    elif a_rx == 0:
        # Tx footprint is an hyperbola and Rx footprint is a line
        xy = compute_line_hyperbola_intersection(
            p1=np.array([array_separation[0] + a_rx * np.sin(m), array_separation[1] + a_rx * np.cos(m)]),
            d1=get_line_direction(m),
            hy_center=np.array([0.0, 0.0]),
            a=a_tx,
            b=b_tx,
            theta=0.0,
            hy_vertical=False,
        )
    else:
        # Tx  and Rx footprint are hyperbolas
        xy = compute_hyperbolas_intersection(
            hy1_center=np.array([0.0, 0.0]),
            hy1_a=a_tx,
            hy1_b=b_tx,
            hy1_theta=0.0,
            hy1_vertical=False,
            hy2_center=np.array([array_separation[0], array_separation[1]]),
            hy2_a=a_rx,
            hy2_b=b_rx,
            hy2_theta=m,
            hy2_vertical=True,
        )
    return xy


@numba.njit
def intersect_tx_rx_beams_triangles(tl, rl, m, tx_steer, rx_steer, array_separation):
    """
    Intersects Tx and Rx beams, and find an analytical solution, based on projected triangle intersection on a focal plane
    r: distance from the Tx to the focal plane (m)
    m: angle of the focal plane (radians)
    tx_steer: steering angle of the Tx beam (radians)
    rx_steer: steering angle of the Rx beam (radians)
    array_separation: (x, y, z) separation of the Rx array from the Tx beam center (m)
    """
    y1 = rl * np.sin(-rx_steer) / np.cos(m)
    y2 = tl * np.sin(tx_steer) * np.tan(m)
    y3 = -array_separation[0] * np.tan(m)
    y4 = array_separation[1]

    x_ = tl * np.sin(tx_steer)
    y_ = y1 + y2 + y3 + y4

    return np.array([x_, y_, np.sqrt(tl**2 - x_**2 - y_**2)])


@numba.njit
def init_slant_ranges(r, array_separation):
    """
    Retrieves initial slant ranges in ARF from Tx and Rx, accounting for Tx and Rx array separation.
    Used in triangle intersection method NCTA.
    """
    tl = np.linalg.norm(r)
    p = np.linalg.norm(array_separation)
    rl = np.sqrt(tl**2 + p**2 - 2 * r.dot(array_separation))

    return tl, rl
