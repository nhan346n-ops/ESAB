"""
Module providing utility classes and functions for lines and hyperbolas intersections,
including methods for plotting.
"""

from typing import Union

import numpy as np
from bokeh.io import show
from bokeh.plotting import figure
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


class Transform:
    def __init__(self, rotation=None, translation=None, apply_rotation_first=True):
        if rotation is None and translation is None:
            self.dim = 3
            self.matrix = np.eye(4)
        elif rotation is not None and translation is not None:
            self.dim = rotation.shape[0]
            self.matrix = np.eye(self.dim + 1)
            self.matrix[: self.dim, : self.dim] = rotation

            if apply_rotation_first:
                self.matrix[: self.dim, self.dim] = translation
            else:
                self.matrix[: self.dim, self.dim] = rotation @ translation
        else:
            raise ValueError("Provide both rotation and translation, or neither.")

        self.apply_rotation_first = apply_rotation_first

    @classmethod
    def from_euler(cls, angles, degrees=True, translation=None, dim=3, apply_rotation_first=True):
        if dim == 3:
            r = Rotation.from_euler("xyz", angles, degrees=degrees)
            rotation = r.as_matrix()
            if translation is None:
                translation = np.zeros(3)
        elif dim == 2:
            angle = np.deg2rad(angles[0]) if degrees else angles[0]
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            if translation is None:
                translation = np.zeros(2)
        else:
            raise ValueError("dim must be 2 or 3")

        return cls(rotation, translation, apply_rotation_first=apply_rotation_first)

    def inverse(self):
        R_inv = self.matrix[: self.dim, : self.dim].T
        if self.apply_rotation_first:
            t = self.matrix[: self.dim, self.dim]
            t_inv = -R_inv @ t
        else:
            # In this case, matrix[:dim, dim] = R @ t originally
            # So inverse translation is just -t
            t_inv = -self.matrix[: self.dim, self.dim]
        return Transform(R_inv, t_inv, apply_rotation_first=self.apply_rotation_first)

    def apply(self, points):
        points = np.atleast_2d(points)
        homog = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed = (self.matrix @ homog.T).T
        return transformed[:, : self.dim] if len(points) > 1 else transformed[0, : self.dim]

    def compose(self, other):
        if self.dim != other.dim:
            raise ValueError("Cannot compose transforms of different dimensions.")
        composed_matrix = self.matrix @ other.matrix
        # By default, keep apply_rotation_first from self
        composed_rotation = composed_matrix[: self.dim, : self.dim]
        composed_translation = composed_matrix[: self.dim, self.dim]
        return Transform(composed_rotation, composed_translation, apply_rotation_first=self.apply_rotation_first)

    def __call__(self, points):
        return self.apply(points)

    def __mul__(self, other):
        return self.compose(other)

    def __repr__(self):
        return f"Transform(dim={self.dim}, apply_rotation_first={self.apply_rotation_first}):\n{self.matrix}"


class Line2D:
    def __init__(self, point, theta):
        """
        point: a point on the line (array-like)
        theta: orientation (radians)
        """
        self.point = np.array(point, dtype=float)
        self._theta = None
        self._direction = None
        self.theta = theta  # Use the setter to initialize direction

    @property
    def theta(self):
        """Angle of the line (in radians) from x-axis."""
        return self._theta

    @theta.setter
    def theta(self, angle_rad):
        """Set angle and update direction vector accordingly."""
        self._theta = float(angle_rad)
        self._direction = np.array([np.cos(self._theta), np.sin(self._theta)], dtype=float)

    @property
    def direction(self):
        """Unit direction vector of the line (derived from theta)."""
        return self._direction

    # def __init__(self, point, direction):

    #     self.point = np.array(point, dtype=float)
    #     self.direction = np.array(direction, dtype=float)

    def parametric_point(self, t):
        """Return points at parameters t"""
        t = np.atleast_1d(t)
        if t.ndim > 1:
            raise ValueError(f"point parameter must be one-dimensional")
        # return self.point + t[:, np.newaxis] * self.direction
        return self.point[:, np.newaxis] + t[np.newaxis, :] * self.direction[:, np.newaxis]

    def parameter_from_point(self, pt):
        """Return the parameter t such that pt = p0 + t*d"""
        pt = np.array(pt)
        v = pt - self.point
        return np.dot(v, self.direction)

    def project_point(self, pt):
        """Project an arbitrary point onto the line"""
        t = self.parameter_from_point(pt)
        return self.parametric_point(t)

    def distance_to_point(self, pt):
        """Return the perpendicular distance from a point to the line"""
        pt = np.array(pt)
        proj = self.project_point(pt)
        return np.linalg.norm(pt - proj)

    def contains(self, pt, tol=1e-8):
        """Check whether a point lies on the line within tolerance"""
        dist = self.distance_to_point(pt)
        return dist < tol

    def intersect(self, other: Union["Line2D", "Hyperbola"], tol=1e-10):
        """
        Returns the intersection point with another Line2D, or None if parallel.
        """
        if not isinstance(other, Line2D) and not isinstance(other, Hyperbola):
            raise TypeError("'other' must be a Line2d or an Hyperbola object")

        if isinstance(other, Hyperbola):
            # If other is an Hyperbola, silently use specific intersection method
            return self.__intersect_with_hyperbola(other)

        p1, d1 = self.point, self.direction
        p2, d2 = other.point, other.direction

        # Construct the system: t*d1 - s*d2 = (p2 - p1)
        A = np.column_stack((d1, -d2))
        b = p2 - p1

        if np.abs(np.linalg.det(A)) < tol:
            return None  # Parallel or coincident (check separately if needed)

        ts = np.linalg.solve(A, b)
        t = ts[0]
        xy = self.parametric_point(t).T

        return xy[0]

    def __intersect_with_hyperbola(self, hyperbola: "Hyperbola"):
        """
        Compute intersection points between the hyperbola and a 2D line.
        Returns a list of 0, 1, or 2 intersection points.
        """
        # Transform to local hyperbola coordinates
        R = hyperbola.rotation_matrix().T  # Inverse rotation
        p0_local = R @ (self.point - hyperbola.center)
        d_local = R @ self.direction
        a, b = hyperbola.a, hyperbola.b

        # manage vertical hyperbola
        x0, y0 = p0_local if not hyperbola.vertical else p0_local[::-1]
        dx, dy = d_local if not hyperbola.vertical else d_local[::-1]

        # Solve quadratic: A*t^2 + B*t + C = 0
        A = (dx**2) / a**2 - (dy**2) / b**2
        B = (2 * x0 * dx) / a**2 - (2 * y0 * dy) / b**2
        C = (x0**2) / a**2 - (y0**2) / b**2 - 1

        t = np.roots([A, B, C])

        # if there is any real positive roots, returns intersection points
        if np.any(np.isreal(t)):
            t = np.real(t[np.isreal(t)])
            xy = self.parametric_point(t)
            for pnt in xy.T:
                # returns only one point
                if hyperbola.domain_contains(pnt):
                    return pnt
        return None

    def plot(
        self,
        ax,
        t_max: float = 1000,
        color: str = "blue",
        render: str = "matplotlib",
        swapxy: bool = False,
        label=None,
    ):
        """
        Plots the 2D line on a given axis. Render can be "matplotlib" or "bokeh".
        optionally set swapxy=True to swap x and y-axis for a top ARF view :  x=along, y=across, z=down.
        """
        xy = self.parametric_point((-t_max, t_max))
        center = self.point
        # legend_label = self.__repr__() if label is None else label
        legend_label = self if label is None else label
        if swapxy:
            xy = xy[::-1]
            center = self.point[::-1]
        if render == "matplotlib":
            ax.plot(*xy, label=legend_label, color=color)
            ax.scatter(*center, marker=".", color=color, label="center")
        elif render == "bokeh":
            ax.line(x=xy[0], y=xy[1], line_width=1, color=color, legend_label=legend_label)
            ax.scatter(x=center[0], y=center[1], marker="o", color=color, legend_label="center")

    def __repr__(self):
        return f"Line2D: point={tuple(self.point)}, direction={tuple(self.direction)}"
        # return f"Line2D: point={tuple(self.point)}, theta={self.theta}"


class Hyperbola:
    def __init__(self, a, b, theta=0.0, center=(0.0, 0.0), vertical=False):
        """
        a, b — major and minor semi-axes (must be non-zero)
        theta — rotation of hyperbola in radians from the x-axis
        center — (x, y) center of the hyperbola
        vertical — if True, transverse axis is vertical (y-axis), otherwise horizontal (x-axis)
        """
        if a == 0 or b == 0:
            # don't allow degenerate hyperbola
            raise ValueError("a and b must be non-zero")
        self.a = a
        self.b = b
        self.theta = theta
        self.vertical = vertical
        self.center = np.array(center)

    def c(self):
        """
        Computes distance from center to foci, aka linear eccentricity.
        """
        return np.sqrt(self.a**2 + self.b**2)

    def f(self):
        """
        Computes distance from center to hyperbola directrix
        """
        return self.a**2 / self.c()

    def foci(self):
        """
        Returns the coordinates of the foci of the hyperbola.
        """
        c = self.c()
        # In local frame
        if self.vertical:
            f1_local = np.array([0, -c])
            f2_local = np.array([0, c])
        else:
            f1_local = np.array([-c, 0])
            f2_local = np.array([c, 0])

        R = self.rotation_matrix()
        f1 = self.center + R @ f1_local
        f2 = self.center + R @ f2_local
        return np.vstack([f1, f2])

    def asymptote_directions(self):
        """
        Returns unit direction vectors of the two asymptotes
        """
        if self.vertical:
            v1 = np.array([1, self.a / self.b])
            v2 = np.array([1, -self.a / self.b])
        else:
            v1 = np.array([1, self.b / self.a])
            v2 = np.array([1, -self.b / self.a])

        R = self.rotation_matrix()
        return R @ v1 / np.linalg.norm(v1), R @ v2 / np.linalg.norm(v2)

    def point_in_local_frame(self, point):
        """
        Returns point coordinates in local hyperbola frame : x-axis aligned with hyperbola focal axis, and centered @ hyperbola center
        """
        R = self.rotation_matrix()
        p_local = R.T @ (point - self.center)
        return p_local if not self.vertical else p_local[::-1]

    def domain_contains(self, point: np.ndarray) -> bool:
        """
        Returns True if the point is on the right hyperbola branch.
        """
        x_local, y_local = self.point_in_local_frame(point)

        if (x_local / self.a) < 0:
            # x/a < 0 : point is not on the right hyperbola branch
            # print(f"x/a < 0 : {point} is not on the right hyperbola branch")
            return False

        if abs(x_local) < self.f():
            # point lies between origin and directrix, outside hyperbola domain
            return False

        # if (x_local / self.a) < tolerance:
        #     # "x < tolerance * a → point is not on hyperbola domain, tolerance < 1"
        #     # print(f"x/a < 1 → {point} is not on hyperbola domain")
        #     return False

        if abs(y_local) > abs(self.b / self.a * x_local):
            # point is outside asymptote domain
            return False

        return True

    def distance_to(self, point):
        """
        Returns local x offset between point and hyperbola.
        """
        x_local, y_local = self.point_in_local_frame(point)

        t = np.arcsinh(y_local / self.b)
        x_check = self.a * np.cosh(t)

        return abs(x_local - x_check)

    def rotation_matrix(self):
        """
        Returns the rotation matrix to rotate the hyperbola to the global frame.
        """
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c, -s], [s, c]])

    def parametric_point(self, t):
        """
        Returns a point on hyperbola, using parametric coordinate in local hyperbola frame, then rotated to global.
        """
        x = self.a * np.cosh(t)
        y = self.b * np.sinh(t)
        p_local = np.array([x, y] if not self.vertical else [y, x])
        return self.center[:, np.newaxis] + self.rotation_matrix() @ p_local

    def plot(
        self,
        ax,
        t_max: float = 2,
        num_points: int = 100,
        color: str = "blue",
        render: str = "matplotlib",
        swapxy: bool = False,
        label=None,
    ):
        """
        Plots the hyperbola on a given axis. Render can be "matplotlib" or "bokeh".
        optionally set swapxy=True to swap x and y-axis for a top ARF view :  x=along, y=across, z=down.
        """
        t = np.linspace(-t_max, t_max, num_points)
        xy = self.parametric_point(t)
        center = self.center
        foci = self.foci()
        legend_label = self if label is None else label
        if swapxy:
            xy = xy[::-1]
            center = self.center[::-1]
            foci = np.roll(foci, 1, axis=1)

        if render == "matplotlib":
            ax.plot(*xy, label=legend_label, color=color)
            ax.scatter(*center, marker=".", color=color, label="center")
            ax.scatter(*foci.T, marker="+", color=color, label="foci")
        elif render == "bokeh":
            ax.line(x=xy[0], y=xy[1], line_width=1, color=color, legend_label=legend_label)
            ax.scatter(x=center[0], y=center[1], marker="o", color=color, legend_label="center")
            ax.scatter(x=foci[:, 0], y=foci[:, 1], marker="+", color=color, legend_label="foci")

    def intersect(self, other: Union["Hyperbola", "Line2D"]):
        """
        Returns the intersection points of two Hyperbolas, using analytical method.
        """
        if not isinstance(other, Hyperbola) and not isinstance(other, Line2D):
            raise TypeError("'other' must be an Hyperbola or Line2d object")

        if isinstance(other, Line2D):
            # If other is a Line2D, use its intersection method
            return self.__intersect_with_line(other)

        # construct transformation that translates first hyperbola focus point to origin and then rotated an angle α
        # about the coordinate origin, where α is the angle between first hyperbola focal axis and x-axis.
        xy_to_focus = Transform(
            translation=-self.foci()[0],
            rotation=get_rotation_matrix(self.foci()[0], self.foci()[1]),
            apply_rotation_first=False,
        )
        # Transform every foci to new places
        foci_t = xy_to_focus(np.vstack([self.foci(), other.foci()]))
        # extract parameters
        a = foci_t[1, 0]
        b = foci_t[2, 0]
        c = foci_t[2, 1]
        d = foci_t[3, 0]
        e = foci_t[3, 1]

        xy = np.vstack(
            [
                solve_quadratic(a, b, c, 2 * self.a, 2 * other.a),
                solve_quadratic(a, b, c, -2 * self.a, 2 * other.a),
                solve_quartic(a, b, c, d, e, 2 * self.a, 2 * other.a),
                solve_quartic(a, b, c, d, e, -2 * self.a, 2 * other.a),
            ]
        )
        # remove nan points
        xy = xy[~np.isnan(xy).any(axis=1)]
        # early return if no intersection found
        if xy.size == 0:
            return None
        # transform back coordinates to the original coordinate system
        xy = xy_to_focus.inverse()(xy)
        # remove possible duplicates (common solution of the two systems)
        xy = np.unique(xy, axis=0)
        # keep only intersection points that are on the right branch of each hyperbola
        within_both_domain = [self.domain_contains(pnt) and other.domain_contains(pnt) for pnt in xy]
        xy = xy[within_both_domain]
        # retain closest remaining point to hyperbola
        return (
            xy[np.argmin([np.sqrt(self.distance_to(pnt) ** 2 + other.distance_to(pnt) ** 2) for pnt in xy])]
            if xy.size > 0  # intersection point is not on the right hyperbola branch
            else None
        )

    def __intersect_with_line(self, line: "Line2D"):
        """
        Compute intersection points between the hyperbola and a 2D line.
        Returns a list of 0, 1, or 2 intersection points.
        """
        # Transform to local hyperbola coordinates
        R = self.rotation_matrix().T  # Inverse rotation
        p0_local = R @ (line.point - self.center)
        d_local = R @ line.direction

        # manage vertical hyperbola
        x0, y0 = p0_local if not self.vertical else p0_local[::-1]
        dx, dy = d_local if not self.vertical else d_local[::-1]

        # Solve quadratic: A*t^2 + B*t + C = 0
        A = (dx**2) / self.a**2 - (dy**2) / self.b**2
        B = (2 * x0 * dx) / self.a**2 - (2 * y0 * dy) / self.b**2
        C = (x0**2) / self.a**2 - (y0**2) / self.b**2 - 1

        t = np.roots([A, B, C])

        # if there is any real positive roots, returns intersection points
        if np.any(np.isreal(t)):
            t = np.real(t[np.isreal(t)])
            xy = line.parametric_point(t)
            for pnt in xy.T:
                # returns only one point
                if self.domain_contains(pnt):
                    return pnt
        return None

    def __repr__(self):
        return f"Hyperbola: a={self.a}, b={self.b}, center={tuple(self.center)}, theta={self.theta}, vertical={self.vertical}"


def get_rotation_matrix(first_point, second_point):
    """
    Return the rotation matrix to rotate the vector AB (1st hyperbolas foci) to the x-axis.
    """
    cos_alpha = (second_point[0] - first_point[0]) / np.linalg.norm(second_point - first_point)
    sin_alpha = (second_point[1] - first_point[1]) / np.linalg.norm(second_point - first_point)
    # rotation matrix
    return np.array([[cos_alpha, sin_alpha], [-sin_alpha, cos_alpha]])


def solve_quadratic(a, b, c, l, r):
    """
    Retrieves intersection points of two hyperbolas in xy-plane, which are perpendicular and symmetric to the x-axis,
    Solves system of hyperbolic equations using coordinate transformation and quadratic polynomial simplification,
    implementing algorithm from : J. Vesely and S. V. Doan, 2015, doi: 10.1109/RADIOELEK.2015.7129064
    Algorithm has been corrected to account for the case when hyperbolas are perpendicular and symmetric to the x-axis,
    which simplifies the quartic equation to a quadratic one : the paper proposed solution is not working in this case.
    if c=-e => U3=0 and if b=d => U4=0, so eq. 17 of the Vasely et al. paper simplifies to a quadratic equation
    """
    A = (a**2 - l**2) / (2 * a)
    B = -l / a
    G = b**2 + c**2 - 2 * A * b
    H = 2 * c
    J = -2 * B * b
    M = b**2 + c**2 - 2 * A * b
    N = -2 * c

    Q = (H - N) / (2 * r)
    S = (G - M - r**2) / (2 * r)

    U1 = -1 + Q**2 - B**2 * Q**2
    U2 = -J - 2 * A * B * Q**2
    U5 = S**2 - M - A**2 * Q**2

    k = get_real_positive_roots(np.roots([U1, U2, U5]))
    # k = quadratic_real_positive_roots(U1, U2, U5)
    # if there is any real positive roots, returns intersection points
    if k.size > 0 and np.all(np.isfinite(k)):  # np.any(np.isreal(k)) and np.any(k[np.isreal(k)] > 0):
        # k = np.real(k[np.isreal(k)])
        # y coordinate is symetric to local x-axis -> one root gives two intersection points with same x coordinate
        xy = np.tile(np.array([[1, 1], [1, -1]], dtype=float), (len(k), 1))
        xy[:2] = xy[:2] * np.array([A + B * k[0], np.sqrt(k[0] ** 2 - A**2 - 2 * A * B * k[0] - B**2 * k[0] ** 2)])
        if len(k) > 1:
            xy[-2:] = xy[-2:] * np.array(
                [A + B * k[1], np.sqrt(k[1] ** 2 - A**2 - 2 * A * B * k[1] - B**2 * k[1] ** 2)]
            )
    else:  # No intersection
        xy = np.full((1, 2), np.nan)

    return xy


def solve_quartic(a, b, c, d, e, l, r) -> np.ndarray | None:
    """
    Retrieves intersection points of two hyperbolas in xy-plane, which are not perpendicular nor symmetric to the x-axis
    Solves system of hyperbolic equations using coordinate transformation and quartic polynomial simplification,
    implementing algorithm from : J. Vesely and S. V. Doan, 2015, doi: 10.1109/RADIOELEK.2015.7129064
    """
    A = (a**2 - l**2) / (2 * a)
    B = -l / a
    F = -2 * B * d
    G = d**2 + e**2 - 2 * A * d
    H = -2 * e
    J = -2 * B * b
    M = b**2 + c**2 - 2 * A * b
    N = -2 * c

    P = (F - J) / (2 * r)
    Q = (H - N) / (2 * r)
    S = (G - M - r**2) / (2 * r)

    U1 = P**2 - 1 + Q**2 - B**2 * Q**2
    U2 = 2 * P * S - J - 2 * A * B * Q**2
    U3 = 2 * Q * S - N
    U4 = 2 * P * Q
    U5 = S**2 - M - A**2 * Q**2

    W1 = -(A**2)
    W2 = -2 * A * B
    W3 = -(B**2 - 1)
    W4 = U3**2
    W5 = 2 * U3 * U4
    W6 = U4**2
    W7 = -U5
    W8 = -U2
    W9 = -U1

    M4 = W3 * W6 - W9**2
    M3 = W2 * W6 + W3 * W5 - 2 * W8 * W9
    M2 = W2 * W5 + W1 * W6 - 2 * W7 * W9 - W8**2 + W3 * W4
    M1 = -2 * W7 * W8 + W2 * W4 + W1 * W5
    M0 = W1 * W4 - W7**2

    # solve full quartic equation (eq. 20 in Vasely et al. paper)
    k = get_real_positive_roots(np.roots([M4, M3, M2, M1, M0]))
    denominator = U3 + U4 * k
    # Todo : tracer le cercle de rayon -U3/U4
    # if there is any real positive roots, and non zero denominator, returns intersection points
    if k.size > 0 and np.all(np.isfinite(k)) and np.any(denominator):  # np.any(k):
        # Solution #1 : 1 root gives 1 intersection point
        xy = np.zeros((len(k), 2), dtype=np.float64)
        xy[:, 0] = A + B * k
        xy[:, 1] = -(U1 * k**2 + U2 * k + U5) / denominator

        # # Solution #2 : 1 root gives 2 intersection points with same x coordinate and symmetric y coordinates
        # # need to test afterward for intersection coordinates belonging to the right hyperbola branch
        # xy = np.zeros((len(k), 2), dtype=np.float64)
        # xy[:, 0] = A + B * k
        # xy[:, 1] = np.sqrt((1 - B**2) * k**2 - 2 * A * B * k - A**2)
        # xy = expand_xy(xy)

        # Solution #3 : find y coordinate through Tx hyperbola equation (need to keep semi-major and semi-minor axis)
        # hence 1 root gives 2 intersection points with same x coordinate and symmetric y coordinates
        # xy[:, 1] = semiminoraxis / semimajoraxis * np.sqrt((xy[:, 0] - a / 2) ** 2 - semimajoraxis**2)
        # xy = expand_xy(xy)

    else:  # No intersection
        xy = np.full((1, 2), np.nan)

    return xy  # , np.tile(xy_old, (2, 1))  # , k, -U3 / U4  # * k), a, b, c, d, e


def get_real_positive_roots(roots, angle_deg_tol=1e-6):
    """
    Filters the roots to return only real positive roots.
    Imagine roots are considered real if their angle is close to 0 or 180 degrees.
    """

    is_real_roots = np.logical_or(np.isreal(roots), np.abs(np.angle(roots, deg=True)) % 180 < angle_deg_tol)
    if np.any(is_real_roots):
        real_roots = np.real(roots[is_real_roots])
        return real_roots[real_roots > 0]
    return np.array([])


def intersect_tx_rx_beams_analytical(r, m, tx_steer, rx_steer, array_separation, plot=False, render="matplotlib"):
    """
    Intersects Tx and Rx beams, and find an analytical solution, based on hyperbolas intersection on a focal plane
    r: distance from the Tx to the focal plane (m)
    m: angle of the focal plane (radians)
    tx_steer: steering angle of the Tx beam (radians)
    rx_steer: steering angle of the Rx beam (radians)
    array_separation: (x, y, z) separation of the Rx array from the Tx beam center (m)
    """
    # init Tx beam
    a_tx = r * np.tan(tx_steer)
    b_tx = r
    if a_tx == 0:  # or np.abs(a_tx / b_tx) < line_tol:
        tx_beam = Line2D(point=(a_tx, 0), theta=np.pi / 2)
        # tx_beam = Line2D(point=(0, 0), theta=np.pi / 2)
    else:
        tx_beam = Hyperbola(a=a_tx, b=b_tx)

    # init Rx beam
    a_rx = (r - array_separation[2]) * np.tan(-rx_steer)
    b_rx = r - array_separation[2]
    if a_rx == 0:  # or np.abs(a_rx / b_rx) < line_tol:
        # rx_beam = Line2D(point=(array_separation[0], array_separation[1]), theta=m)
        rx_beam = Line2D(
            point=(array_separation[0] + a_rx * np.sin(m), array_separation[1] + a_rx * np.cos(m)), theta=m
        )
    else:
        rx_beam = Hyperbola(
            a=a_rx,
            b=b_rx,
            vertical=True,
            center=(array_separation[0], array_separation[1]),
            theta=m,
        )

    # compute intersection on the focal plane
    sounding = tx_beam.intersect(rx_beam)

    if plot:
        if render == "matplotlib":
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            tx_beam.plot(ax, label="Tx beam", color="blue", swapxy=True)
            rx_beam.plot(ax, label="Rx beam", color="red", swapxy=True)
            if sounding is not None:
                ax.scatter(sounding[:, 1], sounding[:, 0], color="green", marker="o", label="Intersection Points")
            # legend outside the plot
            ax.set_xlabel("y 'across'")
            ax.set_ylabel("x 'along'")
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
            ax.grid()
            plt.show()
        elif render == "bokeh":
            fig = figure(
                title="beam intersection",
                width=1200,
                height=400,
                match_aspect=True,
                x_axis_label="y 'across'",
                y_axis_label="x 'along'",
            )
            fig.xaxis.fixed_location = 0.0
            fig.yaxis.fixed_location = 0.0
            tx_beam.plot(fig, color="blue", label="Tx beam", render="bokeh", swapxy=True)
            rx_beam.plot(fig, t_max=0.5, label="Rx beam", color="red", render="bokeh", swapxy=True)
            if sounding is not None:
                fig.scatter(
                    x=sounding[1],
                    y=sounding[0],
                    color="green",
                    marker="o",
                    legend_label="Intersection Points",
                )
            # legend outside the plot
            show(fig)

    return sounding
