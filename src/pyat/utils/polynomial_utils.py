"""
Module providing polynomial utilities to compute roots of quadratic and quartic equations,
"""

import cmath
import math

import numpy as np
from numba import njit


@njit
def quadratic_roots(a0: float, b0: float, c0: float) -> np.ndarray:
    """
    Analytical closed-form solver for quadratic equation.
    Inputs are coefficients of the quadratic polynomial:  a0*x^2 + b0*x + c0 = 0
    Returns a tuple of two roots of a given polynomial (stored by default as complex to embrace all type of root).
    Adapted from Nino Krvavica FQS (Fast Quartic and Cubic solver).
    """
    if a0 == 0:
        raise ZeroDivisionError(f"a0 coefficient must be non-zero")

    # Reduce the quadratic equation to to form: x^2 + ax + b = 0
    a, b = b0 / a0, c0 / a0

    # compute discriminant
    a0 = -0.5 * a
    delta = a0 * a0 - b
    sqrt_delta = cmath.sqrt(delta)

    # compute Roots
    r1 = a0 - sqrt_delta
    r2 = a0 + sqrt_delta

    return np.array([r1, r2])


@njit
def resolvent_cubic_root(a0: float, b0: float, c0: float, d0: float) -> float:
    """
    Analytical closed-form solver for cubic equation, tailored for root finding of quartic's cubic resolvent.
    Returns only one real root out of the three possible ones.
    Parameters : oefficients of the Cubic polynomial: a0*x^3 + b0*x^2 + c0*x + d0 = 0
    Returns  roots: float
    Adapted from Nino Krvavica FQS (Fast Quartic and Cubic solver).
    """
    if a0 == 0:
        raise ZeroDivisionError(f"a0 coefficient must be non-zero")

    # Normalize cubic equation to x^3 + a*x^2 + bx + c = 0
    a, b, c = b0 / a0, c0 / a0, d0 / a0

    # Some repeating constants and variables
    third = 1.0 / 3.0
    a13 = a * third
    a2 = a13 * a13

    # Additional intermediate variables
    f = third * b - a2
    g = a13 * (2 * a2 - b) + c
    h = 0.25 * g * g + f * f * f

    def cubic_root(x):
        """
        Computes cubic root of a number, while maintaining its sign.
        """
        if x.real >= 0:
            return x**third
        else:
            return -((-x) ** third)

    if f == g == h == 0:
        return -cubic_root(c)

    elif h <= 0:
        j = math.sqrt(-f)
        theta = -0.5 * g / (j * j * j)
        if abs(theta - 1.0) < 1e-9:
            # avoid out of domain
            # theta ~=1 --> k == 0.0 and m == 1.0
            m = 1.0
        else:
            k = math.acos(-0.5 * g / (j * j * j))
            m = math.cos(third * k)
        return 2 * j * m - a13

    else:
        sqrt_h = cmath.sqrt(h)
        S = cubic_root(-0.5 * g + sqrt_h)
        U = cubic_root(-0.5 * g - sqrt_h)
        S_plus_U = S + U
        return S_plus_U - a13


@njit
def quartic_roots(a0: float, b0: float, c0: float, d0: float, e0: float) -> np.ndarray:
    """
    Analytical closed-form solver for quartic equation.
    Inputs are coefficients of the quartic polynomial:  a0*x^4 + b0*x^3 + c0*x^2 + d0*x + e0 = 0.
    Returns a tuple of four roots (stored by default as complex to embrace all type of root)
    Adapted from Nino Krvavica FQS (Fast Quartic and Cubic solver).
    """
    if a0 == 0:
        raise ZeroDivisionError(f"a0 coefficient must be non-zero")

    # Normalize quartic equation to x^4 + a*x^3 + b*x^2 + c*x + d = 0
    a, b, c, d = b0 / a0, c0 / a0, d0 / a0, e0 / a0

    # Some repeating variables
    a0 = 0.25 * a
    a02 = a0 * a0

    # Coefficients of resolvent cubic equation
    p = 3 * a02 - 0.5 * b
    q = a * a02 - b * a0 + 0.5 * c
    r = 3 * a02 * a02 - b * a02 + c * a0 - d

    # Get one root of the resolvent cubic equation
    z0 = resolvent_cubic_root(1, p, r, p * r - 0.5 * q * q)

    # Additional variables
    s = cmath.sqrt(2 * p + 2 * z0.real + 0j)
    if s == 0:
        t = z0 * z0 + r
    else:
        t = -q / s

    # Compute roots by quadratic equations
    r0, r1 = quadratic_roots(1, s, z0 + t)
    r2, r3 = quadratic_roots(1, -s, z0 - t)

    return np.array([r0 - a0, r1 - a0, r2 - a0, r3 - a0])


@njit
def get_real_roots(roots: np.ndarray, angle_deg_tol: float = 1e-6) -> np.ndarray:
    """
    Filters the roots to return only real positive roots.
    Imagine roots are considered real if their angle is close to 0 or 180 degrees.
    Returns NaN array if no real roots.
    """

    is_real_roots = np.logical_or(np.isreal(roots), np.abs(np.angle(roots, deg=True)) % 180 < angle_deg_tol)
    if np.any(is_real_roots):
        real_roots = np.real(roots[is_real_roots])
        return real_roots
    return np.full(1, np.nan)


@njit
def quartic_real_positive_roots(a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
    """
    Returns real and positive only roots of quartic a*x^4 + b*x^3 + c*x^2 + d*x + e = 0.
    """
    # compute all quadratic roots
    roots = quartic_roots(a, b, c, d, e)
    # return real positive roots only
    real_roots = get_real_roots(roots)
    return real_roots[real_roots > 0]


@njit
def quadratic_real_roots(a: float, b: float, c: float) -> np.ndarray:
    """
    Returns real only roots of quadratic a*x^2 + b*x + c = 0.
    """
    # compute all 4 quartic roots
    roots = quadratic_roots(a, b, c)
    # return real roots only
    return get_real_roots(roots)


@njit
def quadratic_real_positive_roots(a: float, b: float, c: float) -> np.ndarray:
    """
    Returns real and positive only roots of quadratic a*x^2 + b*x + c = 0.
    """
    # compute all 4 quartic roots
    roots = quadratic_roots(a, b, c)
    # return real positive roots only
    real_roots = get_real_roots(roots)
    return real_roots[real_roots > 0]


@njit
def solve_quadratic(a: float, b: float, c: float, l: float, r: float) -> np.ndarray:
    """
    Retrieves intersection points of two hyperbolas in xy-plane, which are perpendicular and symmetric to the x-axis,
    Solves system of hyperbolic equations using coordinate transformation and quadratic polynomial simplification,
    implementing algorithm from : J. Vesely and S. V. Doan, 2015, doi: 10.1109/RADIOELEK.2015.7129064
    Algorithm has been corrected to account for the case when hyperbolas are perpendicular and symmetric to the x-axis,
    which simplifies the quartic equation to a quadratic one : the paper proposed solution is not working in this case.
    if c=-e => U3=0 and if b=d => U4=0, so eq. 17 of the Vasely et al. paper simplifies to a quadratic equation
    """
    # some constants
    b2 = b * b
    c2 = c * c

    A = (a**2 - l**2) / (2 * a)
    B = -l / a
    G = b2 + c2 - 2 * A * b
    H = 2 * c
    J = -2 * B * b
    M = b2 + c2 - 2 * A * b
    N = -2 * c

    Q = (H - N) / (2 * r)
    Q2 = Q * Q
    S = (G - M - r**2) / (2 * r)

    U1 = -1 + Q2 - B**2 * Q2
    U2 = -J - 2 * A * B * Q2
    U5 = S**2 - M - A**2 * Q2

    k = quadratic_real_positive_roots(U1, U2, U5)
    # if there is any real positive roots, returns intersection points
    if k.size > 0 and np.all(np.isfinite(k)):
        # y coordinate is symetric to local x-axis -> one root gives two intersection points with same x coordinate
        # xy = np.tile(np.array([[1, 1], [1, -1]], dtype=float), (len(k), 1))
        xy = np.empty((2 * k.size, 2), dtype=np.float64)
        for i in range(k.size):
            xy[2 * i] = [1.0, 1.0]
            xy[2 * i + 1] = [1.0, -1.0]

        xy[:2] = xy[:2] * np.array([A + B * k[0], np.sqrt(k[0] ** 2 - A**2 - 2 * A * B * k[0] - B**2 * k[0] ** 2)])
        if len(k) > 1:
            xy[-2:] = xy[-2:] * np.array(
                [A + B * k[1], np.sqrt(k[1] ** 2 - A**2 - 2 * A * B * k[1] - B**2 * k[1] ** 2)]
            )
    else:  # No intersection
        xy = np.full((1, 2), np.nan)

    return xy


@njit
def solve_quartic(a: float, b: float, c: float, d: float, e: float, l: float, r: float) -> np.ndarray:
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
    # k = np.roots([M4, M3, M2, M1, M0])
    # k = get_real_positive_roots(Polynomial([M0, M1, M2, M3, M4]).roots())
    # return k
    # Todo : tracer le cercle de rayon -U3/U4
    # if there is any real positive roots, returns intersection points
    k = quartic_real_positive_roots(M4, M3, M2, M1, M0)
    denominator = U3 + U4 * k
    if k.size > 0 and np.all(np.isfinite(k)) and np.any(denominator):  # np.any(k):
        # Solution #1 : 1 root gives 1 intersection point
        xy = np.zeros((k.size, 2), dtype=np.float64)
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
