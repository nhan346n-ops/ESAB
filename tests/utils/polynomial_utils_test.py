import numpy as np
import pytest

from pyat.utils.polynomial_utils import quadratic_roots


@pytest.mark.parametrize(
    "a, b, c, expected",
    [
        # Two real roots
        (1, -3, 2, (2.0, 1.0)),  # x^2 - 3x + 2 = 0 --> x=1,2
        (2, -8, 6, (3.0, 1.0)),  # 2x^2 - 8x + 6 = 0 --> x=1,3
        # Double root
        (1, -2, 1, (1.0, 1.0)),  # x^2 - 2x + 1 = 0 --> x=1 (double root)
        # Complex roots
        (1, 0, 1, (complex(0, -1), complex(0, 1))),  # x^2 + 1 = 0 --> x=±i
        (1, 2, 5, (complex(-1, -2), complex(-1, 2))),  # x^2 + 2x + 5 = 0
        # Negative leading coefficient
        (-1, 3, -2, (2.0, 1.0)),  # -x^2 + 3x - 2 = 0 --> x=1,2
    ],
)
def test_quadratic_roots(a, b, c, expected):
    roots = quadratic_roots(a, b, c)
    # Roots may be returned in any order
    assert np.allclose(
        sorted(roots, key=lambda x: (x.real, x.imag)), sorted(expected, key=lambda x: (x.real, x.imag)), atol=1e-12
    )


def test_quadratic_roots_zero_leading():
    # Should raise ZeroDivisionError if a == 0
    with pytest.raises(ZeroDivisionError):
        quadratic_roots(0, 10, 50)


def test_quadratic_roots_against_numpy():
    # Test random quadratic coefficients against numpy.roots
    rng = np.random.default_rng(42)
    for _ in range(100):
        a = rng.uniform(-10, 10)
        # Ensure a != 0
        if abs(a) < 1e-8:
            a = 1.0
        b = rng.uniform(-10, 10)
        c = rng.uniform(-10, 10)
        expected = np.roots([a, b, c])
        actual = quadratic_roots(a, b, c)
        # Roots may be returned in any order
        assert np.allclose(
            sorted(actual, key=lambda x: (x.real, x.imag)), sorted(expected, key=lambda x: (x.real, x.imag)), atol=1e-12
        )
