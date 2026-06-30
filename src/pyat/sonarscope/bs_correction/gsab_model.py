import numpy as np
from scipy.optimize import curve_fit
from numpy.typing import ArrayLike

from pyat.utils.signal import db_to_db_mean_energy, db_to_energy, energy_to_db


class GsabDataCoefficients:
    # Specular level A (dB/m2)
    # Specular angular extent B (deg)
    # Lambert reference C (dB/m2)
    # Lambert decrement D
    # Transitory level E (dB/m2)
    # Transitory angular extent F (deg)

    # dB Units choosen to fit with Sonarscope. Before used in the gsab standard formula, coeffs should be converted in natural energy and angles devided by 2

    def __init__(self, coeffs: ArrayLike):
        coeff_array = np.array(coeffs)
        self.a = np.float32(coeff_array[0])
        self.b = np.float32(coeff_array[1])
        self.c = np.float32(coeff_array[2])
        self.d = np.float32(coeff_array[3])
        self.e = np.float32(coeff_array[4])
        self.f = np.float32(coeff_array[5])

    def __str__(self) -> str:
        return (
            f"GsabDataCoefficients(a={self.a}, b={self.b}, c={self.c}, d={self.d}, e={self.e}, f={self.f})\n"
            "Specular level A (dB/m2) \n"
            "Specular angular extent B (deg) \n"
            "Lambert reference C (dB/m2) \n"
            "Lambert decrement D \n"
            "Transitory level E (dB/m2) \n"
            "Transitory angular extent F (deg) \n"
            "dB Units choosen to fit with Sonarscope. Before used in the gsab standard formula, coeffs should be converted in natural energy and angles devided by 2"
        )

    def linear_coeffs(self):
        return LinearGsabDataCoefficients(
            [
                db_to_energy(self.a),
                self.b / 2,
                db_to_energy(self.c),
                self.d,
                db_to_energy(self.e),
                self.f / 2,
            ],
        )

    @staticmethod
    def default():
        return GsabDataCoefficients([1, 5, 0, 2, 0, 20])


class LinearGsabDataCoefficients:
    # Specular level A (dB/m2)
    # Specular angular extent B (deg)
    # Lambert reference C (dB/m2)
    # Lambert decrement D
    # Transitory level E (dB/m2)
    # Transitory angular extent F (deg)

    # dB Units choosen to fit with Sonarscope. Before used in the gsab standard formula, coeffs should be converted in natural energy and angles devided by 2

    def __init__(self, coeffs: ArrayLike):
        coeff_array = np.array(coeffs)
        self.a = np.float32(coeff_array[0])
        self.b = np.float32(coeff_array[1])
        self.c = np.float32(coeff_array[2])
        self.d = np.float32(coeff_array[3])
        self.e = np.float32(coeff_array[4])
        self.f = np.float32(coeff_array[5])

    def values(self):
        return (
            self.a,
            self.b,
            self.c,
            self.d,
            self.e,
            self.f,
        )

    def ab_coeffs(self):
        return LinearGsabDataCoefficients([self.a, self.b, 0.0, self.d, 0.0, self.f])

    def cd_coeffs(self):
        return LinearGsabDataCoefficients([0.0, self.b, self.c, self.d, 0.0, self.f])

    def ef_coeffs(self):
        return LinearGsabDataCoefficients([0.0, self.b, 0.0, self.d, self.e, self.f])

    def dB_coeffs(self):
        return GsabDataCoefficients(
            [
                energy_to_db(self.a),
                2 * self.b,
                energy_to_db(self.c),
                self.d,
                energy_to_db(self.e),
                2 * self.f,
            ]
        )


class GsabDataModel:
    def __init__(self, x, y, count):
        self.coeffs = GsabDataCoefficients.default()
        self.x = x
        self.y = y
        self.count = count
        self.init_coeffs()

    def init_coeffs(self):
        # initial conditions example :
        # Specular level A :  -9.1 -9.1 -13.6 -6.1
        # Specular angular extent B : 4.91 5.91 5.51 5.11
        # Lambert reference C : -17.9 -23.9 -29.1 -27.3
        # Lambert decrement D : 2 2 2 2
        # Transitory level E : -17.5 -20.9 -24.5 -20.7
        # Transitory angular extent F 16.21 17.91 16.21 12.41
        self.coeffs = GsabDataCoefficients(
            [np.nanmax(self.y), 6.0, db_to_db_mean_energy(self.y), 2, np.nanmin(self.y), 20.0]
        )
        self.coeffs_min = GsabDataCoefficients(
            [np.nanmin(self.y) - 6, 5.0, np.nanmin(self.y), 0.5, np.nanmin(self.y) - 6, 20.0]
        )
        self.coeffs_max = GsabDataCoefficients(
            [
                np.nanmax(self.y) + 3,
                20.0,
                np.nanmax(self.y),
                4.0,
                np.nanmax(self.y) + 3,
                np.nanmax(np.abs(self.x)) - 10.0,
            ],
        )

    def gsab_func(self, x, a, b, c, d, e, f):
        return 10 * np.log10(
            a * np.exp(-(x**2) / (2 * b**2)) + c * np.power(np.cos(np.radians(x)), d) + e * np.exp(-(x**2) / (2 * f**2))
        )

    def apply_gsab_func(self, linear_coeffs: LinearGsabDataCoefficients):
        return self.gsab_func(
            self.x,
            linear_coeffs.a,
            linear_coeffs.b,
            linear_coeffs.c,
            linear_coeffs.d,
            linear_coeffs.e,
            linear_coeffs.f,
        )

    def apply(self):
        return self.apply_gsab_func(self.coeffs.linear_coeffs())

    def fit_gsab(self):
        # Perform the curve fit
        mask = np.isfinite(self.x) & np.isfinite(self.y) & (self.count > 0)
        sigma = np.sqrt(np.max(self.count[mask]) * 1.0 / self.count[mask])
        # False positive pytlint warning
        # pylint: disable-next=unbalanced-tuple-unpacking
        popt, _ = curve_fit(
            self.gsab_func,
            self.x[mask],
            self.y[mask],
            p0=self.coeffs.linear_coeffs().values(),
            bounds=(self.coeffs_min.linear_coeffs().values(), self.coeffs_max.linear_coeffs().values()),
            sigma=sigma,
            nan_policy="omit",
        )

        # Extract the optimized parameters
        self.coeffs = LinearGsabDataCoefficients(popt).dB_coeffs()
        return self.coeffs
