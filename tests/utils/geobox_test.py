from pyat.common.geo_file import SR_PSEUDO_MERCATOR
from pyat.utils.argument_utils import Geobox
from pyat.utils.coords import DEG_MIN_SEC_from_DEGREES, DEGREES_from_DEG_MIN_SEC


def test_round_lonlat():
    """Check round method with lonlat coordinates"""
    geobox = Geobox(
        DEGREES_from_DEG_MIN_SEC("43°57'17'' N"),  # North
        DEGREES_from_DEG_MIN_SEC("43°54'55'' N"),  # South
        DEGREES_from_DEG_MIN_SEC("030°48'48'' E"),  # West
        DEGREES_from_DEG_MIN_SEC("030°53'04'' E"),  # East
    )
    geobox.realign()  # By befault, realigns bounds on arcmin
    # south/west must be rounded
    assert DEG_MIN_SEC_from_DEGREES(geobox.upper) == (43, 58, 0, False)
    assert DEG_MIN_SEC_from_DEGREES(geobox.lower) == (43, 54, 00, False)
    assert DEG_MIN_SEC_from_DEGREES(geobox.left) == (30, 48, 00, False)
    assert DEG_MIN_SEC_from_DEGREES(geobox.right) == (30, 54, 00, False)


def test_precision():
    """Check round method with lonlat coordinates, second near 0"""
    geobox = Geobox(
        43.75000013994674,  # N 43°45'00'' North
        41.06666666666667,  # N 41°04'00'' South
        8.23333355151117,  # E 008°14'00'' West
        4.05,  # E 004°03'00'' East
    )
    geobox.realign()  # By befault, realigns bounds on arcmin
    # south/west must be rounded
    assert DEG_MIN_SEC_from_DEGREES(geobox.upper) == (43, 45, 0, False)
    assert DEG_MIN_SEC_from_DEGREES(geobox.lower) == (41, 4, 00, False)
    assert geobox.left == 8.233333333333333  # Can't check with DEG_MIN_SEC_from_DEGREES because of precision
    assert DEG_MIN_SEC_from_DEGREES(geobox.right) == (4, 3, 00, False)


def test_round_mercator():
    """Check round method with a mercator projection"""
    geobox = Geobox(
        3918003.129,  # Upper
        3913623.854,  # Lower
        1064.741,  # Left
        6774.864,  # Right
        SR_PSEUDO_MERCATOR,
    )

    geobox.realign(5.0, 2.0)  # realigns bounds to a multiple of 5 on x and multiple of 2 on y
    assert geobox.upper == 3918004
    assert geobox.lower == 3913622
    assert geobox.left == 1060
    assert geobox.right == 6775
