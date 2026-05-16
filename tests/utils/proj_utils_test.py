import pytest
from pyproj.exceptions import CRSError

from pyat.utils.proj_utils import validate_proj4_string


def test_validate_proj4_string_valid():
    # Standard valid UTM proj4 string
    proj_str = "+proj=utm +zone=33 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    result = validate_proj4_string(proj_str)
    assert "+proj=utm" in result
    assert "+zone=33" in result
    assert "+no_defs" in result


def test_validate_proj4_string_missing_plus():
    # Missing '+' in one token
    proj_str = "proj=utm +zone=33 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    result = validate_proj4_string(proj_str)
    assert "+proj=utm" in result


def test_validate_proj4_string_flag_only_with_equal():
    # Flag-only keyword with '='
    proj_str = "+proj=utm +zone=33 +no_defs= +datum=WGS84"
    result = validate_proj4_string(proj_str)
    assert "+no_defs" in result
    assert "+no_defs=" not in result


def test_validate_proj4_string_missing_equal():
    # Parameter missing '=' and not a flag
    proj_str = "+proj=utm +zone33 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    with pytest.raises(CRSError) as excinfo:
        validate_proj4_string(proj_str)
    assert "Missing '=' in parameter" in str(excinfo.value)


def test_validate_proj4_string_empty_value():
    # Parameter ends with '=' (empty value)
    proj_str = "+proj=utm +zone= +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    with pytest.raises(CRSError) as excinfo:
        validate_proj4_string(proj_str)
    assert "Empty value for parameter" in str(excinfo.value)


def test_validate_proj4_string_invalid_proj():
    # Completely invalid proj string
    proj_str = "+proj=invalid +zone=33"
    with pytest.raises(CRSError) as excinfo:
        validate_proj4_string(proj_str)
    assert "CRS parsing error" in str(excinfo.value)


def test_validate_proj4_string_all_flags():
    # Only flag-only keywords
    proj_str = "+no_defs +over +south +guam +geoc +wktext"
    with pytest.raises(CRSError):
        validate_proj4_string(proj_str)
