from pyproj import CRS
from pyproj.exceptions import CRSError


def validate_proj4_string(proj_str):
    """
    Validates a PROJ string for correctness and completeness.
    :param proj_str: PROJstring to validate
    :return: valid PROJ string if valid, otherwise raises an error
    """
    flag_only_keywords = ["+no_defs", "+over", "+south", "+guam", "+geoc", "+wktext"]  # known flag-only keywords
    issues = []
    valid_proj_str = ""

    # Basic syntax check and quick fixes
    tokens = proj_str.strip().split()
    for token in tokens:
        if not token.startswith("+"):
            issues.append(f"\tMissing '+' at start of token: '{token}'\n")
            # try to fix it by adding the '+'
            valid_proj_str += f"+{token} "
        elif "=" in token and token.split("=")[0] in flag_only_keywords:
            issues.append(f"\tFlag-only keyword ending with '=': '{token}'\n")
            # try to fix it by removing the '='
            valid_proj_str += token.split("=")[0] + " "
        elif "=" not in token and token not in flag_only_keywords:
            issues.append(f"\tMissing '=' in parameter or unrecognized flag: '{token}'\n")
            # no possible fix for this, so just add it to the issues and keep the token
            valid_proj_str += token + " "
        elif token.endswith("="):
            issues.append(f"\tEmpty value for parameter: '{token}'\n")
            # no possible fix for this, so just add it to the issues keep the token
            valid_proj_str += token + " "
        else:
            valid_proj_str += token + " "

    # Attempt to parse fixed proj string using pyproj
    try:
        CRS.from_proj4(valid_proj_str)
        return valid_proj_str
    except CRSError as e:
        # Parsing error of tentative fix, let's get back to the original proj_str
        try:
            CRS.from_proj4(proj_str)
        except CRSError as e:
            issues.append(f"\tCRS parsing error: {e}\n")

    # Finaly, raises an error if there are any issues
    if issues:
        raise CRSError(f"Invalid PROJ string: '{proj_str}'\nIssues:\n {''.join(issues)}")

    # or return the proj_str if no issues for further use
    return proj_str


def lon_lat_to_utm_proj4(longitude, latitude):
    """
    returns the proj4 string for the UTM projection of a given lon/lat
    :param longitude: longitude in degrees
    :param latitude: latitude in degrees
    :return: proj4 string
    """
    utm_band = str(_lonlat_to_zone_number(longitude, latitude))
    w = "" if latitude >= 0 else " +south"
    return f"+proj=utm +zone={utm_band}{w} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"


def _lonlat_to_zone_number(longitude, latitude):
    """
    Returns the UTM zone number for a given longitude and latitude.
    :param longitude: Longitude in degrees
    :param latitude: Latitude in degrees
    :return: UTM zone number
    """
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude < 9:
            return 31
        elif longitude < 21:
            return 33
        elif longitude < 33:
            return 35
        elif longitude < 42:
            return 37

    return int((longitude + 180) / 6) + 1
