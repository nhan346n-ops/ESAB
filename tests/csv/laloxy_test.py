import os

import pyat
import pytest
import tempfile
from pyat.csv import laloxy

INPUT_TEST_DATA = """a;b;c;d;e;f;g;h;i;j;
24.09722933;84.61457158;11.44;21.44;31.44;10.00;12;02;;;
24.03642231;81.61463279;41.10;31.10;21.10;20.00;1;0;;;
24.68902290;83.61798830;;;0;;;;0;;
25.34491703;87.61786748;21.10;21.10;21.10;0.00;1;0;;;
25.49859012;85.59687661;;;2;;;;;2;
25.52652437;82.59684409;;45;;;;;;;"""
EXPECTED_OUTPUT = """+000773639.42;+009462374.22;11.44;21.44;31.44;10.0;12.0;2.0;;;
+000924640.13;+009162888.72;41.1;31.1;21.1;20.0;1.0;0.0;;;
+000830697.30;+009366480.21;;;0.0;;;;0.0;;
+000626252.95;+009763860.05;21.1;21.1;21.1;0.0;1.0;0.0;;;
+000734432.74;+009565742.75;;;2.0;;;;;2.0;
+000894113.73;+009270875.07;;45.0;;;;;;;"""
INPUT_TEST_DATA_REFORMED = """24.09723;84.61457;11.44;21.44;31.44;10.0;12.0;2.0;;;
24.03642;81.61463;41.1;31.1;21.1;20.0;1.0;0.0;;;
24.68902;83.61799;;;0.0;;;;0.0;;
25.34492;87.61787;21.1;21.1;21.1;0.0;1.0;0.0;;;
25.49859;85.59688;;;2.0;;;;;2.0;
25.52652;82.59684;;45.0;;;;;;;"""
INPUT_TEST_DATA_THALIA = """latitude;longitude;elevation
48.329925361940184;-4.681994281239172;-18.6322184801
48.32992437577629;-4.681977780041037;-18.6403018236
48.32992333925961;-4.681960419834223;-18.6407080888
48.32992243002875;-4.681945205791632;-18.6486598253
48.329921537800196;-4.681930263597675;-18.6502677202
48.32991822294802;-4.681874728766759;-18.6482440233"""
EXPECTED_OUTPUT_THALIA = """+007144780.58;-000829678.75;-18.6322184801
+007144780.70;-000829675.83;-18.6403018236
+007144780.84;-000829672.76;-18.6407080888
+007144780.95;-000829670.06;-18.6486598253
+007144781.07;-000829667.42;-18.6502677202
+007144781.49;-000829657.59;-18.6482440233"""


def test_converters_from_wildDEGREES():
    assert pyat.utils.coords.DEGREES_from_wildDEGREES("-180°") == "-180.0"
    assert pyat.utils.coords.DEGREES_from_wildDEGREES("180") == "180.0"


def test_converters_from_DEGREES():
    func = pyat.utils.coords.formatted_coordinates
    assert func(-19.9128, -77.508333, "{d}") == ("19.9128", "77.508333")
    assert func(-19, -77.508333, "{D}") == ("19", "77")
    assert func(-19.9128, -77.508333, "{b}") == ("-19.9128", "-77.508333")
    assert func(164.754167, -77.508333, "{D} {M} {S} {w}") == ("164 45 15.0012 E", "77 30 29.9988 S")
    assert func(-164.754167, -77.508333, "{D} {M} {S} {w}") == ("164 45 15.0012 W", "77 30 29.9988 S")
    assert func(164.754167, -77.508333, "{x}", rounding=2) == ("933813.46", "-11324361.6")


def template_laloxy_object(input_data: str, expected_output: str, input_format: str, **kwargs):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as fd:
        tmp_in = fd.name
        fd.write(input_data)
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as fd:
        tmp_out = fd.name
    if "delimiter" not in kwargs:
        kwargs["delimiter"] = ";"
    obj = laloxy.Laloxy(tmp_in, tmp_out, {"Longitude/X": 0, "Latitude/Y": 1}, input_format, overwrite=True, **kwargs)
    obj()
    with open(tmp_out) as fd:
        found_output = fd.read()
    assert found_output.strip() == expected_output
    os.remove(tmp_in)
    os.remove(tmp_out)


def test_laloxy_object_base():
    proj_lonlat = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    proj_utm = "+proj=utm +zone=30 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    template_laloxy_object(
        INPUT_TEST_DATA,
        EXPECTED_OUTPUT,
        "DEGREES",
        skip_rows=1,
        output_format="CARAIBES_XY",
        input_proj=proj_lonlat,
        output_proj=proj_utm,
    )


def test_laloxy_object_base_reverse():
    proj_lonlat = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    proj_utm = "+proj=utm +zone=30 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    template_laloxy_object(
        EXPECTED_OUTPUT,
        INPUT_TEST_DATA_REFORMED,
        "XY",
        output_format="DEGREES",
        rounding=5,
        skip_rows=0,
        input_proj=proj_utm,
        output_proj=proj_lonlat,
    )


def test_laloxy_object_minithalia():
    proj_lonlat = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    proj_utm = "+proj=utm +zone=30 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    template_laloxy_object(
        INPUT_TEST_DATA_THALIA,
        EXPECTED_OUTPUT_THALIA,
        "DEGREES",
        output_format="CARAIBES_XY",
        skip_rows=1,
        input_proj=proj_lonlat,
        output_proj=proj_utm,
    )


def test_laloxy_object_minithalia_to_itself():
    proj = "+proj=utm +zone=30 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    template_laloxy_object(
        EXPECTED_OUTPUT_THALIA,
        EXPECTED_OUTPUT_THALIA,
        input_format="XY",
        output_format="CARAIBES_XY",
        input_proj=proj,
        output_proj=proj,
    )


def test_laloxy_object_with_output_caraibe_degminsec():
    proj = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    indata = "167.533300;-21.50000000;-1423.77"
    template_laloxy_object(
        indata,
        "+167  31.998000;-21  30.000000;-1423.77",
        input_format="DEGREES",
        output_format="CARAIBES_DEG_MIN_DEC",
        input_proj=proj,
        output_proj=proj,
    )
    template_laloxy_object(
        indata,
        "+000000167.53;-000000021.50;-1423.77",
        input_format="DEGREES",
        output_format="CARAIBES_XY",
        input_proj=proj,
        output_proj=proj,
    )
    template_laloxy_object(
        indata, "167.5333;-21.5;-1423.77", input_format="DEGREES", output_format="XY", input_proj=proj, output_proj=proj
    )


def test_laloxy_object_with_laloxy_example_data():
    proj_deg = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    proj_xy = "+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    data_deg = "+7.945494 +43.431848 -2367.6"
    data_xy = "+000414653.07 +004809312.58 -2367.6"
    template_laloxy_object(
        data_deg,
        data_xy,
        input_format="CARAIBES_DEGREES",
        output_format="CARAIBES_XY",
        input_proj=proj_deg,
        output_proj=proj_xy,
        delimiter=r"\s+",
    )
    template_laloxy_object(
        data_xy,
        data_deg,
        input_format="CARAIBES_XY",
        output_format="CARAIBES_DEGREES",
        input_proj=proj_xy,
        output_proj=proj_deg,
        delimiter=r"\s+",
    )


def test_laloxy_object_utm_to_latlon():
    in_proj = "+proj=utm +zone=30 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    out_proj = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    template_laloxy_object(
        "425404.89;5316784.01;42",
        "-4.0;48.0;42",
        input_format="XY",
        output_format="DEGREES",
        skip_rows=0,
        input_proj=in_proj,
        output_proj=out_proj,
    )


def test_laloxy_example_csv():
    """Show we can't handle a specific csv example"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as fd:
        tmp_in = fd.name
        # columns:    0  1    3  4  5  6 7 8      9 10 11      12  13  14
        fd.write(
            r"""11 04 1991 06 00 00 N43 27.7373 E  8  1.0971   2   0
11 04 1991 06 33 20 N43 28.8249 E  7 53.3882   2   0"""
        )
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as fd:
        tmp_out = fd.name
    with pytest.raises(NotImplementedError):
        laloxy.Laloxy(
            tmp_in,
            tmp_out,
            {"Longitude/X": (11, 12, 10), "Latitude/Y": (8, 9, 7)},
            "{w}{P}  {m:07.4f}",
            delimiter=" ",
            overwrite=True,
        )
    os.remove(tmp_in)
    os.remove(tmp_out)


def test_converters_robustness_1():
    dd_to_xy = pyat.utils.coords.create_lonlat_to_xy_converter()
    xy_to_dd = pyat.utils.coords.create_xy_to_lonlat_converter(rounding=2)
    print(dd_to_xy([164.75], [-77.51]))
    assert xy_to_dd(*dd_to_xy([164.75], [-77.51])) == ((164.75,), (-77.51,))


def test_converters_robustness_2():
    xy_to_dd = pyat.utils.coords.create_xy_to_lonlat_converter()
    dd_to_xy = pyat.utils.coords.create_lonlat_to_xy_converter(rounding=2)
    assert dd_to_xy(*xy_to_dd([1396487.30], [494064.21])) == ((1396487.30,), (494064.21,))


def test_converters_robustness_3_with_proj():
    proj = "+proj=utm +zone=30 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    xy_to_dd = pyat.utils.coords.create_xy_to_lonlat_converter(proj)
    dd_to_xy = pyat.utils.coords.create_lonlat_to_xy_converter(proj, rounding=2)
    assert dd_to_xy(*xy_to_dd([1396487.30], [494064.21])) == ((1396487.30,), (494064.21,))


def test_converters_robustness_4_with_proj():
    proj = "+proj=utm +zone=30 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    xy_to_dd = pyat.utils.coords.create_xy_to_lonlat_converter(proj)
    dd_to_xy = pyat.utils.coords.create_lonlat_to_xy_converter(proj, rounding=2)
    assert dd_to_xy(*xy_to_dd([24.09722933], [84.61457158])) == ((24.1,), (84.62,))
    assert xy_to_dd([24.09722933], [84.61457158]) == ((-7.488528,), (0.0007632,))
