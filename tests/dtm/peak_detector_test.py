#! /usr/bin/env python3
# coding: utf-8
"""Testing peak and hole finder.

See test_artificial_dtm_with_shapefilter for an explanatory tour
of the generic_test_func API.
"""


import glob
import os
import shutil
import tempfile as tmp
from typing import List, Union

import numpy as np

import tests.directory_utils as directory
import tests.generator.dtm_generator as dtm_generator
from pyat.dtm.analyse.peak_detector import PeakFinder
from tests.generator.kml_generator import create_kml

DEFAULT_PARAMS = {
    "geo_masks": [],
    "peak_detection_threshold": 10,
    "percentile": 80,
    "size": 5,
    "use_gradient": False,
    "use_percent": False,
    "percent": 10,
    "percent_kernel": 2,
    "use_stdev": False,
    "maximum_allowed_std": 130,
    "kernel_size_for_mean_computation": 2,
    "kernel_size_for_stdev": 2,
    "use_holes_detection": True,
    "maximum_hole_area_in_pixel": 100,
}
FILL_HOLES_OUTDATA = """
"ID";"LATITUDE_DEG";"LONGITUDE_DEG";"LATITUDE_DMD";"LONGITUDE_DMD";"HEIGHT_ABOVE_SEA_SURFACE";"SEA_FLOOR_LAYER";"MARKER_COLOR";"MARKER_SIZE";"MARKER_SHAPE";"GROUP";"CLASS";"COMMENT"
"001";"{lat}";"{lon}";"";"";"0";"Globe default layer";"#ff0000ff";"50";"Sphere";"";"";""
""".strip()


# Elevations : 2 holes ([2,2] = [8,8] = np.nan) and 1 peak ([5,5] = -80.00)
ELEVATIONS = np.array(
    [  #                                        peak↴
        [-98.23, -99.83, -99.99, -98.14, -98.39, -98.43, -98.29, -99.25, -98.44, -99.25],
        [-98.18, -99.60, -99.58, -98.38, -99.17, -98.79, -98.32, -98.13, -99.74, -98.07],
        [-99.18, -98.73, np.nan, -99.69, -99.76, -98.64, -98.31, -98.96, -99.45, -98.67],
        [-99.40, -99.55, -99.55, -99.73, -99.68, -99.97, -99.06, -98.43, -99.98, -99.88],
        [-99.65, -99.91, -99.99, -98.49, -98.37, -98.88, -99.96, -99.52, -98.59, -98.47],
        [-98.33, -99.66, -99.24, -99.88, -98.84, -80.00, -98.57, -98.33, -99.51, -99.79],  # ← peak
        [-98.05, -99.36, -98.06, -98.67, -99.08, -99.42, -98.27, -99.22, -98.99, -98.40],
        [-99.59, -99.53, -99.58, -99.76, -98.23, -99.59, -99.66, -99.14, -99.86, -98.54],
        [-98.75, -98.62, -98.45, -99.36, -98.62, -99.16, -99.20, -98.39, np.nan, -99.82],
        [-99.09, -99.87, -98.35, -99.96, -99.64, -98.57, -99.04, -98.14, -99.47, -99.41],
    ],
    dtype=np.float32,
)


def new_peak_finder(**kwargs):
    "Create a new PeakFinder instance with DEFAULT_PARAMS and given kwargs"
    params = {**DEFAULT_PARAMS, **kwargs}  # kwargs takes precedence over defaults
    return PeakFinder(**params)


def run_output_comparison(fname_glob: str, expected_file_content: str):
    "Open the (expected unique) file matching given glob, and compare it with expected content."
    # NB: glob is here used because fnames first letters are often randomly generated (tempfile) in test cases
    fname_detected = glob.glob(fname_glob)
    if len(fname_detected) == 0:
        dirname = os.path.split(fname_glob)[0]
        dir_exists = "exists" if os.path.exists(dirname) else "does not exist either"
        raise FileNotFoundError(f'Glob "{fname_glob}" doesn\'t match any file (but directory {dirname} {dir_exists}).')
    assert len(fname_detected) == 1, "given glob matches multiple files"
    fname = next(iter(fname_detected))
    with open(fname, encoding="utf8") as fd:
        outdata = fd.read().strip()
        assert outdata == expected_file_content


def make_dtm():
    """
    Produce a the DTM and return its path.
    """
    return dtm_generator.make_dtm_with_elevations(
        (-8.7505, 49.0055),
        (-8.7415, 49.0145),
        ELEVATIONS,
    )


def generic_test_func(
    reffile: str, shape: Union[List[List[float]], str], expected_files_and_latlon: dict, peak_finder_kwargs: dict = None
) -> AssertionError:
    """This function generate a pytest-ready function testing that given
    reffile provides, when geofilter using given shapefile is used,
    the files provided as expected_files_and_latlon keys,
    with their value being the lat/lon expected to be found in those files.

    """
    output_dir = tmp.mktemp(suffix="_peak_detector_test")
    if isinstance(shape, (list, tuple)):  # must be a list of coordinates
        shapefile = create_kml(tmp.gettempdir(), {"zone": shape})
    else:  # is nothing
        shapefile = None
    # pick reference_file exact location
    if os.path.isabs(reffile):
        reference_file = reffile
    else:  # reference_file was given relative to the pyat test directory
        reference_file = directory.get_test_directory() + "/raw/" + reffile
    assert os.path.isfile(reference_file)

    # get shape_files as a liste of files
    shape_files = [shapefile] if shapefile else []
    assert not shapefile or os.path.isfile(shape_files[0])  # a given shapefile implies its existence

    if peak_finder_kwargs is not None:
        new_peak_finder(i_paths=reference_file, o_paths=output_dir, geo_masks=shape_files, **peak_finder_kwargs)()
    else:
        new_peak_finder(i_paths=reference_file, o_paths=output_dir, geo_masks=shape_files)()

    # now parse output file and check values
    outfiles = list(os.listdir(output_dir))
    assert len(outfiles) == len(expected_files_and_latlon), "Generated files are not those expected"
    for fname, expected_latlon in expected_files_and_latlon.items():
        run_output_comparison(os.path.join(output_dir, fname), FILL_HOLES_OUTDATA.format(**expected_latlon))
    shutil.rmtree(output_dir)
    if shapefile:
        os.remove(shapefile)


def test_fill_holes():
    generic_test_func("fill_holes.dtm.nc", None, {"fill_holes.dtm.nc_holes_0.csv": {"lat": "3.5", "lon": "1.5"}})


def test_fill_holes_with_shapefilter_1():
    generic_test_func(
        "fill_holes.dtm.nc",
        [
            [1.1361852156453431, 3.287055531804087],
            [1.1700759075386569, 2.922633716448478],
            [3.158852765399656, 2.939258630933021],
            [3.316601880538013, 3.3205356233632775],
            [2.880285588337804, 3.4858153795967657],
        ],
        {"fill_holes.dtm.nc_holes_0.csv": {"lat": "2.5", "lon": "1.5"}},
    )


def test_fill_holes_with_shapefilter_2():
    generic_test_func(
        "fill_holes.dtm.nc",
        [
            [1.7380237906866625, 4.790781137863862],
            [1.7867969532946466, 2.4502281207232843],
            [-1.3633255509186994, 2.4937567374786127],
            [-0.30396851854392004, 4.113957207194873],
        ],
        {"fill_holes.dtm.nc_holes_0.csv": {"lat": "4.166666666666667", "lon": "-0.16666666666666669"}},
    )


def test_fill_holes_with_shapefilter_3():
    generic_test_func(
        "fill_holes.dtm.nc",
        [
            [3.6814541457919403, 3.39470927779958],
            [3.652241007759311, 0.43608496701851507],
            [2.1356312849549757, 0.4843252759181203],
            [3.136294297106403, 2.1955521485238463],
        ],
        {"fill_holes.dtm.nc_holes_0.csv": {"lat": "2.5", "lon": "2.5"}},
    )


def test_fill_holes_with_shapefilter_4():
    generic_test_func(
        "fill_holes.dtm.nc",
        [
            [3.573965984676826, 0.2800898164759222],
            [0.5765260404706954, 0.22322030215254873],
            [0.42983201839685947, 3.6266326936977276],
            [2.9616729784053755, 2.5583280597742926],
            [3.0956100973813405, 1.8866076131052776],
        ],
        {"fill_holes.dtm.nc_holes_0.csv": {"lat": "3.5", "lon": "2.0"}},
    )


def test_artificial_dtm():
    dtm_path = make_dtm()
    generic_test_func(
        dtm_path,
        None,
        {
            "*.dtm.nc_gradient_detection_0.csv": {"lat": "49.01040540540541", "lon": "-8.746581081081082"},
            "*.dtm.nc_holes_0.csv": {"lat": "49.014", "lon": "-8.743"},
            "*.dtm.nc_holes_1.csv": {"lat": "49.008", "lon": "-8.749"},
        },
        {"use_gradient": True, "maximum_hole_area_in_pixel": 2},
    )
    os.remove(dtm_path)


def test_artificial_dtm_with_shapefilter():
    dtm_path = make_dtm()
    generic_test_func(
        dtm_path,
        [
            [-8.754284227185007, 49.01153781024292],
            [-8.738155709103046, 49.0116731123702],
            [-8.737921950278823, 49.0044586439892],
            [-8.752025129490717, 49.00436207140577],
        ],
        {
            # Expecting 1 peak detection using gradient method
            # Expecting 1 hole detection using holes detection method
            # So, out folder must contain 2 files
            "*.dtm.nc_gradient_detection_0.csv": {"lat": "49.00847368421053", "lon": "-8.746494736842106"},
            "*.dtm.nc_holes_0.csv": {"lat": "49.008", "lon": "-8.749"},
        },
        # maximum hole area is set (to 2, because the maximal value is exclusive),
        #  because by default its 100, way above the size of the geo mask,
        #  hence the mask itself was considered a hole.
        {"use_gradient": True, "maximum_hole_area_in_pixel": 2},
    )
    os.remove(dtm_path)
