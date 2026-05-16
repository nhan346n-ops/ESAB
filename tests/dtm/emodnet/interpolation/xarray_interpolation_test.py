# pylint:disable=no-member

import tempfile as tmp

import numpy as np
from pygws.service.progress_monitor import DefaultMonitor

import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.transform.interpolation.xarray_interpolate as interpolateProcess
import tests.generator.dtm_generator as dtm_generator
from pyat.dtm import dtm_driver


def make_dtm(temp_dir: str) -> str:
    """
    Generate a DTM with no elevation at the center (16 cells)
            -10°                 -9°
              |                   |
         49°  +-------------------+
              |Elev=[-900, -1000] |
              |    +---------+    |
              |    |   NaN   |    |
              |    +---------+    |
              |                   |
         48°  +-------------------+
    """
    elevations = 100 * np.random.default_rng().random((10, 10)) - 1000
    row, col = np.indices((4, 4))
    elevations[row + 3, col + 3] = dtm_driver.get_missing_value(DtmConstants.ELEVATION_NAME)
    value_count = np.full_like(elevations, 2, dtype=dtm_driver.get_type(DtmConstants.VALUE_COUNT))
    value_count[row + 3, col + 3] = dtm_driver.get_missing_value(DtmConstants.VALUE_COUNT)
    # Nb of missing values in value_count :
    unique, counts = np.unique(value_count, return_counts=True)
    frequencies = dict(zip(unique, counts))
    assert frequencies[dtm_driver.get_missing_value(DtmConstants.VALUE_COUNT)] == 16
    assert frequencies[2] == 84

    # Set a cdi to 2 where cells are empty
    cdi_index = np.ones_like(elevations, dtype=dtm_driver.get_type(DtmConstants.CDI_INDEX))
    cdi_index[row + 3, col + 3] = 2

    return dtm_generator.make_dtm_with_data(
        (-9.0, 49.0),
        (-10, 48.0),
        {
            DtmConstants.ELEVATION_NAME: elevations,
            DtmConstants.VALUE_COUNT: value_count,
            DtmConstants.CDI_INDEX: cdi_index,
        },
        temp_dir,
    )


def test_interpolation():
    """
    test the interpolation function
    """
    with tmp.TemporaryDirectory() as temp_dir:
        path_dtm = make_dtm(temp_dir)
        path_o_dtm = tmp.mktemp(suffix=".dtm.nc", dir=temp_dir)
        interpolateProcess.InterpolateProcess(
            i_paths=[path_dtm],
            o_paths=[path_o_dtm],
            overwrite=True,
            monitor=DefaultMonitor,
        )

        with dtm_driver.open_dtm(path_o_dtm) as o_dtm_driver:
            elevation = o_dtm_driver[DtmConstants.ELEVATION_NAME]
            assert np.count_nonzero(np.isnan(elevation)) == 0
