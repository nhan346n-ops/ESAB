import os
import tempfile as tmp

import numpy as np
from osgeo import gdal, osr
from pygws.service.progress_monitor import DefaultMonitor

from pyat.tiff import tiff_gridder
from pyat.utils import argument_utils


def generate_tiff_gridder(path_tiff):
    coord = {
        "north": -12.000,
        "south": -13.000,
        "west": 45.000,
        "east": 46.000,
    }

    geobox = argument_utils.parse_geobox("coord", coord)
    geobox.spatial_reference = osr.SpatialReference()
    geobox.spatial_reference.ImportFromProj4("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    grid = tiff_gridder.TiffGridder(
        tiff_path=path_tiff,
        geobox=geobox,
        spatial_resolution=0.25,
        monitor=DefaultMonitor,
    )
    return grid


def test_grid():
    with tmp.TemporaryDirectory() as temp_dir:
        try:
            longitudes = np.array([45.000, 45.100, 45.200, 45.300, 45.400, 45.500, 45.600, 45.700, 45.800, 45.900])
            latitudes = np.array(
                [-12.000, -12.100, -12.200, -12.300, -12.400, -12.500, -12.600, -12.700, -12.800, -12.900]
            )
            echos = np.array([-500.0, -40.0, -30.0, -20.0, -10.0, -0.0, 10.0, 20.0, 30.0, 400.0])
            path_tiff = tmp.mktemp(suffix=".tiff", dir=temp_dir)
            grid = generate_tiff_gridder(path_tiff)
            grid.initialize_tiff_file(float)
            columns, rows = grid.project_coords(longitudes, latitudes)

            assert (columns == [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]).all()
            assert (rows == [4, 3, 3, 2, 2, 2, 1, 1, 0, 0]).all()
            assert columns.size == rows.size == 10
            grid.grid_average(columns, rows, echos)
            grid.finalize_tiff()
            dataset = gdal.Open(path_tiff, gdal.GA_ReadOnly)
            for x in range(1, dataset.RasterCount + 1):
                band = dataset.GetRasterBand(x)
                array = band.ReadAsArray()
            assert array[0][0] == -35.0
            assert array[1][1] == -15.0
            assert array[1][2] == -0.0
            assert array[2][2] == 15.0
            assert array[3][3] == 215.0
        finally:
            dataset = None
            os.remove(path_tiff)
