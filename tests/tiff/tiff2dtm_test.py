#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile as tmp

import numpy as np
import pytest

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DTM
from pyat.dtm.convert.gdal_raster_to_dtm import export_gdal_raster_to_dtm
from tests.generator.tiff_generator import generate_tiff


def test_180th_tiff_to_dtm():
    """
    Convert a LonLat tiff to a LonLat DTM.
    Raster is spanning the 180th meridian
    """
    try:
        # Generations from -90 to -100
        elevations = 10 * np.random.default_rng().random((10, 10)) - 100
        path_tiff = generate_tiff(
            left=179.997,
            top=-20.0150,
            proj4="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
            resolution=0.001,
            data=elevations,
        )
        path_dtm = tmp.mktemp(suffix=".dtm.nc")
        export_gdal_raster_to_dtm(
            i_paths=[path_tiff],
            o_paths=[path_dtm],
            title="TTitle",
            institution="IInstitution",
            source="SSource",
            references="RReferences",
            comment="CComment",
            remove_invalid_elevation=False
        )

        # Open DTM
        with dtm_driver.open_dtm(path_dtm) as dtm:
            # Check Metadata
            assert dtm.dataset.title == "TTitle"
            assert dtm.dataset.institution == "IInstitution"
            assert dtm.dataset.source == "SSource"
            assert dtm.dataset.references == "RReferences"
            assert dtm.dataset.title == "TTitle"
            assert dtm.dataset.comment == "CComment"

            # Check grid size. Same as Tiff because no wrap has been processed
            assert dtm.dtm_file.row_count == 10
            assert dtm.dtm_file.col_count == 10

            # Check projection : lonlat
            assert dtm.dtm_file.spatial_reference.IsGeographic()

            # Check GeoBox. Must be the Geobox specified in coord argument
            assert dtm.dtm_file.north == pytest.approx(-20.015)
            assert dtm.dtm_file.south == pytest.approx(-20.025)
            assert dtm.dtm_file.west == pytest.approx(179.997)
            assert dtm.dtm_file.east == pytest.approx(-179.993)

            # Check elevations at the corners
            dtm_elevations = dtm[DTM.ELEVATION_NAME][:]
            # Tiff and DTM have no the same origin, so DTM[0,0] is TIFF[-1, 0]
            assert dtm_elevations[0, 0] == pytest.approx(elevations[-1, 0])
            assert dtm_elevations[0, -1] == pytest.approx(elevations[-1, -1])
            assert dtm_elevations[-1, 0] == pytest.approx(elevations[0, 0])
            assert dtm_elevations[-1, -1] == pytest.approx(elevations[0, -1])

    finally:
        os.remove(path_tiff)
        os.remove(path_dtm)
