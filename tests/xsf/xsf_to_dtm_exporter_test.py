#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile

from osgeo import osr

import pyat.common.geo_file as gf
import pyat.dtm.dtm_standard_constants as DtmConstants
from pyat.sounder.sounder_to_dtm import SounderToDtmExporter
from pyat.dtm import dtm_driver
from tests.generator.xsf_generator import XsfGenerator


def generate_xsf(folder: str) -> str:
    """
    Creates a XSF file with
        2 navigation positions :
            W 4.005 / N 48.0
            W 4.0 / N 48.005
        4 beams on both sides of the 2 navigation positions :
            BEAM1 -> W 4.006669079479038 / N 48.00075171097127, depth = 10m
            BEAM2 -> W 4.003330969015191 / N 47.99924826467987, depth = 20m
            BEAM3 -> W 4.001669240784506 / N 48.00575171031162, depth = 10m
            BEAM4 -> W 3.998330807722907 / N 48.004248265335306, depth = 20m
    """
    generator = XsfGenerator(folder)
    return generator.initialize_file(
        latitude_min_deg=48.0,
        latitude_max_deg=48.005,
        longitude_min_deg=-4.005,
        longitude_max_deg=-4.0,
        ping_count=2,
        beam_count=2,
        min_depth_m=10.0,
        max_depth_m=20.0,
    )


def generate_xsf_2(folder: str) -> str:
    """
    Creates a XSF file with
        2 navigation positions :
            W 3.995 / N 48.0
            W 3.990 / N 48.005
    """
    generator = XsfGenerator(folder)
    return generator.initialize_file(
        latitude_min_deg=48.0,
        latitude_max_deg=48.005,
        longitude_min_deg=-3.995,
        longitude_max_deg=-3.990,
        ping_count=2,
        beam_count=2,
        min_depth_m=10.0,
        max_depth_m=20.0,
    )


def test_convert_xsf_wsg84_dtm(gap_filling: bool = False):
    """
    Converts a xsf to DTM format in a lonlat projection
    """
    with tempfile.TemporaryDirectory() as o_path:
        o_dtm_path = tempfile.mktemp(suffix=".nc", dir=o_path)
        xsf_path = generate_xsf(o_path)

        # Export to a DTM
        # Spatial resolution : 8 arc second
        cell_size = (1.0 / 3600.0) * 8.0
        # Grid origin (Shifting the center of the first cell) :  W 4.007° / N 47.999°
        lon_0 = -4.007 - cell_size / 2
        lat_0 = 47.999 - cell_size / 2

        export_xsf_to_dtm(xsf_path, o_dtm_path, gf.SR_WGS_84, lon_0, lat_0, cell_size, gap_filling)

        # Expected elevation grid :
        #  +----------+----------+----------+----------+----------+----------|
        #  | 48.00788 |          |          |          |          |          |
        #  +----------+----------+----------+----------+----------+----------|
        #  | 48.00566 |          |          |   BEAM3  |          |          |
        #  +----------+----------+----------+----------+----------+----------|
        #  | 48.00344 |          |          |          |          |  BEAM4   |
        #  +----------+----------+----------+----------+----------+----------|
        #  | 48.00122 |   BEAM1  |          |          |          |          |
        #  +----------+----------+----------+----------+----------+----------|
        #  | 47.999   |          |          |   BEAM2  |          |          |
        #  +----------+----------+----------+----------+----------+----------|
        #  |          |  -4.007  | -4.00477 | -4.00255 | -4.00033 | -3.99811 |
        #  +----------+----------+----------+----------+----------+----------|

        # update netcdf file with the results
        with dtm_driver.open_dtm(o_dtm_path, mode="r+") as i_dtm_driver:
            # Check grid size
            assert i_dtm_driver.dtm_file.row_count == 5
            assert i_dtm_driver.dtm_file.col_count == 5
            # Check BEAM1
            assert i_dtm_driver[DtmConstants.ELEVATION_NAME][1, 0] == -10.0
            # Check BEAM2
            assert i_dtm_driver[DtmConstants.ELEVATION_NAME][0, 2] == -20.0
            # Check BEAM3
            assert i_dtm_driver[DtmConstants.ELEVATION_NAME][3, 2] == -10.0
            # Check BEAM4
            assert i_dtm_driver[DtmConstants.ELEVATION_NAME][2, 4] == -20.0


def test_convert_xsf_wsg84_dtm_with_gap_filling():
    test_convert_xsf_wsg84_dtm(gap_filling=True)


def test_merge_xsf_to_dtm(gap_filling: bool = False):
    """
    Converts a 2 xsf files to DTM format in a lonlat projection
    """
    with tempfile.TemporaryDirectory() as o_path:
        o_dtm_path = tempfile.mktemp(suffix=".dtm.nc", dir=o_path)
        xsf_path_1 = generate_xsf(o_path)
        xsf_path_2 = generate_xsf_2(o_path)

        exporter = SounderToDtmExporter(
            i_paths=[xsf_path_1, xsf_path_2],
            o_paths=[o_dtm_path],
            target_resolution=0.002,
            target_spatial_reference="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
            coord={
                "north": 48.007,
                "south": 47.999,
                "west": -4.007,
                "east": -3.987,
            },
            cdi={os.path.basename(xsf_path_1): "CDI1", os.path.basename(xsf_path_2): "CDI2"},
            gap_filling=gap_filling
        )
        exporter()

        with dtm_driver.open_dtm(o_dtm_path, mode="r+") as i_dtm_driver:
            # -4.006° / 48.0° from xsf_path_1 => CDI1
            assert i_dtm_driver[DtmConstants.VALUE_COUNT][0, 0] == 1
            assert i_dtm_driver[DtmConstants.CDI_INDEX][0, 0] == 0
            assert i_dtm_driver[DtmConstants.CDI][0] == "CDI1"
            # -3.996° / 48.0° from xsf_path_2 => CDI2
            assert i_dtm_driver[DtmConstants.VALUE_COUNT][0, 5] == 1
            assert i_dtm_driver[DtmConstants.CDI_INDEX][0, 5] == 1
            assert i_dtm_driver[DtmConstants.CDI][1] == "CDI2"


def test_merge_xsf_to_dtm_with_gap_filling():
    test_merge_xsf_to_dtm(gap_filling=True)


def test_convert_xsf_to_eqc_dtm():
    """
    Converts a xsf to DTM format in a Equidistant Cylindrical projection
    """
    with tempfile.TemporaryDirectory() as o_path:
        o_dtm_path = tempfile.mktemp(suffix=".nc", dir=o_path)
        xsf_path = generate_xsf(o_path)

        # Best mercator projection (found using gdalwarp)
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4(
            "+proj=tmerc +lat_0=47.999 +lon_0=-4.007 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
        )
        # Spatial resolution : 170.0 m
        cell_size = 170.0
        # Shift the grid origin to have the center of the first cell at (0, 0).
        x_0 = -cell_size / 2
        y_0 = -cell_size / 2

        export_xsf_to_dtm(xsf_path, o_dtm_path, spatial_reference, x_0, y_0, cell_size)

        # Expected elevation grid :
        #  +-----+----------+----------+----------+----------+----------|
        #  | 680 |          |          |   BEAM3  |          |          |
        #  +-----+----------+----------+----------+----------+----------|
        #  | 510 |          |          |          |          |   BEAM4  |
        #  +-----+----------+----------+----------+----------+----------|
        #  | 340 |          |          |          |          |          |
        #  +-----+----------+----------+----------+----------+----------|
        #  | 170 |   BEAM1  |          |          |          |          |
        #  +-----+----------+----------+----------+----------+----------|
        #  | 0   |          |          |   BEAM2  |          |          |
        #  +-----+----------+----------+----------+----------+----------|
        #  |     |     0    |    170   |    340   |    510   |    680   |
        #  +-----+----------+----------+----------+----------+----------|

        # update netcdf file with the results
        with dtm_driver.open_dtm(o_dtm_path, mode="r+") as i_dtm_driver:
            # Check grid size
            assert i_dtm_driver.dtm_file.row_count == 5
            assert i_dtm_driver.dtm_file.col_count == 5
            # Check BEAM1
            assert i_dtm_driver[DtmConstants.ELEVATION_NAME][1, 0] == -10.0
            # Check BEAM2
            assert i_dtm_driver[DtmConstants.ELEVATION_NAME][0, 2] == -20.0
            # Check BEAM3
            assert i_dtm_driver[DtmConstants.ELEVATION_NAME][4, 2] == -10.0
            # Check BEAM4
            assert i_dtm_driver[DtmConstants.ELEVATION_NAME][3, 4] == -20.0


def export_xsf_to_dtm(
    xsf_path: str,
    o_dtm_path: str,
    spatial_reference: osr.SpatialReference,
    lon_0: float,
    lat_0: float,
    cell_size: float,
    gap_filling: bool = False
):
    """
    Perform the conversion to DTM
    """
    print("\nConverting ", xsf_path, " to ", o_dtm_path)
    # Grid size : 5x5
    grid_size = 5
    exporter = SounderToDtmExporter(
        i_paths=[xsf_path],
        o_paths=[o_dtm_path],
        target_resolution=cell_size,
        target_spatial_reference=spatial_reference.ExportToProj4(),
        coord={
            "north": lat_0 + cell_size * grid_size,
            "south": lat_0,
            "west": lon_0,
            "east": lon_0 + cell_size * grid_size,
        },
        layers=[
            DtmConstants.ELEVATION_NAME,
            DtmConstants.ELEVATION_MIN,
            DtmConstants.ELEVATION_MAX,
            DtmConstants.STDEV,
            DtmConstants.BACKSCATTER,
            DtmConstants.MIN_ACROSS_DISTANCE,
            DtmConstants.MAX_ACROSS_DISTANCE,
            DtmConstants.MAX_ACCROSS_ANGLE,
        ],
        gap_filling=gap_filling,
    )
    exporter()


if __name__ == "__main__":
    # Generates XSF test files
    print(generate_xsf("E:/temp"))
    print(generate_xsf_2("E:/temp"))
