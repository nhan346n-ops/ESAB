#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile as tmp
import unittest

import numpy as np
from osgeo import osr

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DtmConstants
from pyat.dtm.convert.gridded_csv_to_dtm import GriddedCsvToDtm

SR_MERCATOR = osr.SpatialReference()
SR_MERCATOR.ImportFromProj4("+proj=merc +lon_0=0 +lat_ts=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


class TestGriddedCsv2Dtm(unittest.TestCase):
    def test_csv_lat_lon_export(self):
        """
        Convert a CSV (Emo format) to DTM
        """
        # CSV file with 4 lines : NW, NE, SW, SE
        # SE is duplicate to test mean elevations ()
        csv_content = """ Longitude;Latitude;Min Elev;Max Elev;Elevation;Std dev;Test_int;Interpol;Smooth Elev;Fake;CDI
                          -9.6838  ;52.4911 ;3411.01 ;3411.21 ;3411.11  ;1.60   ;1       ;5       ;3447.22    ;3.58;CDI_1
                          -9.6828  ;52.4911 ;3422.02 ;3422.22 ;3422.12  ;1.70   ;2       ;6       ;3445.80    ;====;CDI_2
                          -9.6838  ;52.4901 ;3433.03 ;3433.23 ;3433.13  ;1.80   ;3       ;7       ;3445.24    ;Fake;
                          -9.6828  ;52.4901 ;9999.99 ;9999.99 ;9999.99  ;9.99   ;9       ;1       ;9999.99    ;9999;
                          -9.6828  ;52.4901 ;3444.04 ;3444.24 ;3444.14  ;1.90   ;4       ;8       ;3442.93    ;    ;CDI_4
            """
        try:
            path_csv = tmp.mktemp(suffix=".csv")
            path_dtm = tmp.mktemp(suffix=".nc.dtm")
            with open(path_csv, "w", encoding="utf-8") as csv_file:
                csv_file.write(csv_content.replace(" ", ""))

            # Export CSV -> DTM
            exporter = GriddedCsvToDtm(
                i_paths=[path_csv],
                o_paths=[path_dtm],
                indexes={
                    "Longitude/X": "0",
                    "Latitude/Y": "1",
                    "Min elevation": "2",
                    "Max elevation": "3",
                    "Elevation": "4",
                    "Std dev": "5",
                    "Test_int": "6",
                    "Interpolation flag": "7",
                    "Elevation smoothed": "8",
                    "CDI": "10",
                },
                headers_types={
                    "Longitude/X": "float",
                    "Latitude/Y": "float",
                    "Min elevation": "float",
                    "Max elevation": "float",
                    "Elevation": "float",
                    "Std dev": "float",
                    "Test_int": "int",
                    "Interpolation flag": "int",
                    "Elevation smoothed": "float",
                    "CDI": "str",
                },
                delimiter=";",
                skip_rows=1,
                depth_sign=-1.0,
                recompute_geobox=True,
                target_resolution=0.001,
                auto_rounding_arcmin=False,
                allow_undefined_cdi=True,
            )
            exporter()

            # Open DTM
            with dtm_driver.open_dtm(path_dtm) as dtm:
                # Check projection : lonlat
                assert dtm.dtm_file.spatial_reference.IsGeographic()

                # Check elevation => cell filled with the mean value
                assert np.array_equal(
                    dtm[DtmConstants.ELEVATION_NAME][:],
                    np.array([[-3433.13, -3444.14], [-3411.11, -3422.12]], dtype=np.float32),
                )
                # Check min elevation => cell filled with the last value
                assert np.array_equal(
                    dtm[DtmConstants.ELEVATION_MIN][:],
                    np.array([[-3433.03, -3444.04], [-3411.01, -3422.02]], dtype=np.float32),
                )
                # Check max elevation => cell filled with the last value
                assert np.array_equal(
                    dtm[DtmConstants.ELEVATION_MAX][:],
                    np.array([[-3433.23, -3444.24], [-3411.21, -3422.22]], dtype=np.float32),
                )
                # Check Standard deviation => cell filled with the last value
                assert np.array_equal(
                    dtm[DtmConstants.STDEV][:],
                    np.array([[1.8, 1.9], [1.6, 1.7]], dtype=np.float32),
                )
                # Check the additional int layer
                assert np.array_equal(
                    dtm["Test_int"][:],
                    np.array([[3, 4], [1, 2]], dtype=np.int32),
                )
                # Check interpolation flag => cell filled with the last value
                assert np.array_equal(
                    dtm[DtmConstants.INTERPOLATION_FLAG][:],
                    np.array([[7, 8], [5, 6]], dtype=np.int32),
                )
                # Check smoothed elevation => cell filled with the last value
                assert np.array_equal(
                    dtm[DtmConstants.ELEVATION_SMOOTHED_NAME][:],
                    np.array([[-3445.24, -3442.93], [-3447.22, -3445.80]], dtype=np.float32),
                )
                # Check CDI. No CDI in first cell
                assert np.array_equal(
                    np.ma.getmask(dtm[DtmConstants.CDI_INDEX][:]), np.array([[True, False], [False, False]])
                )
        finally:
            os.remove(path_csv)
            os.remove(path_dtm)

    def test_csv_lat_lon_spanning_180th_export(self):
        """
        Convert a CSV (XYZ format) to DTM. Coordinates span the 180th meridian
        """
        # CSV file with 4 lines : NW, NE, SW, SE
        csv_content = """  179,5   ;52,5 ;-3411,11
                          -179,5   ;52,5 ;-3422,12
                           179,5   ;51,5 ;-3433,13
                          -179,5   ;51,5 ;-3444,14
            """
        try:
            path_csv = tmp.mktemp(suffix=".csv")
            path_dtm = tmp.mktemp(suffix=".nc.dtm")
            with open(path_csv, "w", encoding="utf-8") as csv_file:
                csv_file.write(csv_content.replace(" ", ""))

            # Export CSV -> DTM
            exporter = GriddedCsvToDtm(
                i_paths=[path_csv],
                o_paths=[path_dtm],
                indexes={"Longitude/X": "0", "Latitude/Y": "1", "Elevation": "2"},
                headers_types={"Longitude/X": "float", "Latitude/Y": "float", "Elevation": "float"},
                decimal_point=",",
                recompute_geobox=True,
                target_resolution=1,
                auto_rounding_arcmin=False,
            )
            exporter()

            # Open DTM
            with dtm_driver.open_dtm(path_dtm) as dtm:
                # Check projection : lonlat
                assert dtm.dtm_file.spatial_reference.IsGeographic()

                # Check elevation => cell filled with the mean value
                assert np.array_equal(
                    dtm[DtmConstants.ELEVATION_NAME][:],
                    np.array([[-3433.13, -3444.14], [-3411.11, -3422.12]], dtype=np.float32),
                )

        finally:
            os.remove(path_csv)
            os.remove(path_dtm)

    def test_csv_mercator_export(self):
        """
        Convert a CSV with mercator coordinates to DTM
        """
        # CSV file with 4 lines : NW, NE, SW, SE
        csv_content = """ X        ,Y       ,Min Elev,Max Elev,Elevation
                          -9.6838  ,52.4911 ,-3411.01 ,-3411.21 ,-3411.11
                          -9.6828  ,52.4911 ,-3422.02 ,-3422.22 ,-3422.12
                          -9.6838  ,52.4901 ,-3433.03 ,-3433.23 ,-3433.13
                          -9.6828  ,52.4901 ,-3444.04 ,-3444.24 ,-3444.14
            """
        try:
            path_csv = tmp.mktemp(suffix=".csv")
            path_dtm = tmp.mktemp(suffix=".dtm.nc")
            with open(path_csv, "w", encoding="utf-8") as csv_file:
                csv_file.write(csv_content.replace(" ", ""))

            # Export CSV -> DTM
            exporter = GriddedCsvToDtm(
                i_paths=[path_csv],
                spatial_reference=SR_MERCATOR.ExportToProj4(),
                o_paths=[path_dtm],
                indexes={
                    "Longitude/X": "0",
                    "Latitude/Y": "1",
                    "Min elevation": "2",
                    "Max elevation": "3",
                    "Elevation": "4",
                },
                headers_types={
                    "Longitude/X": "float",
                    "Latitude/Y": "float",
                    "Min elevation": "float",
                    "Max elevation": "float",
                    "Elevation": "float",
                },
                delimiter=",",
                skip_rows=1,
                depth_sign=1.0,
                recompute_geobox=True,
                target_resolution=0.001,
                auto_rounding_arcmin=False,
            )
            exporter()

            # Open DTM
            with dtm_driver.open_dtm(path_dtm) as dtm:
                # Check projection
                assert dtm.dtm_file.spatial_reference.IsProjected()
                assert dtm.dtm_file.spatial_reference.IsSame(SR_MERCATOR)
                # Check elevation
                assert np.array_equal(
                    dtm[DtmConstants.ELEVATION_NAME][:],
                    np.array([[-3433.13, -3444.14], [-3411.11, -3422.12]], dtype=np.float32),
                )
                # Check min elevation
                assert np.array_equal(
                    dtm[DtmConstants.ELEVATION_MIN][:],
                    np.array([[-3433.03, -3444.04], [-3411.01, -3422.02]], dtype=np.float32),
                )
                # Check max elevation
                assert np.array_equal(
                    dtm[DtmConstants.ELEVATION_MAX][:],
                    np.array([[-3433.23, -3444.24], [-3411.21, -3422.22]], dtype=np.float32),
                )
        finally:
            os.remove(path_csv)
            os.remove(path_dtm)


if __name__ == "__main__":
    unittest.main()
