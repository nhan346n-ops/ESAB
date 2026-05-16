#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile as tmp

import numpy as np
import pytest
from osgeo import osr

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DTM
from pyat.dtm.convert.csv_to_dtm import CsvToDtm

SR_MERCATOR = osr.SpatialReference()
SR_MERCATOR.ImportFromProj4("+proj=merc +lon_0=0 +lat_ts=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def test_csv_lat_lon_export():
    """
    Convert a CSV (Emo format) to DTM
    """
    # CSV file with 4 lines : NW, NE, SW, SE
    # SE is duplicate to test mean elevations ()
    csv_content = """   Longitude;Latitude;Min Elev;Max Elev;Elevation;Std dev;Test_int;Interpol;Smooth Elev;Fake;CDI
                        -9.6838  ;52.4911 ;3411.01 ;3411.21 ;3411.11  ;1.60   ;1       ;5       ;3447.22    ;3.58;CDI_1
                        -9.6828  ;52.4911 ;3422.02 ;3422.22 ;3422.12  ;1.70   ;2       ;6       ;3445.80    ;====;CDI_2
                        -9.6838  ;52.4901 ;3433.03 ;3433.23 ;3433.13  ;1.80   ;3       ;7       ;3445.24    ;Fake;
                        -9.6828  ;52.4901 ;9999.99 ;9999.99 ;9999.99  ;9.99   ;9       ;1       ;9999.99    ;9999;
                        -9.6828  ;52.4901 ;3444.04 ;3444.24 ;3444.14  ;1.90   ;4       ;8       ;3442.93    ;    ;CDI_4
        """
    with tmp.TemporaryDirectory() as o_path:
        path_csv = tmp.mktemp(dir=o_path, suffix=".csv")
        path_dtm = tmp.mktemp(dir=o_path, suffix=".nc.dtm")
        with open(path_csv, "wt", encoding="utf8") as csv_file:
            csv_file.write(csv_content.replace(" ", ""))

        # Export CSV -> DTM
        exporter = CsvToDtm(
            i_paths=[path_csv],
            o_paths=[path_dtm],
            coord={
                "north": 52.4920,
                "south": 52.4900,
                "west": -9.6840,
                "east": -9.6820,
            },
            target_resolution=0.001,
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
            title="TTitle",
            institution="IInstitution",
            source="SSource",
            references="RReferences",
            comment="CComment",
            allow_undefined_cdi=True,
        )
        exporter()

        # Open DTM
        with dtm_driver.open_dtm(path_dtm) as dtm:
            # Check Metadata
            assert dtm.dataset.title == "TTitle"
            assert dtm.dataset.institution == "IInstitution"
            assert dtm.dataset.source == "SSource"
            assert dtm.dataset.references == "RReferences"
            assert dtm.dataset.title == "TTitle"
            assert dtm.dataset.comment == "CComment"

            # Check grid size
            assert dtm.dtm_file.row_count == 2
            assert dtm.dtm_file.col_count == 2

            # Check projection : lonlat
            assert dtm.dtm_file.spatial_reference.IsGeographic()

            # Check GeoBox. Must be the Geobox specified in coord argument
            assert dtm.dtm_file.north == pytest.approx(52.4920)
            assert dtm.dtm_file.south == pytest.approx(52.4900)
            assert dtm.dtm_file.west == pytest.approx(-9.6840)
            assert dtm.dtm_file.east == pytest.approx(-9.6820)

            # Check elevation => cell filled with the mean value
            assert np.array_equal(
                dtm[DTM.ELEVATION_NAME][:],
                np.array([[-3433.13, (-3444.14 - 9999.99) / 2], [-3411.11, -3422.12]], dtype=np.float32),
            )
            # Check min elevation => cell filled with the last value
            assert np.array_equal(
                dtm[DTM.ELEVATION_MIN][:],
                np.array([[-3433.03, -3444.04], [-3411.01, -3422.02]], dtype=np.float32),
            )
            # Check max elevation => cell filled with the last value
            assert np.array_equal(
                dtm[DTM.ELEVATION_MAX][:],
                np.array([[-3433.23, -3444.24], [-3411.21, -3422.22]], dtype=np.float32),
            )
            # Check Standard deviation => cell filled with the last value
            assert np.array_equal(
                dtm[DTM.STDEV][:],
                np.array([[1.8, 1.9], [1.6, 1.7]], dtype=np.float32),
            )
            # Check the additional int layer
            assert np.array_equal(
                dtm["Test_int"][:],
                np.array([[3, 4], [1, 2]], dtype=np.int32),
            )
            # Check interpolation flag => cell filled with the last value
            assert np.array_equal(
                dtm[DTM.INTERPOLATION_FLAG][:],
                np.array([[7, 8], [5, 6]], dtype=np.int32),
            )
            # Check smoothed elevation => cell filled with the last value
            assert np.array_equal(
                dtm[DTM.ELEVATION_SMOOTHED_NAME][:],
                np.array([[-3445.24, -3442.93], [-3447.22, -3445.80]], dtype=np.float32),
            )
            # Check CDI. No CDI in first cell
            assert np.array_equal(np.ma.getmask(dtm[DTM.CDI_INDEX][:]), np.array([[True, False], [False, False]]))

            # Check value_count not present
            assert DTM.VALUE_COUNT not in dtm


def test_csv_lat_lon_spanning_180th_export():
    """
    Convert a CSV (XYZ format) to DTM. Coordinates span the 180th meridian
    """
    # CSV file with 4 lines : NW, NE, SW, SE
    csv_content = """  179,5   ;52,5 ;-3411,11
                        -179,5   ;52,5 ;-3422,12
                        179,5   ;51,5 ;-3433,13
                        -179,5   ;51,5 ;-3444,14
        """
    with tmp.TemporaryDirectory() as o_path:
        path_csv = tmp.mktemp(dir=o_path, suffix=".csv")
        path_dtm = tmp.mktemp(dir=o_path, suffix=".nc.dtm")
        with open(path_csv, "wt", encoding="utf8") as csv_file:
            csv_file.write(csv_content.replace(" ", ""))

        # Export CSV -> DTM
        exporter = CsvToDtm(
            i_paths=[path_csv],
            o_paths=[path_dtm],
            coord={"north": 53.0, "south": 51.0, "west": 179.0, "east": -179.0},
            target_resolution=1.0,
            indexes={"Longitude/X": "0", "Latitude/Y": "1", "Elevation": "2"},
            headers_types={"Longitude/X": "float", "Latitude/Y": "float", "Elevation": "float"},
            decimal_point=",",
        )
        exporter()

        # Open DTM
        with dtm_driver.open_dtm(path_dtm) as dtm:
            # Check grid size
            assert dtm.dtm_file.row_count == 2
            assert dtm.dtm_file.col_count == 2

            # Check projection : lonlat
            assert dtm.dtm_file.spatial_reference.IsGeographic()

            # Check GeoBox. Must be the Geobox specified in coord argument
            assert dtm.dtm_file.north == pytest.approx(53.0)
            assert dtm.dtm_file.south == pytest.approx(51.0)
            assert dtm.dtm_file.west == pytest.approx(179.0)
            assert dtm.dtm_file.east == pytest.approx(-179.0)

            # Check Lon / Lat
            assert np.allclose(dtm.get_x_axis()[:], np.array([179.5, -179.5], dtype=np.float64))
            assert np.allclose(dtm.get_y_axis()[:], np.array([51.5, 52.5], dtype=np.float64))

            # Check elevation => cell filled with the mean value
            assert np.array_equal(
                dtm[DTM.ELEVATION_NAME][:],
                np.array([[-3433.13, -3444.14], [-3411.11, -3422.12]], dtype=np.float32),
            )


def test_csv_mercator_export():
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
    with tmp.TemporaryDirectory() as o_path:
        path_csv = tmp.mktemp(dir=o_path, suffix=".csv")
        path_dtm = tmp.mktemp(dir=o_path, suffix=".nc.dtm")
        with open(path_csv, "wt", encoding="utf8") as csv_file:
            csv_file.write(csv_content.replace(" ", ""))

        # Export CSV -> DTM
        exporter = CsvToDtm(
            i_paths=[path_csv],
            spatial_reference=SR_MERCATOR.ExportToProj4(),
            o_paths=[path_dtm],
            coord={
                "north": 52.4920,
                "south": 52.4900,
                "west": -9.6840,
                "east": -9.6820,
            },
            target_resolution=0.001,
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
        )
        exporter()

        # Open DTM
        with dtm_driver.open_dtm(path_dtm) as dtm:

            # Check grid size
            assert dtm.dtm_file.row_count == 2
            assert dtm.dtm_file.col_count == 2

            # Check projection
            assert dtm.dtm_file.spatial_reference.IsProjected()
            assert dtm.dtm_file.spatial_reference.IsSame(SR_MERCATOR)

            # Check GeoBox. Must be the Geobox specified in coord argument

            assert dtm.dtm_file.north == pytest.approx(52.4920)
            assert dtm.dtm_file.south == pytest.approx(52.4900)
            assert dtm.dtm_file.west == pytest.approx(-9.6840)
            assert dtm.dtm_file.east == pytest.approx(-9.6820)

            # Check X/Y
            assert np.allclose(dtm.get_x_axis()[:], np.array([-9.6835, -9.6825], dtype=np.float64))
            assert np.allclose(dtm.get_y_axis()[:], np.array([52.4905, 52.4915], dtype=np.float64))

            # Check elevation
            assert np.array_equal(
                dtm[DTM.ELEVATION_NAME][:],
                np.array([[-3433.13, -3444.14], [-3411.11, -3422.12]], dtype=np.float32),
            )
            # Check min elevation
            assert np.array_equal(
                dtm[DTM.ELEVATION_MIN][:],
                np.array([[-3433.03, -3444.04], [-3411.01, -3422.02]], dtype=np.float32),
            )
            # Check max elevation
            assert np.array_equal(
                dtm[DTM.ELEVATION_MAX][:],
                np.array([[-3433.23, -3444.24], [-3411.21, -3422.22]], dtype=np.float32),
            )


def test_stddev_computation():
    """
    Convert a CSV (Emo format) to DTM and check auto computed layers
    """
    # CSV file with 7 lines : NW, NE, SW and 4 SE
    csv_content = """   Longitude;Latitude;Elevation
                        -9.6838  ;52.4911 ;3411.01
                        -9.6828  ;52.4911 ;3422.02
                        -9.6838  ;52.4901 ;3433.03
        """
    se_elevations = [3444.0, 3433.0, 3422.0, 3411.0]
    for elevations in se_elevations:
        csv_content = csv_content + "-9.6828  ;52.4901 ;" + str(elevations) + "\n"
    with tmp.TemporaryDirectory() as o_path:
        path_csv = tmp.mktemp(dir=o_path, suffix=".csv")
        path_dtm = tmp.mktemp(dir=o_path, suffix=".nc.dtm")
        with open(path_csv, "wt", encoding="utf8") as csv_file:
            csv_file.write(csv_content.replace(" ", ""))

        # Export CSV -> DTM
        exporter = CsvToDtm(
            i_paths=[path_csv],
            o_paths=[path_dtm],
            coord={"north": 52.4920, "south": 52.4900, "west": -9.6840, "east": -9.6820},
            target_resolution=0.001,
            indexes={"Longitude/X": "0", "Latitude/Y": "1", "Elevation": "2"},
            headers_types={"Longitude/X": "float", "Latitude/Y": "float", "Elevation": "float"},
            delimiter=";",
            skip_rows=1,
            auto_layers=["elevation_min", "elevation_max", "stdev", "value_count"],
        )
        exporter()

        # Open DTM
        with dtm_driver.open_dtm(path_dtm) as dtm:
            # Check grid size
            assert dtm.dtm_file.row_count == 2
            assert dtm.dtm_file.col_count == 2

            # Check elevation => [[SW, SE], [NW, NE]]
            assert np.array_equal(
                dtm[DTM.ELEVATION_NAME][:],
                np.array([[3433.03, np.mean(se_elevations)], [3411.01, 3422.02]], dtype=np.float32),
            )
            # Check min elevation => cell filled with the last value
            assert np.array_equal(
                dtm[DTM.ELEVATION_MIN][:],
                np.array([[3433.03, np.min(se_elevations)], [3411.01, 3422.02]], dtype=np.float32),
            )
            # Check max elevation => cell filled with the last value
            assert np.array_equal(
                dtm[DTM.ELEVATION_MAX][:],
                np.array([[3433.03, np.max(se_elevations)], [3411.01, 3422.02]], dtype=np.float32),
            )
            # Check Standard deviation => cell filled with the last value
            assert np.isclose(dtm[DTM.STDEV][0, 1], np.std(se_elevations, dtype=np.float32), atol=0.02)

            # Check value_count
            assert np.array_equal(
                dtm[DTM.VALUE_COUNT][:],
                np.array([[1, 4], [1, 1]], dtype=np.int32),
            )


def test_filtering():
    """
    Convert a CSV (Emo format) to DTM and check filtering
    """
    # CSV file with 4 lines : NW, NE, SW and 4 SE
    csv_content = """   Longitude;Latitude;Elevation
                        -9.6838  ;52.4911 ;3411.01
                        -9.6828  ;52.4911 ;3422.02
                        -9.6838  ;52.4901 ;3433.03
                        -9.6828  ;52.4901 ;3444.04
        """
    with tmp.TemporaryDirectory() as o_path:
        path_csv = tmp.mktemp(dir=o_path, suffix=".csv")
        path_dtm = tmp.mktemp(dir=o_path, suffix=".nc.dtm")
        with open(path_csv, "wt", encoding="utf8") as csv_file:
            csv_file.write(csv_content.replace(" ", ""))

        # Export CSV -> DTM
        exporter = CsvToDtm(
            i_paths=[path_csv],
            o_paths=[path_dtm],
            coord={"north": 52.4920, "south": 52.4900, "west": -9.6840, "east": -9.6820},
            target_resolution=0.001,
            indexes={"Longitude/X": "0", "Latitude/Y": "1", "Elevation": "2"},
            headers_types={"Longitude/X": "float", "Latitude/Y": "float", "Elevation": "float"},
            delimiter=";",
            skip_rows=1,
            auto_layers=["value_count"],
            min_elevation=3420.0,
            max_elevation=3440.0,
        )
        exporter()

        # Open DTM
        with dtm_driver.open_dtm(path_dtm) as dtm:
            # Check elevation => [[SW, SE], [NW, NE]]. NW, SE are filtered
            assert dtm[DTM.ELEVATION_NAME][0, 0] is not np.ma.masked
            assert dtm[DTM.ELEVATION_NAME][0, 1] is np.ma.masked
            assert dtm[DTM.ELEVATION_NAME][1, 0] is np.ma.masked
            assert dtm[DTM.ELEVATION_NAME][1, 1] is not np.ma.masked
            # Check value_count
            assert dtm[DTM.VALUE_COUNT][0, 0] == 1
            assert dtm[DTM.VALUE_COUNT][0, 1] is np.ma.masked
            assert dtm[DTM.VALUE_COUNT][1, 0] is np.ma.masked
            assert dtm[DTM.VALUE_COUNT][1, 1] == 1


def test_min_sounding():
    """
    Convert a CSV to DTM and check the filtering on min valid soundings
    """
    # CSV file with 7 lines : NW, NE, SW and 4 SE
    csv_content = """   Longitude;Latitude;Elevation
                        -9.6838  ;52.4911 ;3411.01
                        -9.6828  ;52.4911 ;3422.02
                        -9.6838  ;52.4901 ;3433.03
                        -9.6838  ;52.4901 ;3433.03
                        -9.6828  ;52.4901 ;3444.04
                        -9.6828  ;52.4901 ;3444.04
                        -9.6828  ;52.4901 ;3444.04
        """
    with tmp.TemporaryDirectory() as o_path:
        path_csv = tmp.mktemp(dir=o_path, suffix=".csv")
        path_dtm = tmp.mktemp(dir=o_path, suffix=".nc.dtm")
        with open(path_csv, "wt", encoding="utf8") as csv_file:
            csv_file.write(csv_content.replace(" ", ""))

        # Export CSV -> DTM
        exporter = CsvToDtm(
            i_paths=[path_csv],
            o_paths=[path_dtm],
            coord={"north": 52.4920, "south": 52.4900, "west": -9.6840, "east": -9.6820},
            target_resolution=0.001,
            indexes={"Longitude/X": "0", "Latitude/Y": "1", "Elevation": "2"},
            headers_types={"Longitude/X": "float", "Latitude/Y": "float", "Elevation": "float"},
            delimiter=";",
            skip_rows=1,
            min_sounds=3,
        )
        exporter()

        # Open DTM
        with dtm_driver.open_dtm(path_dtm) as dtm:
            # Check elevation => [[SW, SE], [NW, NE]]. Only SE is not filtered
            assert dtm[DTM.ELEVATION_NAME][0, 0] is np.ma.masked
            assert dtm[DTM.ELEVATION_NAME][0, 1] is not np.ma.masked
            assert dtm[DTM.ELEVATION_NAME][1, 0] is np.ma.masked
            assert dtm[DTM.ELEVATION_NAME][1, 1] is np.ma.masked


def test_set_cdi():
    """
    Convert a CSV to DTM and check the cdi
    """
    # CSV file : NW, NE, SW and SE
    csv_content = """   Longitude;Latitude;Elevation
                        -9.6838  ;52.4911 ;3411.01
                        -9.6828  ;52.4911 ;3422.02
                        -9.6838  ;52.4901 ;3433.03
                        -9.6828  ;52.4901 ;3444.04
        """
    with tmp.TemporaryDirectory() as o_path:
        path_csv = tmp.mktemp(dir=o_path, suffix=".csv")
        path_dtm = tmp.mktemp(dir=o_path, suffix=".nc.dtm")
        with open(path_csv, "wt", encoding="utf8") as csv_file:
            csv_file.write(csv_content.replace(" ", ""))

        # Export CSV -> DTM
        exporter = CsvToDtm(
            i_paths=[path_csv],
            o_paths=[path_dtm],
            coord={"north": 52.4920, "south": 52.4900, "west": -9.6840, "east": -9.6820},
            target_resolution=0.001,
            indexes={"Longitude/X": "0", "Latitude/Y": "1", "Elevation": "2"},
            headers_types={"Longitude/X": "float", "Latitude/Y": "float", "Elevation": "float"},
            delimiter=";",
            skip_rows=1,
            cdi={os.path.basename(path_csv): "Test_CDI"},
        )
        exporter()

        # Open DTM
        with dtm_driver.open_dtm(path_dtm) as dtm:
            # Check CDI
            assert np.array_equal(dtm[DTM.CDI_INDEX][:], np.zeros((2, 2), dtype=int))


def test_merge_csv_to_dtm():
    """
    Convert a CSV (Emo format) to DTM
    """
    # CSV file : NW, NE, SW and SE
    csv1_content = """   Longitude;Latitude;Elevation
                        -9.6838  ;52.4911 ;3411.01
                        -9.6828  ;52.4911 ;3422.02
                        -9.6838  ;52.4901 ;3433.03
                        -9.6828  ;52.4901 ;3444.04
        """
    csv2_content = """   Longitude;Latitude;Elevation
                        -9.6838  ;52.4931 ;3451.05
                        -9.6828  ;52.4931 ;3462.06
                        -9.6838  ;52.4921 ;3473.07
                        -9.6828  ;52.4921 ;3484.08
        """
    with tmp.TemporaryDirectory() as o_path:
        path_csv1 = tmp.mktemp(suffix=".csv", dir=o_path)
        with open(path_csv1, "wt", encoding="utf8") as csv1_file:
            csv1_file.write(csv1_content.replace(" ", ""))
        path_csv2 = tmp.mktemp(suffix=".csv", dir=o_path)
        with open(path_csv2, "wt", encoding="utf8") as csv2_file:
            csv2_file.write(csv2_content.replace(" ", ""))

        path_dtm = tmp.mktemp(suffix=".nc.dtm", dir=o_path)

        # Merge CSV -> DTM
        exporter = CsvToDtm(
            i_paths=[path_csv1, path_csv2],
            o_paths=[path_dtm],
            coord={"north": 52.4940, "south": 52.4900, "west": -9.6840, "east": -9.6820},
            target_resolution=0.001,
            indexes={"Longitude/X": "0", "Latitude/Y": "1", "Elevation": "2"},
            headers_types={"Longitude/X": "float", "Latitude/Y": "float", "Elevation": "float"},
            delimiter=";",
            skip_rows=1,
        )
        exporter()

        # Open DTM
        with dtm_driver.open_dtm(path_dtm) as dtm:
            # Check grid size
            assert dtm.dtm_file.row_count == 4
            assert dtm.dtm_file.col_count == 2

            # Check elevation
            assert np.array_equal(
                dtm[DTM.ELEVATION_NAME][:],
                np.array(
                    [[3433.03, 3444.04], [3411.01, 3422.02], [3473.07, 3484.08], [3451.05, 3462.06]], dtype=np.float32
                ),
            )
