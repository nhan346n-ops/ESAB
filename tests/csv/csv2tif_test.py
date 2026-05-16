#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile as tmp

import numpy as np
from osgeo import gdal

from pyat.csv.csv_to_tiff_exporter import CsvToTiffExporter


def test_csv_lat_lon_export():
    """
    Convert a CSV (Emo format) to Tiff
    """
    csv_content = """   Longitude;Latitude;Min Elev;Max Elev;Elevation;Std dev;Test_int;Interpol;Smooth Elev;Fake;CDI
                        -9.6838  ;52.4911 ;3411.01 ;3411.21 ;3411.11  ;1.60   ;1       ;5       ;3447.22    ;3.58;CDI_1
                        -9.6828  ;52.4911 ;3422.02 ;3422.22 ;3422.12  ;1.70   ;2       ;6       ;3445.80    ;====;CDI_2
                        -9.6838  ;52.4901 ;3433.03 ;3433.23 ;3433.13  ;1.80   ;3       ;7       ;3445.24    ;Fake;
                        -9.6828  ;52.4901 ;9999.99 ;9999.99 ;9999.99  ;9.99   ;9       ;1       ;9999.99    ;9999;
                        -9.6828  ;52.4901 ;3444.04 ;3444.24 ;3444.14  ;1.90   ;4       ;8       ;3442.93    ;    ;CDI_4
        """
    with tmp.TemporaryDirectory() as o_path:
        path_csv = tmp.mktemp(suffix=".csv", dir=o_path)
        with open(path_csv, "w") as csv_file:
            csv_file.write(csv_content.replace(" ", "").replace(";", "\t   \t"))

        # Export CSV -> Tif
        exporter = CsvToTiffExporter(
            i_paths=[path_csv],
            o_paths=o_path,
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
                "Value": "4",
            },
            headers_types={
                "Longitude/X": "float",
                "Latitude/Y": "float",
                "Value": "float",
            },
            delimiter="…",
            skip_rows=1,
            depth_sign=-1,
        )
        exporter()

        # Check the nb of tif produced
        expected_tiff = path_csv.replace(".csv", ".tif")
        assert os.path.exists(expected_tiff)

        dataset = gdal.Open(expected_tiff)
        try:
            band: gdal.Band = dataset.GetRasterBand(1)
            # Check elevation => cell filled with the mean value
            assert band.DataType == gdal.GDT_Float32
            assert np.array_equal(
                band.ReadAsArray(),
                np.array([[-3411.11, -3422.12], [-3433.13, -3444.14]], dtype=np.float32),
            )
        finally:
            del dataset
