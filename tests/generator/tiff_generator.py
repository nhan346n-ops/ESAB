#! /usr/bin/env python3
# coding: utf-8

import tempfile as tmp

import numpy as np
from osgeo import gdal, osr


def generate_tiff(left: float, top: float, proj4: str, resolution: float, data: np.ndarray) -> str:
    """
    Simple function to generate a tiff file
    """
    path_tiff = tmp.mktemp(suffix=".tiff")
    dataset: gdal.Dataset = gdal.GetDriverByName("GTiff").Create(
        path_tiff, data.shape[0], data.shape[1], 1, gdal.GDT_Float32
    )
    dataset.SetGeoTransform((left, resolution, 0.0, top, 0.0, -resolution))
    projection = osr.SpatialReference()
    projection.ImportFromProj4(proj4)
    dataset.SetProjection(projection.ExportToWkt())
    band: gdal.Band = dataset.GetRasterBand(1)
    band.SetNoDataValue(np.nan)
    band.WriteArray(data)
    dataset.FlushCache()
    band = None
    dataset = None
    return path_tiff


if __name__ == "__main__":
    elevations = 10 * np.random.default_rng().random((100, 100)) - 100
    for hole in [1, 10, 50, 80]:
        for x in range(-1, 2):
            for y in range(-1, 2):
                elevations[hole + x, hole + y] = np.nan

    path_tiff = generate_tiff(
        left=-30.0,
        top=40.0,
        proj4="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        resolution=0.001,
        data=elevations,
    )
    print(path_tiff)
