from osgeo import gdal
from osgeo import gdalconst
import numpy as np


def read_tiff(file_name: str) -> np.ndarray:
    """
    read a geotiff and wrap its content in a numpy ndarray

    """

    # display differences
    src = gdal.Open(file_name, gdalconst.GA_ReadOnly)
    band = src.GetRasterBand(1)
    values = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    if nodata:
        values = np.ma.masked_equal(values, nodata)
    return values
