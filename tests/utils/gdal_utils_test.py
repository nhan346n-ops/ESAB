import numpy as np
from osgeo import gdal

from pyat.utils.gdal_utils import GDALDataset, apply_colormap


def test_apply_colormap(tmp_path):
    """Apply the colour relief on a tiny raster and check for a 4-band output."""

    # create a simple 2x2 raster with increasing elevation values
    src_file = tmp_path / "input.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(src_file), 2, 2, 1, gdal.GDT_Float32)
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ds.GetRasterBand(1).WriteArray(arr)
    ds.GetRasterBand(1).SetNoDataValue(np.nan)
    ds.FlushCache()
    ds = None

    out_file = tmp_path / "output.tif"
    with GDALDataset(str(src_file)) as ds:
        success = apply_colormap(ds, str(out_file))

    assert success, "apply_colormap should return True for a valid dataset"
    assert out_file.exists(), "Output file must have been created"

    # open and inspect the result
    ds2 = gdal.Open(str(out_file), gdal.GA_ReadOnly)
    assert ds2 is not None
    assert ds2.RasterCount == 4, "Resulting image should have 4 bands (RGBA)"
    ds2 = None
