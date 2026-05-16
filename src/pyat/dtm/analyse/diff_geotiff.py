import numpy as np
from osgeo import gdal, gdalconst


class DiffGeotiff:
    """compute a difference between two geotiff dtm , one considered as reference,
    the other one reprojected and translated to match the reference dtm size"""

    def __init__(self):
        pass

    def compute(self, source: str, reference_file: str, outputfile: str):
        """
        project input file into srs from reference_file and keeping the size of reference file
        """

        # Source
        src = gdal.Open(source, gdalconst.GA_ReadOnly)
        src_proj = src.GetProjection()
        src_geotrans = src.GetGeoTransform()

        # We want a section of source that matches this:

        match_ds = gdal.Open(reference_file, gdalconst.GA_ReadOnly)
        match_proj = match_ds.GetProjection()
        match_geotrans = match_ds.GetGeoTransform()
        wide = match_ds.RasterXSize
        high = match_ds.RasterYSize

        # Output / destination
        dst = gdal.GetDriverByName("GTiff").Create(outputfile, wide, high, 1, gdalconst.GDT_Float32)
        dst.SetGeoTransform(match_geotrans)
        dst.SetProjection(match_proj)
        # bug in gdal, we need to prefill with invalid values
        band = dst.GetRasterBand(1)
        nodata = src.GetRasterBand(1).GetNoDataValue()
        if nodata is None:
            band.SetNoDataValue(np.nan)
        else:
            band.SetNoDataValue(nodata)
        a = np.ndarray(shape=(dst.RasterYSize, dst.RasterXSize))
        a.fill(nodata)
        band.WriteArray(a)
        # Do the work
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)

        # compute difference
        band = dst.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        # Create a masked array for making calculations without nodata values
        reprojected_data = band.ReadAsArray()
        reprojected_data = np.ma.masked_equal(reprojected_data, nodata)

        reference_data = match_ds.GetRasterBand(1).ReadAsArray()
        reference_data = np.ma.masked_equal(reference_data, src.GetRasterBand(1).GetNoDataValue())

        d = np.subtract(reference_data, reference_data)

        values = np.subtract(reprojected_data, reference_data)
        values = np.abs(values)
        oldmask = np.array(values.mask)
        values.mask = False
        values[oldmask] = nodata
        band.WriteArray(values)

        del dst  # Flush
