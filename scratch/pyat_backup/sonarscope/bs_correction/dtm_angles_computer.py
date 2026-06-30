import tempfile

import numpy as np
from osgeo import gdal, osr
from scipy.interpolate import RegularGridInterpolator

import pyat.dtm.dtm_standard_constants as DtmConstants
from pyat.sonarscope.common.configuration import default_config
from pyat.utils import gdal_utils
from pyat.utils.coords import create_lonlat_to_xy_converter
from pyat.utils.gdal_utils import gdal_to_netcdf, TemporaryDataset


class DtmAnglesComputer:
    def __init__(self, ref_path: str = None):
        self.ref_path = ref_path
        self.temp_dir = None
        self.slope_mercator_dataset = None
        self.aspect_mercator_dataset = None
        self.slope_array = None
        self.aspect_array = None
        self.proj_string = None

    def _compute_dtm_slope_aspect(self):
        "Compute slope aspect from DTM with GDAL"

        default_config.logger.info(f"Computing Slope {self.ref_path} dataset")

        gdal.UseExceptions()
        # Create the path of the output sub dataset and open input sub dataset.
        reference_dataset = gdal.Open(f"NETCDF:{self.ref_path}:{DtmConstants.ELEVATION_NAME}")
        default_config.logger.info(f"Opening {self.ref_path} dataset")
        wkt = reference_dataset.GetProjection()
        inSRS_converter = osr.SpatialReference()  # makes an empty spatial ref object
        inSRS_converter.ImportFromWkt(wkt)  # populates the spatial ref object with our WKT SRS

        # Project the reference file when spatial reference is geographic
        projected_dataset = None
        default_config.logger.info(f"Compute slope/aspect in metric coordinates")
        source_dataset = reference_dataset
        self.proj_string = wkt
        if inSRS_converter.IsGeographic():
            ulx, xres, xskew, uly, yskew, yres = reference_dataset.GetGeoTransform()
            lrx = ulx + (reference_dataset.RasterXSize * xres)
            lry = uly + (reference_dataset.RasterYSize * yres)
            centerx = (ulx + lrx) / 2
            centery = (uly + lry) / 2
            self.proj_string = f"+proj=merc +lat_ts={centery} +lon_0={centerx} +ellps=WGS84"
            mercator_file = tempfile.mktemp(suffix="_elevation_merc.tiff", dir=self.temp_dir)
            default_config.logger.info(f"Warping {self.ref_path} to a mercator projection ({self.proj_string}")
            source_dataset = gdal.Warp(
                mercator_file, reference_dataset, dstSRS=self.proj_string, resampleAlg="bilinear"
            )
            projected_dataset = TemporaryDataset(source_dataset, mercator_file)

        slope_merc = tempfile.mktemp(suffix="_slope_merc.tiff", dir=self.temp_dir)
        aspect_merc = tempfile.mktemp(suffix="_aspect_merc.tiff", dir=self.temp_dir)

        # reproject in UTM or mercator with gdal. then call slope processing
        slope_mercator_dataset = gdal.DEMProcessing(
            destName=slope_merc,
            srcDS=source_dataset,
            processing="slope",
            computeEdges=True,
        )  # compute slope
        aspect_mercator_dataset = gdal.DEMProcessing(
            destName=aspect_merc, srcDS=source_dataset, processing="aspect", computeEdges=True, zeroForFlat=True
        )  # compute aspect

        # create auto erasable wrapper
        self.slope_mercator_dataset = TemporaryDataset(slope_mercator_dataset, slope_merc)
        self.aspect_mercator_dataset = TemporaryDataset(aspect_mercator_dataset, aspect_merc)

        self.slope_array = gdal_to_netcdf(self.slope_mercator_dataset.dataset)
        self.slope_array[self.slope_array == -9999] = np.nan
        self.aspect_array = gdal_to_netcdf(self.aspect_mercator_dataset.dataset)
        self.aspect_array[self.aspect_array == -9999] = np.nan

        default_config.logger.debug(f"done computing slope in : {self.slope_mercator_dataset}")
        default_config.logger.debug(f"done computing aspect in : {self.aspect_mercator_dataset}")

        # close datasets
        reference_dataset = None
        source_dataset = None
        projected_dataset = None
        slope_mercator_dataset = None
        aspect_mercator_dataset = None

    def _create_interpolator(self, dataset: gdal.Dataset, values: np.ndarray):
        # we read all values but of course, this could be improved by reading only the usefull values

        Xgeo, Ygeo = gdal_utils.get_x_y_coordinates(dataset)
        # reverse Ygeo to match netcdf convention
        Ygeo = Ygeo[::-1]

        interpolator_function = RegularGridInterpolator(
            (Ygeo, Xgeo), values=values, method="linear", bounds_error=False, fill_value=None
        )
        return interpolator_function

    def _interpolate_slope_aspect_from_lonlat(
        self, longitudes: np.ndarray, latitudes: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        "compute slope aspect from DTM and interpolate values to given lon/lat"

        if self.proj_string is None:
            self._compute_dtm_slope_aspect()

        dd_to_xy = create_lonlat_to_xy_converter(proj=self.proj_string)
        xs, ys = dd_to_xy(longitudes.ravel(), latitudes.ravel())

        slope_interpolator = self._create_interpolator(self.slope_mercator_dataset.dataset, values=self.slope_array)
        cos_aspect_interpolator = self._create_interpolator(
            self.aspect_mercator_dataset.dataset, values=np.cos(np.deg2rad(self.aspect_array))
        )
        sin_aspect_interpolator = self._create_interpolator(
            self.aspect_mercator_dataset.dataset, values=np.sin(np.deg2rad(self.aspect_array))
        )

        slope_array = slope_interpolator((ys, xs))
        cos_aspect_array = cos_aspect_interpolator((ys, xs))
        sin_aspect_array = sin_aspect_interpolator((ys, xs))
        aspect_array = np.rad2deg(np.arctan2(sin_aspect_array, cos_aspect_array))

        return slope_array.reshape(longitudes.shape), aspect_array.reshape(longitudes.shape)

    def retrieve_across_along_slope_from_lonlat(
        self, longitudes: np.ndarray, latitudes: np.ndarray, source_headings: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        """Computes and return acrosstrack and alongtrack DTM slopes given input positions ans orientation
        @param longitudes (degree)
        @param latitudes (degree)
        @param source_headings (degree)
        """

        dtm_slope, dtm_aspect = self._interpolate_slope_aspect_from_lonlat(longitudes, latitudes)
        across_slope = np.rad2deg(
            np.arctan(np.tan(np.deg2rad(dtm_slope)) * np.sin(np.deg2rad(source_headings[:, None] - dtm_aspect)))
        )
        along_slope = np.rad2deg(
            np.arctan(np.tan(np.deg2rad(dtm_slope)) * np.cos(np.deg2rad(source_headings[:, None] - dtm_aspect)))
        )
        return across_slope, along_slope
