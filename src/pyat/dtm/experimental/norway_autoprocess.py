import math
import os
import tempfile as tmp
from typing import Dict
import numpy as np
import pandas as pd
from osgeo import gdal, osr

import pyat.utils.application_utils as app_util
import pyat.utils.pyat_logger as log
from pyat.dtm.cdi.set_cdi_process import SetCdiProcess
from pyat.dtm.transform.update_boundingbox import ReprojectProcess
from pyat.dtm.convert.gdal_raster_to_dtm import export_gdal_raster_file_to_dtm


class NorwayAutoProcess:
    """
    A process to ease EMODnet processing for Norway (at least).

    This process takes

    * input files as tiff (tested for UTM projection),
    * cdi as a csv list

    It reprojects to a 1/16 res in lat long and applies cdi to the dtm.nc files
    """

    def _norway_autoprocess(self, input_file: str, output_file: str, cdi_dict: Dict[str, str]):
        # retrieve working directory to save temporary files
        workdir = os.path.dirname(output_file)

        # tmp file for utm projection file
        tmp_utm_dtm = os.path.splitext(os.path.basename(input_file))[0]
        tmp_utm_dtm = tmp.mktemp(dir=workdir, suffix=".dtm.nc", prefix=tmp_utm_dtm)

        export_gdal_raster_file_to_dtm(i_path=input_file, o_path=tmp_utm_dtm, overwrite=True)

        # now data is exported as a dtm
        # reproject dataset

        # prepare parameters
        # output file
        tmp_latlon_dtm = tmp.mktemp(dir=workdir, suffix=".dtm.nc", prefix="_lat_lon.dtm.nc")
        i_paths = [tmp_utm_dtm]
        o_paths = [tmp_latlon_dtm]

        # we need to compute the destination bounding box in lat/lon

        target_spatial_reference = osr.SpatialReference()
        target_spatial_reference.ImportFromEPSG(4326)
        target_proj4 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

        # retrieve coordinates and round to 1/16 arcmin
        # open the file
        dataset = gdal.Open(tmp_utm_dtm)

        # gdal stuff to create projections
        input_projection = dataset.GetProjection()
        input_spatial_reference = osr.SpatialReference()
        input_spatial_reference.ImportFromWkt(input_projection)

        # retrieve grid bounds from dataset
        geotransform = dataset.GetGeoTransform()
        x1, x2 = geotransform[0], geotransform[0] + geotransform[1] * dataset.RasterXSize
        y1, y2 = geotransform[3], geotransform[3] + geotransform[5] * dataset.RasterYSize

        # project the four corner of the grid to latitude / longitude
        transform = osr.CoordinateTransformation(input_spatial_reference, target_spatial_reference)
        a, b, c, d = transform.TransformPoints([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])

        # get min/max values to estimate bounding box
        min_longitude = np.min((a[0], b[0], c[0], d[0]))
        max_longitude = np.max((a[0], b[0], c[0], d[0]))
        min_latitude = np.min((a[1], b[1], c[1], d[1]))
        max_latitude = np.max((a[1], b[1], c[1], d[1]))

        # we need to round it to the lowest/highest arcmin
        min_longitude = math.floor(min_longitude * 60)
        min_longitude = min_longitude / 60.0  # swith back to degrees
        max_longitude = math.ceil(max_longitude * 60)
        max_longitude = max_longitude / 60.0  # swith back to degrees

        min_latitude = math.floor(min_latitude * 60)
        min_latitude = min_latitude / 60.0  # swith back to degrees
        max_latitude = math.ceil(max_latitude * 60)
        max_latitude = max_latitude / 60.0  # swith back to degrees

        # create coordinate= bounding box parameter
        coord = {"north": max_latitude, "south": min_latitude, "west": min_longitude, "east": max_longitude}
        target_resolution = 1 / (16 * 60)  # EMODnet is 1/16 of arcmin

        # reproject data to lat/lon
        reproject = ReprojectProcess(
            i_paths=i_paths,
            coord=coord,
            o_paths=o_paths,
            overwrite=True,
            target_spatial_reference=target_proj4,
            target_resolution=target_resolution,
        )
        reproject()
        dataset = None

        # now apply CDI to generated file

        base_filename = os.path.basename(input_file)
        cdi_process = SetCdiProcess(
            i_paths=o_paths, cdi={os.path.basename(tmp_latlon_dtm): cdi_dict[base_filename]}, o_paths=[output_file]
        )
        cdi_process()

        os.remove(tmp_utm_dtm)
        os.remove(tmp_latlon_dtm)

    def __init__(self, **params):
        """Init function, initialize class member
        this will parse parameters and store them"""

        # create a logger, will allow to print in Globe console with info, error, warning level
        self.logger = log.logging.getLogger(NorwayAutoProcess.__name__)

        # parse input file parameters
        if "i_paths" in params:
            self.input_files = params["i_paths"]
        else:
            # If parameter is not found for any reason raise an exception
            raise Exception("Parameter i_paths is missing")

        # parse output file parameters
        if "o_paths" in params:
            self.output_files = params["o_paths"]
        else:
            # If parameter is not found for any reason raise an exception
            raise Exception("Parameter o_paths is missing")

        if "cdi_file" in params:
            self.cdi_file = params["cdi_file"]
        else:
            # If parameter is not found for any reason raise an exception
            raise Exception("Parameter cdi_file is missing")

        # parse parameter overwrite, if not found set to false by default
        self.overwrite = bool(params["overwrite"]) if "overwrite" in params else False

    def read_cdi(self, file):
        """read cdi file and"""
        cdis = pd.read_csv(file)
        geotiff_name = cdis["Geotiff"]
        cdi_reference = cdis["CDI_reference"]
        value_dict = dict(zip(geotiff_name, cdi_reference))
        return value_dict

    def __call__(self):
        """Run the process"""
        self.logger.info(f"--- Starting")
        self.logger.info(f"--- Run process input file(s) {self.input_files}")

        # PUT YOUR CODE HERE
        # hack, retrieve outputfile to get output working directory

        cdi_dict = self.read_cdi(self.cdi_file)

        for input_file, output_file in zip(self.input_files, self.output_files):
            # compute output file name
            try:
                self._norway_autoprocess(input_file=input_file, output_file=output_file, cdi_dict=cdi_dict)
            except Exception as e:
                self.logger.error(f"An error occurred while processing {input_file}", e)
        self.logger.info(f"--- Stopping")


if __name__ == "__main__":
    app_util.launch_application(app_util.get_json_configuration_file(__file__), NorwayAutoProcess)
