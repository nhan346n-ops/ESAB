#! /usr/bin/env python3
# coding: utf-8
import os
from typing import Dict

from osgeo import gdal, osr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.pyat_logger as log
from pyat.dtm.dtm_standard_constants import ELEVATION_NAME
from pyat.utils.gdal_utils import gdal_progress_callback

cog_driver = gdal.GetDriverByName("COG")
spatial_ref = osr.SpatialReference()
spatial_ref.ImportFromEPSG(3857)
web_mercator = spatial_ref.ExportToWkt()


class DtmToCog:
    """
    Exports a Digital Terrain Model (DTM, .dtm.nc) to Cloud Optimized GeoTIFF (COG, .tif).
    """

    def __init__(
        self,
        i_paths: list,
        o_paths: list = None,
        overwrite: bool = False,
        monitor=DefaultMonitor,
    ):
        """Constructor."""
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.resulting_files = []
        self.overwrite = overwrite
        self.monitor = monitor
        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __process_data(self, i_dtm_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor) -> None:
        """
        Exports one DTM to COG.
        """
        ind = self.i_paths.index(i_dtm_driver.dtm_file.file_path)
        o_path = self.o_paths[ind]
        # Check if output file does not already exist.
        if os.path.exists(o_path):
            if not self.overwrite:
                self.logger.warning(f"{o_path} skipped (already exists and overwrite option is false).")
                return
            os.remove(o_path)

        # Open elevation layer.
        src_path = f"NETCDF:{i_dtm_driver.dtm_file.file_path}:{ELEVATION_NAME}"
        input_dataset = gdal.Open(src_path)

        options = ["COMPRESS=DEFLATE", "TARGET_SRS=" + web_mercator]

        # Export to COG.
        self.logger.info(f"Export {src_path} to {o_path} with options : {options}")
        output_ds = cog_driver.CreateCopy(
            o_path,
            input_dataset,
            options=options,
            callback=gdal_progress_callback,
            callback_data=[0, "exporting DTM to COG", monitor.split(1)],
        )

        if output_ds is not None:
            output_ds = None
            self.resulting_files.append(o_path)
        else:
            raise IOError(f"Unable to create {o_path}")

    def __call__(self) -> Dict:
        process_util.process_each_input_file_in_read_mode(
            self.i_paths,
            self.__class__.__name__,
            self.logger,
            self.monitor,
            self.__process_data,
        )
        return {"outfile": [str(file_path) for file_path in self.resulting_files]}
