from typing import Dict

from osgeo import gdal, ogr, osr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_standard_constants as dtmconstants
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.pyat_logger as log
from pyat.dtm import dtm_driver
from pyat.utils.gdal_utils import GDALDataset, OGRDataset, gdal_progress_callback


def create_isobath_layer(ogr_ds: OGRDataset, spatial_reference: osr.SpatialReference) -> ogr.Layer:
    """
    creates an isobath layer in the specified OGR dataset.
    """
    # create layer and define spatial reference
    contour_lyr = ogr_ds.CreateLayer("isobath", spatial_reference)
    # define fields of id and elev
    contour_lyr.CreateField(ogr.FieldDefn("ID", ogr.OFTInteger))
    contour_lyr.CreateField(ogr.FieldDefn("elev", ogr.OFTReal))

    return contour_lyr


class Dtm2Isobath:
    """
    Computes DTM isobaths.
    """

    def __init__(
        self,
        i_paths: list[str],
        o_paths: list[str] = None,
        isobath_interval: float = 50,
        overwrite: bool = False,
        monitor=DefaultMonitor,
    ):
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.isobath_interval = isobath_interval
        self.resulting_files = []
        self.overwrite = overwrite
        self.monitor = monitor
        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __compute_isobaths(self, i_dtm_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor, contour_base=0) -> None:
        """
        Computes DTM isobath using GDAL's ContourGenerate() function.
        """
        idx = self.i_paths.index(i_dtm_driver.dtm_file.file_path)
        dtm_name = f'NETCDF:"{i_dtm_driver.dtm_file.file_path}":{dtmconstants.ELEVATION_NAME}'
        o_shp = self.o_paths[idx]

        with GDALDataset(dtm_name) as rasterDs:
            with OGRDataset(o_shp) as isobathDs:
                elevation_band = rasterDs.GetRasterBand(1)

                # create isobath layer in output shapefile with same spatial reference as DTM
                self.logger.info(f"Creating file {o_shp}")
                contour_lyr = create_isobath_layer(
                    ogr_ds=isobathDs, spatial_reference=i_dtm_driver.dtm_file.spatial_reference
                )

                # generate isobaths using GDAL
                result = gdal.ContourGenerate(
                    elevation_band,  # Band srcBand
                    self.isobath_interval,  # double contourInterval
                    contour_base,  # double contourBase
                    [],  # int fixedLevelCount
                    0,  # int useNoData
                    0,  # double noDataValue
                    contour_lyr,  # Layer dstLayer
                    0,  # int idField
                    1,  # int elevField
                    callback=gdal_progress_callback,
                    callback_data=[0, "generating isobath", monitor],
                )

                if result == 0:
                    self.resulting_files.append(o_shp)

    def __call__(self) -> Dict:
        process_util.process_each_input_file_in_read_mode(
            self.i_paths,
            self.__class__.__name__,
            self.logger,
            self.monitor,
            self.__compute_isobaths,
        )
        return {"outfile": [str(file_path) for file_path in self.resulting_files]}
