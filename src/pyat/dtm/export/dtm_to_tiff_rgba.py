import logging
import os
from typing import Dict, List

from pygws.service.progress_monitor import DefaultMonitor

import pyat.dtm.dtm_standard_constants as DtmConstants
from pyat.utils.gdal_utils import GDALDataset, apply_colormap


class Dtm2TiffRgba:
    """
    Exports a DTM to TIFF RGBA file.
    """

    def __init__(
        self,
        i_paths: List[str],
        o_paths: List[str],
        overwrite: bool = False,
        monitor=DefaultMonitor,
    ):
        """Init method."""
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.overwrite = overwrite
        self.monitor = monitor
        self.resulting_files = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self) -> Dict:
        """
        Main function to export DTMs to TIFF RGBA
        """
        self.monitor.begin_task("Exporting DTM to Tiff rgba", len(self.i_paths))

        for i_path, o_path in zip(self.i_paths, self.o_paths):
            if self.monitor.check_cancelled():
                self.logger.warning("Cancel requested. Export aborted")
                break

            if not os.path.exists(o_path) or self.overwrite:
                self.logger.info(f"Creating file {o_path}")
                if self.__process_export(i_path, o_path):
                    self.resulting_files.append(o_path)
            else:
                self.logger.warning(f"{o_path} exists and cannot be overwritten")

            self.monitor.worked(1)
        self.monitor.done()

        return {"outfile": [str(file_path) for file_path in self.resulting_files]}

    def __process_export(self, i_path: str, o_path: str) -> bool:
        """Open the DTM and export it to a coloured TIFF via GDAL utilities."""
        result = False
        src_path = f"NETCDF:{i_path}:{DtmConstants.ELEVATION_NAME}"

        try:
            with GDALDataset(src_path) as ds:
                result = apply_colormap(ds, o_path, monitor=self.monitor, logger=self.logger)
        except Exception as e:
            self.logger.error(f"Error processing {i_path}: {e}")

        return result
