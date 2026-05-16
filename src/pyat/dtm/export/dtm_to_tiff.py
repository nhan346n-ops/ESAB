#! /usr/bin/env python3
# coding: utf-8
import os
from typing import Dict, List, Optional

import numpy as np
from osgeo import gdal
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.utils.gdal_utils import gdal_progress_callback


class Dtm2Tiff:
    """ "
    Exports a dtm to tiff file.
    """

    def __init__(
        self,
        i_paths: list,
        o_paths: List[str],
        layers: Optional[Dict[str, bool]] = None,
        target_fillvalue: float = 32767,
        nan_fillvalue: bool = False,
        target_compression: bool = True,
        overwrite: bool = False,
        monitor=DefaultMonitor,
    ):
        """Init method."""
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.layers = layers
        self.resulting_files: List[str] = []
        self.fill_value = np.nan if nan_fillvalue else target_fillvalue
        self.compression = target_compression
        self.overwrite = overwrite
        self.monitor = monitor

        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __process_data(self, i_dtm_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor) -> None:
        ind = self.i_paths.index(i_dtm_driver.dtm_file.file_path)
        ref_o_path = self.o_paths[ind]
        # retrieve extension
        _, ext = os.path.splitext(ref_o_path)

        layers = i_dtm_driver.get_layers()
        for layer_name in layers:
            if (
                layer_name not in dtm_driver.LAYER_NAMES
                or self.layers is not None
                and (layer_name not in self.layers or not self.layers[layer_name])
            ):
                self.logger.info(f"Layer {layer_name} is not in the list of layers to export. Skipping it.")
                continue
            src_path = f"NETCDF:{i_dtm_driver.dtm_file.file_path}:{layer_name}"
            o_path = arg_util.create_output_path(
                i_path=ref_o_path,
                extension=ext,
                suffix=f"_{layer_name.lower()}",
                overwrite=self.overwrite,
            )
            self.logger.info(f"Creating file {o_path}")
            # override nan missing value with user fill_value
            dst_no_data = self.fill_value if np.isnan(dtm_driver.get_missing_value(layer_name)) else None
            creation_options = ["COMPRESS=DEFLATE"] if self.compression else None

            tiff_ds = gdal.Warp(
                o_path,
                src_path,
                options=gdal.WarpOptions(
                    dstNodata=dst_no_data,
                    creationOptions=creation_options,
                    callback=gdal_progress_callback,
                    callback_data=[0, "exporting DTM to Tiff", monitor.split(1)],
                ),
            )
            if tiff_ds is not None:
                tiff_ds = None
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
