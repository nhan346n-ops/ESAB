#! /usr/bin/env python3
# coding: utf-8

import datetime
import os
from os import PathLike
from typing import List, Optional

from osgeo import gdal, gdalconst
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.utils.process_utils as process_util
import pyat.utils.pyat_logger as log
from pyat.utils.gdal_utils import gdal_progress_callback

logger = log.logging.getLogger("convert_tiff_to_csv")


def __export_data(tiff_file: PathLike, csv_file: PathLike, monitor: ProgressMonitor) -> None:
    """
    Launch the export of the file.
    Raised exception : IOError when error occurs while parsing the file
    """
    tiff_ds = __open_tiff(tiff_file)
    try:
        xyz_ds = gdal.Warp(
            csv_file,
            tiff_ds,
            options=gdal.WarpOptions(
                creationOptions=["COLUMN_SEPARATOR=;"],
                format="XYZ",
                callback=gdal_progress_callback,
                callback_data=[0, "exporting Tiff to CSV", monitor],
            ),
        )
        if xyz_ds is not None:
            xyz_ds = None
        else:
            raise IOError(f"Unable to create {csv_file}")
    finally:
        tiff_ds = None


def __open_tiff(tiff_file: str) -> gdal.Dataset:
    """
    Open the tiff. Return the resulting dataset
    """
    dataset = gdal.Open(tiff_file, gdalconst.GA_ReadOnly)
    if dataset is None:
        dataset = None
        raise AttributeError("File is not a Tiff.")
    return dataset


def convert_tiff_to_csv(
    i_paths: List[PathLike],
    o_paths: Optional[List[PathLike]] = None,
    overwrite: bool = False,
    monitor: ProgressMonitor = DefaultMonitor,
) -> None:
    """Utility function to convert TIFF files (or any other GDAL raster file) as CSV."""
    if o_paths:
        o_paths = list(o_paths)
    else:
        # Create output name from the input with the nc extension.
        o_paths = [path[: path.rfind(".")] + ".csv" for path in i_paths]
    if len(o_paths) != len(i_paths):
        raise AttributeError("Number of Output/Input paths must be the same.")

    begin = datetime.datetime.now()
    monitor.set_work_remaining(len(i_paths))
    file_in_error = []
    for tiff_file, csv_file in zip(i_paths, o_paths):
        try:
            logger.info(f"Starting to convert {tiff_file} to {csv_file}")
            if not overwrite and os.path.exists(csv_file):
                logger.warning("File exists and overwrite is not allowed. Convertion aborted.")
            else:
                now = datetime.datetime.now()
                __export_data(tiff_file, csv_file, monitor)

                logger.info(f"End of conversion for {tiff_file} : {datetime.datetime.now() - now} time elapsed\n")
        except Exception as error:
            file_in_error.append(tiff_file)
            logger.error("An exception was thrown!", exc_info=True, stack_info=True)
        monitor.worked(1)

    monitor.done()
    process_util.log_result(logger, begin, file_in_error)
