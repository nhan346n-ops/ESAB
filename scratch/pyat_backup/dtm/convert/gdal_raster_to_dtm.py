#! /usr/bin/env python3
# coding: utf-8

import os
from typing import List, Optional, Dict

import numpy as np
from osgeo import gdal, gdalconst

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DTM
import pyat.utils.pyat_logger as log

logger = log.logging.getLogger("gdal_raster_to_dtm")


def export_gdal_raster_to_dtm(
    i_paths: List[str],
    o_paths: Optional[List[str]] = None,
    overwrite: bool = False,
    title: str = "",
    institution: str = "",
    source: str = "",
    references: str = "",
    comment: str = "",
    remove_invalid_elevation: bool = True,
):
    """
    Convert a list of GDAL raster files (like TIFF...) to new DTM (dtm.nc) files.
    :param i_paths: input file path list
    :param o_paths: output file path list
    :param overwrite: allows to overwrite output file
    :param title: title to add in attributes
    :param institution: institution to add in attributes
    :param source: source to add in attributes
    :param references: references to add in attributes
    :param comment: comment to add in attributes
    :param remove_invalid_elevation: if true, removes extreme elevation values
    """
    if o_paths:
        o_paths = list(o_paths)
    else:
        # Create output name from the input with the nc extension.
        o_paths = [path[: path.rfind(".")] + DTM.EXTENSION for path in i_paths]
    if len(o_paths) != len(i_paths):
        raise AttributeError("Number of Output/Input paths must be the same.")

    for i_path, o_path in zip(i_paths, o_paths):
        export_gdal_raster_file_to_dtm(
            i_path, o_path, overwrite, title, institution, source, references, comment, remove_invalid_elevation
        )


def export_gdal_raster_file_to_dtm(
    i_path: str,
    o_path: str,
    overwrite: bool = False,
    title: str = None,
    institution: str = None,
    source: str = None,
    references: str = None,
    comment: str = None,
    remove_invalid_elevation: bool = True,
) -> None:
    """
    Convert GDAL raster files (like TIFF...) to new DTM (dtm.nc) file.
    :param i_path: input file
    :param o_path: output DTM file
    :param overwrite: allows to overwrite output file
    :param title: title to add in attributes
    :param institution: institution to add in attributes
    :param source: source to add in attributes
    :param references: references to add in attributes
    :param comment: comment to add in attributes
    :param remove_invalid_elevation: if true, removes extreme elevation values
    """
    logger.info(f"Starting to convert {i_path} to {o_path}")
    if not overwrite and os.path.exists(o_path):
        logger.warning(f"File {o_path} already exists and overwrite is not allowed, skipping it.")
        return

    # metadata
    metadata = {}
    if title:
        metadata["title"] = title
    if institution:
        metadata["institution"] = institution
    if source:
        metadata["source"] = source
    if references:
        metadata["references"] = references
    if comment:
        metadata["comment"] = comment
    metadata["history"] = f"Converted from {os.path.basename(i_path)} with PyAT (raster_to_dtm.py)"

    # export
    with dtm_driver.open_dtm(o_path, "w") as o_dtm_driver:
        try:
            dataset = gdal.Open(i_path, gdalconst.GA_ReadOnly)
            if not dataset:
                raise AttributeError(f"Can't open file {i_path} as GDAL dataset.")
            __export_dataset__(dataset, o_dtm_driver, metadata, remove_invalid_elevation)
        finally:
            dataset = None


def __export_dataset__(
    dataset: gdal.Dataset, o_dtm_driver: dtm_driver.DtmDriver, metadata: Dict, remove_invalid_elevation: bool
) -> None:
    """
    Export the gdal dataset to the dtm.
    """
    # Creates dimensions and grid mapping in the output file
    o_dtm_driver.dtm_file.initialize_with_gdal_dataset(dataset)

    o_dtm_driver.initialize_file(metadata)

    # create elevation layer
    data = np.array(dataset.ReadAsArray(), dtype=dtm_driver.get_type(DTM.ELEVATION_NAME))
    data = data[::-1]
    # Manage NoData
    tiff_band: gdal.Band = dataset.GetRasterBand(1)
    no_data_value = tiff_band.GetNoDataValue()
    data[data == no_data_value] = dtm_driver.get_missing_value(DTM.ELEVATION_NAME)

    # filter extreme elevation values
    if remove_invalid_elevation:
        logger.info("Remove elevation values above or less than +/- 32000m")
        data[data > 32000] = dtm_driver.get_missing_value(DTM.ELEVATION_NAME)
        data[data < -32000] = dtm_driver.get_missing_value(DTM.ELEVATION_NAME)

    # Write DTM
    o_dtm_driver.add_layer(layer_name=DTM.ELEVATION_NAME, data=data)
