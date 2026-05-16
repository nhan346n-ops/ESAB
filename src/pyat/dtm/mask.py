#! /usr/bin/env python3
# coding: utf-8

import os
import re
import tempfile
import xml.etree.ElementTree as ET
from typing import List

import numpy as np
import osgeo.gdal as gdal
import osgeo.ogr as ogr

import pyat.dtm.dtm_standard_constants as DtmConstants
from pyat.utils.exceptions.exception_list import ProcessingError
from pyat.utils.gdal_utils import TemporaryDataset as TempDataSet
from pyat.utils.gdal_utils import gdal_to_netcdf


def compute_geo_mask_from_dataset(dataset: gdal.Dataset, mask_files: list) -> np.ndarray:
    """Compute a mask of the area which must be processed.
    The area is set to 1 if data shall be processed, to 0 if it shall be ignored

    Arguments:
        dataset {gdal.Dataset} -- dataset of the processed dtm.
        mask_files {list} -- list of files (**kml, *.shp)

    Raises:
        ProcessingError: failed to rasterize vector file

    Returns:
        np.ndarray -- mask array
    """
    # Get size of the mask
    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    return _compute_geo_mask(
        x_size=x_size, y_size=y_size, geo_transform=geo_transform, projection=projection, mask_files=mask_files
    )


def compute_geo_mask_from_dtm(path: str, mask_files: list, reverse_mask=False) -> np.ndarray:
    """Compute a mask of the area which must be processed.
    The area is set to 1 if data shall be processed, to 0 if it shall be ignored

    Arguments:
        path {str} -- path of the processed dtm.
        mask_files {list} -- list of files (**kml, *.shp)

    Raises:
        ProcessingError: failed to rasterize vector file

    Returns:
        np.ndarray -- mask array
    """
    # open the raster layer and get its relevant properties
    # this will produce systematically a Warning 1: Recode from UTF-8 to CP_ACP failed with the error
    input_dataset = gdal.Open(f"NETCDF:{path}:{DtmConstants.ELEVATION_NAME}")
    result = compute_geo_mask_from_dataset(input_dataset, mask_files)
    del input_dataset

    if reverse_mask:
        result = np.where(result == 0, 1, 0)

    return result


def compute_geo_mask(x_size, y_size, geo_transform, projection, mask_files: list) -> np.ndarray:
    """Compute a mask of the area which must be processed.
    The area is set to 1 if data shall be processed, to 0 if it shall be ignored

    Arguments:
        x_size {int} -- the size of the grid result
        y_size {int} -- the size of the grid result
        geo_transform -- gdal geotransform
        projection -- gdal projection
        mask_files {list} -- list of files (**kml, *.shp)

    Raises:
        ProcessingError: failed to rasterize vector file

    Returns:
        np.ndarray -- mask array
    """
    return _compute_geo_mask(x_size, y_size, geo_transform, projection, mask_files)


def _compute_geo_mask(x_size, y_size, geo_transform, projection, mask_files: list) -> np.ndarray:

    gdal.PushErrorHandler("CPLQuietErrorHandler")
    # create output temporary file name
    mask_temp_file = tempfile.mktemp(suffix=".tiff")
    # mask_temp_file=tempfile.mktemp(suffix=".tiff",dir="d://tmp//")

    # create the target layer (1 band)
    driver = gdal.GetDriverByName("GTiff")
    mask_dataset = driver.Create(mask_temp_file, x_size, y_size, bands=1, eType=gdal.GDT_Byte)
    mask_dataset.SetGeoTransform(geo_transform)
    mask_dataset.SetProjection(projection)

    # Fill and set NoDataValue
    band = mask_dataset.GetRasterBand(1)
    NoData_value = 0
    band.SetNoDataValue(NoData_value)
    band.FlushCache()

    if len(mask_files) == 0:
        # all data is valid, we fill the band
        band.Fill(1)

    # rasterize the vector layer into the target one
    for mask_file in mask_files:
        # read each vector layer
        vector_dataset = ogr.Open(mask_file)
        layer_count = vector_dataset.GetLayerCount()
        for i in range(0, layer_count):
            vector_layer = vector_dataset.GetLayer(i)
            # gdal.Rasterize function does not work, use RasterizeLayer instead
            failed = gdal.RasterizeLayer(mask_dataset, [1], vector_layer, None, None, [1])
            if failed:
                raise ProcessingError(f"failed to rasterize vector file {mask_file} ")
    band.FlushCache()

    # now use dataset
    mask_dataset = TempDataSet(dataset=mask_dataset, filepath=mask_temp_file)
    # convention between netcdf and geotiff differs
    # (netcdf is lower left referenced while geotiff is upper left referenced), thus the area return shall be reversed
    mask = gdal_to_netcdf(mask_dataset.dataset)

    return mask


def _closed_kml_shapefile(fname: str) -> str:
    """Read given KML file, and tries to write a copy, only with
    the first coordinates added a second time at the end so the shape is closed.

    Returns the filename in which that new shape has been written.
    """
    # XML magic to access the coordinates element.
    tree = ET.parse(fname)
    root = tree.getroot()
    # NB: python xml stdlib does not allow easily to explore namespaced data.
    #  here, we will just hope that the examples of namespaces we got are the only possible.
    POSSIBLE_XML_NAMESPACES = ("http://earth.google.com/kml/2.2", "http://www.opengis.net/kml/2.2")
    for xmlns in POSSIBLE_XML_NAMESPACES:
        coordss = list(root.iter(f"{{{xmlns}}}coordinates"))
        if coordss:  # we got it right !
            break
    else:  # no usable namespace found
        raise ValueError(f"KML shape file uses an unknown XML namespace.")
    assert len(coordss) == 1, coordss
    elem_coords = coordss[0]

    # modify the coordinates (if necessary).
    assert isinstance(elem_coords.text, str)
    coords = re.sub(r"\s", " ", elem_coords.text).split(" ")
    if coords[-1] == coords[0]:  # the shape was already closed
        return fname

    # close the shape, and save the xml in a new file
    elem_coords.text += " " + coords[0]
    name, ext = os.path.splitext(fname)
    name = os.path.split(name)[1]
    with tempfile.NamedTemporaryFile(mode="wt", suffix=name + "-closed" + ext, delete=False) as fd:
        mask_temp_file = fd.name
    # the following kwargs are important for GDAL to understand the XML
    ET.register_namespace(
        "", xmlns
    )  # bugfix: default_namespace argument of tree.write was raising a known error. Let's hope that fix has no side-effects.
    tree.write(mask_temp_file, encoding="UTF-8", xml_declaration=True)
    return mask_temp_file


def crop_with_masks(infile_uri: str, mask_files: List[str], outfile: str):
    """Use Gdal Warp to crop data found in filename, returning the result
    as a gdal Dataset object, and writing it as GeoTiff in outfile.

    infile_uri -- GDAL URI to open infile (e.g. NETCDF:{infile}:elevation)
    mask_files -- non-empty list of shapefile (e.g. kml)
    logger -- logger object, used for error handling
    outfile -- path to output GeoTiff file to be written. Can be None.

    return -- the GDAL Dataset describing data, and a boolean indicating whether or not an input mask file(s) needed to be closed.

    """
    malformed_shape_file = False  # until gdal.Warp() says otherwise
    assert mask_files, "list of masks cannot be empty"
    if len(mask_files) == 0:  # no mask to apply, let's open the file
        if outfile:
            raise ValueError("outfile is given, but won't be written as no mask_files were given.")
        return gdal.Open(infile_uri), malformed_shape_file
    if len(mask_files) > 1:
        # NB: when multiple masks are given, data found in ANY mask must be kept.
        #  But, in its most simple implementation, this is the reverse happening:
        #  only data found in ALL mask are kept. That's quite useless.
        # Side technical note: outfile will need to be different at each iteration
        #  if multiple Warp call are operated.
        raise NotImplementedError("Application of multiple masks is not yet implemented.")

    mask_file = mask_files[0]  # to handle multiple masks, replace that by a loop.
    new_mask_file = os.path.join(
        os.path.split(outfile)[0], os.path.splitext(os.path.split(mask_file)[1])[0] + "-closed.kml"
    )
    closed_mask_file = _closed_kml_shapefile(mask_file)
    if closed_mask_file != mask_file:  # the shape needed to be closed.
        malformed_shape_file = True

    outdata = gdal.Warp(
        outfile,
        infile_uri,
        options=gdal.WarpOptions(
            cutlineDSName=closed_mask_file, cropToCutline=True, copyMetadata=True, dstNodata=np.nan, format="GTiff"
        ),
    )
    if outdata is None:  # problem with Warp
        raise NotImplementedError("GDAL Warp did not succeed. See logs for details.")

    return outdata, malformed_shape_file
