#! /usr/bin/env python3
# coding: utf-8
import os
import shutil
from typing import List, NamedTuple, Tuple

import numpy as np
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from pyproj import Transformer, crs

import pyat.utils.pyat_logger as log
import pyat.xsf.xsf_driver as xd
from pyat.sounder.sounder_automatic_filtering import process_with_ndarray

__logger = log.logging.getLogger("AutomaticFiltering")

MERCATOR = "+proj=merc +lon_0=0 +lat_ts={:5f} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"


class AutomaticFilteringArg(NamedTuple):
    """
    Class representing all arguments for configuring the process

    :param i_paths: List of input XSF files
    :param o_paths: List of output XSF files
    :param overwrite: Overwrite existing files

    :param projection: Projection string (default is Mercator)
    :param contamination: The amount of contamination of the data set, i.e. the proportion of outliers in the data set (default is 0.05)
    """

    i_paths: List[str]
    o_paths: List[str]
    overwrite: bool = False

    projection: str | None = None
    contamination: float = 0.05
    monitor: ProgressMonitor = DefaultMonitor


def process(**kwargs) -> None:
    """
    Function accepting all arguments of the process as a dict. Possible arguments are listed in "AutomaticFilteringArg" class
    """
    process_with_AutomaticFilteringArg(AutomaticFilteringArg(**kwargs))


def process_with_AutomaticFilteringArg(args: AutomaticFilteringArg) -> None:
    """
    Main function
    Browsing input XSF files and applying the automatic filtering
    """
    __logger.info("Starting automatic filtering of XSF files")

    for o_path, i_path in zip(args.o_paths, args.i_paths):
        if os.path.exists(o_path):
            if not args.overwrite:
                __logger.warning(f"{o_path} exists and cannot be overwritten")
                continue
            if os.path.samefile(i_path, o_path):
                __logger.warning(f"{o_path} is the same as {i_path}. Skipped")
                continue

        _process_filtering(args, o_path, i_path)


def _make_transformer(projection: str | None, mean_latitude: float) -> Transformer:
    """
    Get the transformation based on the projection
    """
    # By default, the projection is Mercator
    spatial_reference = crs.CRS.from_proj4(MERCATOR.format(mean_latitude))
    if projection is not None:
        try:
            # Check if the specified projection is valid
            projection = crs.CRS.from_proj4(projection)
            if not projection.is_projected:
                __logger.warning("Projection not suitable. Using Mercator instead")
            else:
                spatial_reference = projection
        except crs.CRSError:
            __logger.warning(f"Invalid projection {projection}. Using Mercator instead")
    else:
        __logger.info(f"Using projection {MERCATOR.format(mean_latitude)}")

    return Transformer.from_crs(
        crs.CRS.from_epsg(4326),
        spatial_reference,
        always_xy=True,
    )


def _process_filtering(args: AutomaticFilteringArg, o_path: str, i_path: str) -> None:
    """
    Process the automatic filtering of the XSF file
    - Copy the file
    - Extract data from the XSF file
    - Apply the filtering using mapped-based pyat sercice
    - Update the history of the XSF file
    """
    __logger.info(f"Processing {i_path}")

    # Duplicate the file
    shutil.copy(i_path, o_path)
    # Open the copied file for filtering
    success = True
    with xd.open_xsf(o_path, mode="r+") as xsf_file:
        x, y, z, validity = extract_data(args, xsf_file)
        new_validity = process_with_ndarray(x, y, z, validity, args.contamination)
        _apply_filtering_result(xsf_file, new_validity)
        _update_history(xsf_file)

    # Remove the resulting file if the process failed
    if not success and os.path.exists(o_path):
        os.remove(o_path)


def extract_data(
    xsf_driver: xd.XsfDriver, projection: str | None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract data from the XSF file
    """
    __logger.info("Extracting data from XSF file")
    # Extract data from the XSF file
    latitude = xsf_driver.read_detection_latitude()
    longitude = xsf_driver.read_detection_longitude()
    elevation = -xsf_driver.read_fcs_depths(0, xsf_driver.sounder_file.swath_count).astype(float)
    validity = xsf_driver.read_validity_flags(0, xsf_driver.sounder_file.swath_count)

    # Project coordinates
    mean_latitude = np.nanmean(latitude)
    transformer = _make_transformer(projection, mean_latitude)
    abscissa, ordinate = transformer.transform(longitude, latitude, radians=False)

    return abscissa, ordinate, elevation, validity


def _apply_filtering_result(xsf_driver: xd.XsfDriver, filtering_result: np.ndarray) -> None:
    """
    Apply the filtering result to the XSF file
    :param xsf_driver: XSF file to update
    :param filtering_result: Filtering result : 2 = invalided
    """
    __logger.info("Applying filtering result to XSF file")

    # Xsf layers to be updated
    xsf_status, xsf_details = xsf_driver[xd.STATUS][:], xsf_driver[xd.STATUS_DETAIL][:]

    # Set rejected flags when the filtering result is invalidated
    xsf_status = np.where(filtering_result == 2, xsf_status | xd.STATUS_REJECTED, xsf_status)
    xsf_details = np.where(filtering_result == 2, xd.STATUS_DETAIL_AUTO, xsf_details)

    xsf_driver[xd.STATUS][:] = xsf_status[:]
    xsf_driver[xd.STATUS_DETAIL][:] = xsf_details[:]


def _update_history(xsf_driver: xd.XsfDriver) -> None:
    """
    Update the history of the XSF file
    """
    xsf_driver.append_history_line("Automatic cleaning")
    xsf_driver.update_processing_status({xd.ATT_PROCESSING_STATUS_AUTOMATIC_CLEANING: xd.ATT_PROCESSING_STATUS_FLAG_ON})


if __name__ == "__main__":
    # Example usage
    args = AutomaticFilteringArg(
        i_paths=[r"e:\temp\0077_20190720_224529_EM122_Marion_Dufresne_1.xsf.nc"],
        o_paths=[r"e:\temp\0077_20190720_224529_EM122_Marion_Dufresne_1_filtered.xsf.nc"],
        overwrite=True,
        contamination=0.05,
    )
    process_with_AutomaticFilteringArg(args)
