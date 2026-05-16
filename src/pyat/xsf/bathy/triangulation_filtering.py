#! /usr/bin/env python3
# coding: utf-8
import asyncio
import os
import shutil
import tempfile
from typing import List, NamedTuple, Tuple

import numpy as np
import pygws.client.http.gws_server_configuration as gws_conf
import pygws.client.http.gws_service_launcher as gws_service
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.utils.pyat_logger as log
import pyat.xsf.xsf_driver as xd
from pyat.sounder.sounder_driver import SounderFile
from pyat.xsf.bathy.automatic_filtering import extract_data

__logger = log.logging.getLogger("Filtri")

MERCATOR = "+proj=merc +lon_0=0 +lat_ts={:5f} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
GWS_SERVICE = "Triangulation filtering on raw files"


class FiltriArg(NamedTuple):
    """
    Class representing all arguments for configuring the process

    :param i_paths: List of input XSF files
    :param o_paths: List of output XSF files
    :param overwrite: Overwrite existing files

    :param projection: Projection string (default is Mercator)
    :param method: Filtering method (Height, Normal or Neighbour)
    :param heightInvalidLimit: Height coefficient for sounding invalidation (method height)
    :param heightIterationNumber: Number of treatment iterations (method height)
    :param normalHeightInvalidLimit: Height coefficient for first invalidation (method normal)
    :param normalSelectLimit: Sounding selection parameter (method normal)
    :param normalAngleInvalidLimit: Maximum allowed angle between normals (method normal)
    :param neighbourHeightInvalidLimit: Height coefficient for first invalidation (method neighbour)
    :param neighbourSelectLimit: Sounding selection parameter (method neighbour)
    :param neighbourDistanceInvalidLimit: Maximum distance between soundings (method neighbour)
    :param sliceSize: Size of the slices for filtering
    :param gws_http_port: GWS port automatically set by GWS
    """

    i_paths: List[str]
    o_paths: List[str]
    overwrite: bool = False

    projection: str | None = None
    method: str = "Height"
    heightInvalidLimit: float = 4.0
    heightIterationNumber: int = 30
    normalHeightInvalidLimit: float = 6.0
    normalSelectLimit: int = 4
    normalAngleInvalidLimit: float = 60.0
    neighbourHeightInvalidLimit: float = 5.0
    neighbourSelectLimit: float = 10.0
    neighbourDistanceInvalidLimit: float = 70.0
    sliceSize: int = 250

    gws_http_port: int = 8081
    monitor: ProgressMonitor = DefaultMonitor


def process(**kwargs) -> None:
    """
    Function accepting all arguments of the process as a dict. Possible arguments are listed in "FiltriArg" class
    """
    process_with_FiltriArg(FiltriArg(**kwargs))


def process_with_FiltriArg(args: FiltriArg) -> None:
    """
    Main function
    Browsing input XSF files and applying the filtering method
    """
    __logger.info("Starting triangulation filtering of XSF files")

    # Set up the GWS configuration
    gws_conf.configure_gws(gws_http_port=args.gws_http_port)

    for o_path, i_path in zip(args.o_paths, args.i_paths):
        if os.path.exists(o_path):
            if not args.overwrite:
                __logger.warning(f"{o_path} exists and cannot be overwritten")
                continue
            if os.path.samefile(i_path, o_path):
                __logger.warning(f"{o_path} is the same as {i_path}. Skipped")
                continue

        _process_filtri(args, o_path, i_path)


def _process_filtri(args: FiltriArg, o_path: str, i_path: str) -> None:
    """
    Process the automatic filtering of the XSF file
    - Copy the file
    - Extract data from the XSF file
    - Apply the filtering by invoking the GWS service
    - Update the history of the XSF file
    """
    __logger.info(f"Processing {i_path}")

    # Duplicate the file
    shutil.copy(i_path, o_path)
    # Open the copied file for filtering
    success = True
    with xd.open_xsf(o_path, mode="r+") as xsf_file:
        with tempfile.TemporaryDirectory() as tmp_dir:
            abscissa_file, ordinate_file, depth_file, validity_file = _extract_data(args, xsf_file, tmp_dir)

            out_validity_file = asyncio.run(
                _filter_with_gws(args, xsf_file.sounder_file, abscissa_file, ordinate_file, depth_file, validity_file)
            )

            if os.path.exists(out_validity_file):
                _apply_filtering_result(xsf_file, out_validity_file)
                _update_history(xsf_file)
            else:
                __logger.error(f"Error when invoking GWS service '{GWS_SERVICE}'. Xsf file skipped")
                success = False

    # Remove the resulting file if the process failed
    if not success and os.path.exists(o_path):
        os.remove(o_path)


def _extract_data(args: FiltriArg, xsf_driver: xd.XsfDriver, tmp_dir: str) -> Tuple[str, str, str, str]:
    """
    Extract data from the XSF file and save it to temporary files
    """
    abscissa, ordinate, elevation, validity = extract_data(xsf_driver, args.projection)

    # Save data to temporary files
    abscissa_file = os.path.join(tmp_dir, "abscissa")
    abscissa.tofile(abscissa_file)
    ordinate_file = os.path.join(tmp_dir, "ordinate")
    ordinate.tofile(ordinate_file)
    depth_file = os.path.join(tmp_dir, "depth")
    elevation.tofile(depth_file)
    validity_file = os.path.join(tmp_dir, "validity")
    validity.tofile(validity_file)

    return abscissa_file, ordinate_file, depth_file, validity_file


async def _filter_with_gws(
    args: FiltriArg,
    sounderFile: SounderFile,
    abscissa_file: str,
    ordinate_file: str,
    depth_file: str,
    validity_file: str,
) -> str:
    """Invoke the GWS service to filter the XSF data"""
    __logger.info("Invoking GWS service for filtering")
    out_validity_file = validity_file + ".filtered"

    result = await gws_service.run_service_and_return_output_files(
        GWS_SERVICE,
        {
            "swathCount": sounderFile.swath_count,
            "beamCount": sounderFile.beam_count,
            "inAbscissaFile": abscissa_file,
            "inOrdinateFile": ordinate_file,
            "inDepthFile": depth_file,
            "inValidityFile": validity_file,
            "outValidityFile": out_validity_file,
            "method": args.method,
            "heightInvalidLimit": args.heightInvalidLimit,
            "heightIterationNumber": args.heightIterationNumber,
            "normalHeightInvalidLimit": args.normalHeightInvalidLimit,
            "normalSelectLimit": args.normalSelectLimit,
            "normalAngleInvalidLimit": args.normalAngleInvalidLimit,
            "neighbourHeightInvalidLimit": args.neighbourHeightInvalidLimit,
            "neighbourSelectLimit": args.neighbourSelectLimit,
            "neighbourDistanceInvalidLimit": args.neighbourDistanceInvalidLimit,
            "sliceSize": args.sliceSize,
        },
    )
    if result.is_err():
        __logger.info("GWS service completed with error")
        if os.path.exists(out_validity_file):
            os.remove(out_validity_file)

    return out_validity_file


def _apply_filtering_result(
    xsf_driver: xd.XsfDriver,
    out_validity_file: str,
) -> None:
    """
    Apply the filtering result to the XSF file
    """
    __logger.info("Applying filtering result to XSF file")

    # Xsf layers to be updated
    xsf_status, xsf_details = xsf_driver[xd.STATUS][:], xsf_driver[xd.STATUS_DETAIL][:]

    # Read the filtering result : 2 = invalided, 3 = valided
    filtri_result = np.fromfile(out_validity_file, dtype=np.uint8).reshape(xsf_status.shape)

    # Set rejected flags when the filtering result is invalidated
    xsf_status = np.where(filtri_result == 2, xsf_status | xd.STATUS_REJECTED, xsf_status)
    xsf_details = np.where(filtri_result == 2, xd.STATUS_DETAIL_AUTO, xsf_details)
    # Set rejected flags when the filtering result 3 (invalidated in the first pass and validated in the second)
    xsf_status = np.where(filtri_result == 3, xsf_status | xd.STATUS_REJECTED, xsf_status)
    xsf_details = np.where(filtri_result == 3, xd.STATUS_DETAIL_UNKNOWN, xsf_details)

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
    args = FiltriArg(
        i_paths=[r"e:\temp\0077_20190720_224529_EM122_Marion_Dufresne.xsf.nc"],
        o_paths=[r"e:\temp\0077_20190720_224529_EM122_Marion_Dufresne_filtri_python.xsf.nc"],
        overwrite=True,
    )
    process_with_FiltriArg(args)
