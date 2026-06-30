import logging as log
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pytechsas.sensor.sensor_csv_to_netcdf_converter as sensor_converter
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from pytechsas.sensor.techsas_sanity_check import Anomaly

from pyat.navigation import navigation_factory

logger = log.getLogger("sensor_csv_to_netcdf_converter")


@dataclass
class FileConversionError:
    """Container for conversion error of one file"""

    file_path: str
    error: str = ""
    anomalies: List[Anomaly] = field(default_factory=list)


@dataclass
class ConversionResult:
    """Container for conversion statistics and error tracking"""

    total_files: int = 0
    successful: int = 0
    skipped: int = 0
    failed: int = 0
    errors: List[FileConversionError] = field(default_factory=list)


def convert(
    i_paths: List[str],
    o_paths: List[str],
    navigation_file: List[str] | None = None,
    overwrite: bool = False,
    monitor: ProgressMonitor = DefaultMonitor,
    **kwargs,  # csv_description, sanity_check, ...
) -> None:
    """
    Convert CSV sensor files to NetCDF format with robust error handling.

    Args:
        i_paths: List of input CSV file paths
        o_paths: List of output NetCDF file paths
        navigation_file: Optional navigation file paths
        overwrite: If True, overwrite existing output files
        monitor: Progress monitoring object
        **kwargs: Additional arguments for sensor_converter

    """
    # Validate input lists have matching lengths
    if len(i_paths) != len(o_paths):
        raise ValueError(
            f"Input and output path lists must have same length. "
            f"Got {len(i_paths)} input paths and {len(o_paths)} output paths."
        )

    result = ConversionResult(total_files=len(i_paths))

    monitor.begin_task("'CSV conversion'", len(i_paths) + 2)
    monitor.worked(1)

    # Load navigation data with error handling
    nav_data = None
    if navigation_file:
        try:
            nav_data = navigation_factory.from_files(navigation_file)
            logger.info("Navigation data loaded successfully from %d file(s)", len(navigation_file))
        except Exception as e:
            logger.error("Failed to load navigation data: %s", str(e))
            # Continue without navigation data rather than failing entirely
            logger.warning("Proceeding with conversion without navigation data")

    monitor.worked(1)

    # Process each file pair
    for i_path, o_path in zip(i_paths, o_paths):
        if monitor.check_cancelled():
            logger.warning("Cancelled")
            break

        try:
            # Validate input file exists
            if not os.path.exists(i_path):
                error_msg = f"Input file not found: {i_path}"
                logger.warning(error_msg)
                result.errors.append(FileConversionError(i_path, error_msg))
                result.failed += 1
                monitor.worked(1)
                continue

            # Check if input file is readable
            if not os.access(i_path, os.R_OK):
                error_msg = f"Input file not readable: {i_path}"
                logger.warning(error_msg)
                result.failed += 1
                result.errors.append(FileConversionError(i_path, error_msg))
                monitor.worked(1)
                continue

            # Validate output directory exists
            o_dir = os.path.dirname(o_path)
            if o_dir and not os.path.exists(o_dir):
                try:
                    os.makedirs(o_dir, exist_ok=True)
                    logger.info("Created output directory: %s", o_dir)
                except Exception as e:
                    error_msg = f"Failed to create output directory: {str(e)}"
                    logger.warning(error_msg)
                    result.errors.append(FileConversionError(i_path, error_msg))
                    result.failed += 1
                    monitor.worked(1)
                    continue

            # Check if output file already exists
            if os.path.exists(o_path):
                if not overwrite:
                    logger.warning("File %s already exists, skipping it.", o_path)
                    result.skipped += 1
                    monitor.worked(1)
                    continue

                # Try to remove existing file
                try:
                    os.unlink(o_path)
                except Exception as e:
                    error_msg = f"Failed to remove existing file: {str(e)}"
                    logger.warning(error_msg)
                    result.errors.append(FileConversionError(i_path, error_msg))
                    result.failed += 1
                    monitor.worked(1)
                    continue

            # Attempt conversion
            try:
                anomalies = sensor_converter.convert(
                    csv_path=i_path,
                    netcdf_path=o_path,
                    nav=nav_data,
                    **kwargs,
                )
                if anomalies:
                    result.errors.append(FileConversionError(i_path, anomalies=anomalies))
                    logger.warning("Anomalies detected")

                else:
                    result.successful += 1
                    logger.debug("Successfully converted: %s -> %s", i_path, o_path)

            except Exception as e:
                error_msg = re.sub(r"[\r\n]+", "", f"Fails to convert : {str(e)}")[:200]
                logger.warning(error_msg)
                result.errors.append(FileConversionError(i_path, error_msg))
                result.failed += 1

                # Clean up partial output file if it exists
                if os.path.exists(o_path):
                    try:
                        os.unlink(o_path)
                        logger.debug("Cleaned up partial output file: %s", o_path)
                    except Exception as cleanup_error:
                        logger.error("Failed to clean up partial file %s: %s", o_path, str(cleanup_error))

        except Exception as e:
            # Catch-all for unexpected errors during file processing
            error_msg = f"Unexpected error: {str(e)}"
            logger.error("Unexpected error processing %s: %s", i_path, str(e))
            result.errors.append(FileConversionError(i_path, error_msg))
            result.failed += 1

        finally:
            monitor.worked(1)

    monitor.done()

    # Print summary
    _print_conversion_summary(result, monitor)


def _print_conversion_summary(result: ConversionResult, monitor: ProgressMonitor) -> None:
    """
    Print a detailed summary of the conversion process.

    Args:
        result: ConversionResult object containing statistics
    """
    logger.info("")
    logger.info("Conversion summary")
    logger.info("Total files processed: %d", result.total_files)
    logger.info("Successfully converted: %d", result.successful)
    logger.info("Skipped (already exist): %d", result.skipped)
    if monitor.check_cancelled():
        skipped = result.total_files - result.successful - result.skipped - result.failed
        if skipped > 0:
            logger.info("Skipped (conversion cancelled): %d", skipped)
    logger.info("Failed: %d", result.failed)

    logger.info("")

    # Log individual errors if any
    if result.errors:
        logger.error("Errors encountered:")
        for file_error in result.errors:
            logger.warning("File: %s", file_error.file_path)
            if file_error.error:
                logger.warning("Reason: %s", file_error.error)
            if file_error.anomalies:
                logger.info(f"Reason: {len(file_error.anomalies)} anomalies")
                if len(file_error.anomalies) > 50:
                    logger.info("List of 50th first anomalies :")
                for anomaly in file_error.anomalies[0:50]:
                    logger.info(anomaly.log())
        logger.warning("")
    else:
        logger.info("No errors encountered during conversion.")
        logger.info("")
