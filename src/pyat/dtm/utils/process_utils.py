#! /usr/bin/env python3
# coding: utf-8

import os
import shutil
import tempfile
import warnings
from datetime import datetime
from logging import Logger
from os import PathLike
from typing import Callable, List

from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.utils.argument_utils as arg_util
import pyat.utils.netcdf_utils as nc_util

warnings.simplefilter(action="ignore", category=RuntimeWarning)


def initialize_output_file(i_driver: dtm_driver.DtmDriver, o_driver: dtm_driver.DtmDriver, process_name: str):
    """
    Initialize output DTM as the input one
    Same shape, CRS, grid mapping...
    Append history
    """
    # Initialize output file
    dtm_driver.copy_metadata(i_driver.dtm_file, o_driver.dtm_file)
    o_driver.initialize_file()
    # History
    o_driver.dataset.history = str(i_driver.dataset.history)
    nc_util.set_history_attr(o_driver.dataset, process_name, i_driver.dtm_file.file_path)


def process_each_input_file_to_output_file(
    process_name: str,
    i_paths: List[PathLike],
    process_data_func: Callable[[PathLike, PathLike, ProgressMonitor], None],
    logger: Logger,
    o_paths: List[PathLike] = None,
    suffix: str = "-out",
    extension: str = DtmConstants.EXTENSION_NC,
    overwrite: bool = False,
    monitor: ProgressMonitor = DefaultMonitor,
) -> None:
    """Run a simple process wich produced one file for each input file.

    For each input and output file, invoke the function process_data_func.

    Arguments:
        process_name {str} -- Name of the process
        i_paths {list} -- input file list.
        process_data_func -- function called on each input and output files
        logger -- logger instance
        o_paths {list} -- Optional output file list.
        suffix {str} -- Suffix of generated output path. Used when o_paths is empty or None.
        overwrite {bool} -- true to overwrite output file if exists.
        monitor -- Progress monitor
    """
    if not i_paths:
        raise ValueError(f"Argument i_paths can not be empty")

    begin = datetime.now()
    monitor.set_work_remaining(len(i_paths))
    files_in_error = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for ind, i_path in enumerate(i_paths):
            logger.info(f"Starting {process_name} on {i_path}.")
            begin_tmp = datetime.now()
            sub_monitor = monitor.split(1)
            try:
                o_path = arg_util.create_output_path(
                    i_path,
                    extension=extension,
                    suffix=suffix,
                    o_path=(None if not o_paths else o_paths[ind]),
                    overwrite=overwrite,
                )

                # input file == output file => this is an update
                updating_input_file = os.path.exists(o_path) and os.path.samefile(i_path, o_path)
                if updating_input_file:
                    # Perform the function into a temporary file
                    o_path = os.path.join(tmp_dir, os.path.basename(i_path))
                    logger.info(f"Working with the temporary file : {o_path}")

                # Invoke the function
                process_data_func(i_path, o_path, sub_monitor)
                logger.info(f"File processed : {i_path} in {datetime.now() - begin_tmp}")

                # Overwrite the input file with temporary one
                if updating_input_file:
                    logger.info("Writing the result in the input file")
                    shutil.copy(o_path, i_path)

            except ValueError as error:
                logger.error(f"Error : {str(error)}")
                files_in_error.append(i_path)
            except FileExistsError as e:
                logger.error(
                    f"{e.filename} already exists and overwrite not allowed (allow overwrite with option: '-ow --overwrite)"
                )
                files_in_error.append(i_path)
            except Exception:
                logger.exception(f"Error while processing file {i_path}")
                files_in_error.append(i_path)

            finally:
                sub_monitor.done()

    log_result(logger, begin, files_in_error)


def process_each_input_dtm_to_output_dtm(
    process_name: str,
    i_paths: List[str],
    process_data_func: Callable[[dtm_driver.DtmDriver, dtm_driver.DtmDriver, ProgressMonitor], None],
    logger: Logger,
    o_paths: List[str] = None,
    suffix: str = "-out",
    overwrite: bool = False,
    monitor: ProgressMonitor = DefaultMonitor,
) -> None:
    """Run a simple process wich produced one DTM file for each input DTM file.

    For each input path, open the file, then create the dimensions
    and copy the global attributes. After, process the layers. Finally, close the file.

    Arguments:
        process_name {str} -- Name of the process
        i_paths {list} -- DTM input file list (dtm.nc).
        process_data_func -- function called on each input file
        logger -- logger instance
        o_paths {list} -- Optional output file list (.nc).
        suffix {str} -- Suffix of generated output path. Used when o_paths is empty or None.
        overwrite {bool} -- true to overwrite output file if exists.
        monitor -- Progress monitor
    """

    def __process_one_dtm(i_path: PathLike, o_path: PathLike, sub_monitor):
        # Open files
        with dtm_driver.open_dtm(i_path) as i_driver, dtm_driver.open_dtm(o_path, "w") as o_driver:
            # Process layers
            process_data_func(i_driver, o_driver, sub_monitor)

    process_each_input_file_to_output_file(
        process_name=process_name,
        i_paths=i_paths,
        process_data_func=__process_one_dtm,
        logger=logger,
        o_paths=o_paths,
        suffix=suffix,
        overwrite=overwrite,
        monitor=monitor,
    )


def process_each_input_file_in_write_mode(
    i_paths: list,
    process_name: str,
    logger: Logger,
    monitor: ProgressMonitor,
    process_data_func: Callable[[dtm_driver.DtmDriver, ProgressMonitor], None],
) -> None:
    """Run a process wich performed a specific function on each input file to modify them.

    Arguments:
        i_paths {list} -- NetCDF input file list (.nc).
        process_name {str} -- Name of the process
        logger -- logger instance
        monitor -- Progress monitor
        process_data_func -- funcion called of each opened input file
    """
    __process_each_input_file(i_paths, process_name, logger, monitor, process_data_func, "r+")


def process_each_input_file_in_read_mode(
    i_paths: list,
    process_name: str,
    logger: Logger,
    monitor: ProgressMonitor,
    process_data_func: Callable[[dtm_driver.DtmDriver, ProgressMonitor], None],
) -> None:
    """Run a process wich performed a specific function on each input file opened for reading only.

    Arguments:
        i_paths {list} -- NetCDF input file list (.nc).
        process_name {str} -- Name of the process
        logger -- logger instance
        monitor -- Progress monitor
        process_data_func -- funcion called of each opened input file
        mode -- access mode. (see netCDF4.Dataset constructor)
    """
    __process_each_input_file(i_paths, process_name, logger, monitor, process_data_func, "r")


def __process_each_input_file(
    i_paths: list,
    process_name: str,
    logger: Logger,
    monitor: ProgressMonitor,
    process_data_func: Callable[[dtm_driver.DtmDriver, ProgressMonitor], None],
    mode,
) -> None:
    """Run a process wich performed a specific function on each input file.

    Arguments:
        i_paths {list} -- NetCDF input file list (.nc).
        process_name {str} -- Name of the process
        logger -- logger instance
        monitor -- Progress monitor
        process_data_func -- funcion called of each opened input file
        mode -- access mode. (see netCDF4.Dataset constructor)
    """
    begin = datetime.now()
    monitor.set_work_remaining(len(i_paths))
    files_in_error = []
    for ind, i_path in enumerate(i_paths):
        i_file = None
        logger.info(f"Starting {process_name} on {i_path}.")
        begin_tmp = datetime.now()
        sub_monitor = monitor.split(1)
        try:
            with dtm_driver.open_dtm(i_path, mode) as i_dtm_driver:
                if mode != "r":
                    # History
                    nc_util.set_history_attr(i_dtm_driver.dataset, process_name, i_paths)

                # Process layers
                process_data_func(i_dtm_driver, sub_monitor)

                end_tmp = datetime.now()
                logger.info(f"File processed : {i_path} in {end_tmp - begin_tmp}")

        except ValueError as error:
            logger.error(f"Error : {str(error)}")
            files_in_error.append(i_path)
        except FileExistsError as e:
            logger.error(
                f"{e.filename} already exists and overwrite not allowed (allow overwrite with option: '-ow --overwrite)"
            )
            files_in_error.append(i_path)
        except Exception:
            logger.exception(f"Error while processing file {i_path}")
            files_in_error.append(i_path)

        finally:
            if i_file and i_file.isopen():
                i_file.close()
            sub_monitor.done()

    log_result(logger, begin, files_in_error)


def log_result(logger: Logger, begin, files_in_error):
    """
    Common sequence of code to log the end of a process
    """
    duration = datetime.now() - begin
    errors = len(files_in_error)
    if errors > 0:
        logger.error(f"Process ended in {duration}, with {errors} file{'s' if errors > 1 else ''} in error :")
        for f in files_in_error:
            logger.error(f"-> {f}")
    else:
        logger.info(f"All files successfully processed in {duration}.")
