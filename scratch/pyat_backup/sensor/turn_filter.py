import logging as log
from typing import List, Optional

import pygws.service.execution_context as exec_ctx
import xarray as xr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from pytechsas.utils import turn_filter

from pyat.navigation import navigation_factory

logger = log.getLogger("techsas_turn_filter")


def apply_filter_batch(
    i_paths: List[str],
    filter_method: str = "std",
    heading_period: Optional[int] = None,
    heading_threshold: Optional[float] = None,
    speed_period: Optional[int] = None,
    speed_threshold: Optional[float] = None,
    minimum_duration: Optional[int] = None,
    o_cut_file: Optional[str] = None,
    save_in_input_file: bool = False,
    monitor: ProgressMonitor = DefaultMonitor,
) -> None:
    # Prefer to use RSocket monitor if available
    if exec_ctx.get_root_progress_monitor() is not None:
        monitor = exec_ctx.get_root_progress_monitor()

    monitor.begin_task("Evalutating", 100 * len(i_paths))

    # process each file
    for i_path in i_paths:
        logger.info(f"Processing file '{i_path}'")
        turn_filter_result = None

        try:
            with navigation_factory.from_file(i_path) as nav:
                turn_filter_result = turn_filter.apply_filter_on_file(
                    nav=nav,
                    filter_method=filter_method,
                    heading_period=heading_period,
                    heading_threshold=heading_threshold,
                    speed_period=speed_period,
                    speed_threshold=speed_threshold,
                    minimum_duration=minimum_duration,
                    o_cut_file=o_cut_file,
                )
            break  # Success, exit retry loop

        except PermissionError:
            logger.error(
                f"ERROR: Cannot access file '{i_path}'. "
                "The file is currently locked by another process or application. "
                "Please close any programs that may have this file open and try again."
            )
            # Skip this file and continue with the next one
            monitor.worked(100)

        except Exception as e:
            logger.error(f"Unexpected error processing file '{i_path}': {type(e).__name__}: {e}")
            monitor.worked(100)

        # If we successfully processed the file
        if turn_filter_result is not None:
            monitor.worked(90)

            if save_in_input_file:
                _, _, updated_variables = turn_filter_result
                if updated_variables:
                    # Also handle permission error when writing back
                    try:
                        status_ds = xr.Dataset(data_vars=updated_variables)
                        status_ds.to_netcdf(path=i_path, mode="a")
                    except PermissionError:
                        logger.error(
                            f"ERROR: Cannot write to file '{i_path}'. "
                            "The file is locked by another process. "
                            "Results could not be saved to the input file."
                        )
                    except Exception as e:
                        logger.error(f"Error saving results to '{i_path}': {type(e).__name__}: {e}")
            monitor.worked(10)
