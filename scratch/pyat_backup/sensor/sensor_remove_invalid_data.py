import logging as log
import os
from typing import List

from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from pytechsas.sensor.sensor_remove_invalid_data import apply_remove

logger = log.getLogger(__name__)


def apply_remove_batch(
    i_paths: List[str],
    o_paths: List[str],
    keep_status_variable: bool = False,
    overwrite: bool = False,
    monitor: ProgressMonitor = DefaultMonitor,
) -> None:
    monitor.begin_task("'Remove invalid data'", len(i_paths) + 1)
    monitor.worked(1)

    for i_path, o_path in zip(i_paths, o_paths):
        # Copy input file to output path
        if not overwrite and os.path.exists(o_path):
            logger.warning("File %s already exists, skipping it.", o_path)
            continue

        apply_remove(i_path, o_path, keep_status_variable)

        monitor.worked(1)

    monitor.done()
