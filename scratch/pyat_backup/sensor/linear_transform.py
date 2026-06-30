import logging as log
import os
from typing import List

import pytechsas.sensor.sensor_linear_transform as slt
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

logger = log.getLogger(__name__)


def apply_linear_transform_batch(
    i_paths: List[str],
    o_paths: List[str],
    overwrite: bool = False,
    monitor: ProgressMonitor = DefaultMonitor,
    **kwargs,  # layers, ...
) -> None:
    monitor.begin_task("'Linear transform'", len(i_paths) + 1)
    monitor.worked(1)

    for i_path, o_path in zip(i_paths, o_paths):
        # Check output file
        if os.path.exists(o_path):
            if not overwrite:
                logger.warning("File %s already exists, skipping it.", o_path)
                continue

        # Apply linear_transform
        slt.apply_linear_transform(
            i_path=i_path,
            o_path=o_path,
            **kwargs,
        )

        monitor.worked(1)

    monitor.done()
