import logging as log
import os
from typing import List

import pytechsas.sensor.sensor_smoother as ss
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

logger = log.getLogger(__name__)


def apply_smoothing_batch(
    i_paths: List[str],
    o_paths: List[str],
    layers: List[str],
    algorithm: str = "SAVITZKY_GOLAY",  #  SAVITZKY_GOLAY, BUTTERWORTH, BESSEL, BESSEL_Q, MEDIAN_FILTER
    savgol_order=2,
    savgol_window_size=10,
    butter_critical_freq=0.2,
    butter_order=2,
    butter_sampling_freq=1.0,
    bessel_critical_freq=0.2,
    bessel_order=2,
    bessel_sampling_freq=1.0,
    medfilt_window_size=3,
    overwrite: bool = False,
    monitor: ProgressMonitor = DefaultMonitor,
) -> None:
    monitor.begin_task("'Smoother'", len(i_paths) + 1)

    order = 2
    window_size = 10
    critical_freq = 0.2
    sampling_freq = 1.0
    sensor_algorithm = "savgol"
    if algorithm == "SAVITZKY_GOLAY":
        order = savgol_order
        window_size = savgol_window_size
        sensor_algorithm = "savgol"
    elif algorithm == "BUTTERWORTH":
        order = butter_order
        sampling_freq = butter_sampling_freq
        critical_freq = butter_critical_freq
        sensor_algorithm = "butter"
    elif algorithm == "BESSEL":
        order = bessel_order
        sampling_freq = bessel_sampling_freq
        critical_freq = bessel_critical_freq
        sensor_algorithm = "bessel"
    elif algorithm == "BESSEL_Q":
        sensor_algorithm = "bessel_q"
    elif algorithm == "MEDIAN_FILTER":
        sensor_algorithm = "medfilt"
        window_size = medfilt_window_size
    else:
        raise ValueError(f"error : unknown algorithm '{algorithm}'")

    monitor.worked(1)

    for i_path, o_path in zip(i_paths, o_paths):
        # Check output file
        if os.path.exists(o_path):
            if not overwrite:
                logger.warning("File %s already exists, skipping it.", o_path)
                continue

        # Apply smoothing
        ss.apply_smoothing(
            i_path=i_path,
            o_path=o_path,
            layers=layers,
            algorithm=sensor_algorithm,
            order=order,
            window_size=window_size,
            critical_freq=critical_freq,
            sampling_freq=sampling_freq,
        )

        monitor.worked(1)

    monitor.done()
