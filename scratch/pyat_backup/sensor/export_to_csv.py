import logging as log
import os
from typing import Dict, List

import pytechsas.sensor.sensor_export_to_csv as etc
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

logger = log.getLogger(__name__)


def export_files(
    i_paths: List[str],
    o_path: str,
    overwrite: bool = False,
    separator: str = "Semicolon",
    layers: Dict[str, bool] | None = None,
    monitor: ProgressMonitor = DefaultMonitor,
    **kwargs,  # valid_data_only, ...
) -> None:
    """
    Export NetCDF files to CSV.

    This is a wrapper function that handles separator conversion and file overwrite
    protection before delegating to the pyat export function.

    Args:
        i_paths: List of paths to input NetCDF files.
        o_path: Path to output CSV file.
        overwrite: If True, overwrites existing output file.
        separator: Separator type name ("Comma", "Semicolon", "Space", "Tabulation").
        layers: Dictionary of variable names to export. If None, exports all variables.
        monitor: Progress monitor for tracking export progress.
        **kwargs: Additional parameters passed to pyat export function (e.g., valid_data_only).
    """
    monitor.begin_task("'Export to CSV'", 2)

    # Map separator names to ASCII characters
    ascii_map = {"Comma": ",", "Semicolon": ";", "Space": " ", "Tabulation": "\t"}
    if separator not in ascii_map:
        raise ValueError(f"Unknown separator '{separator}'")
    sep = ascii_map[separator]

    # Check if output file exists and prevent overwrite if not allowed
    if os.path.exists(o_path) and not overwrite:
        logger.warning("File %s already exists, export aborted.", o_path)
        return

    monitor.worked(1)

    # Delegate to pyat export function
    etc.export_files(i_paths=i_paths, o_path=o_path, separator=sep, layers=list(layers) if layers else None, **kwargs)

    monitor.done()
