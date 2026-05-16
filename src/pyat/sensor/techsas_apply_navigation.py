import logging as log
import os
import shutil
from typing import List

import pytechsas.sensor.techsas_apply_navigation as pytechsas_apply_navigation
from pyat.navigation import navigation_factory

logger = log.getLogger("techsas_apply_navigation")


def apply_navigation_batch(
    i_paths: List[str], o_paths: List[str], i_nav_files: List[str], overwrite: bool = False
) -> None:
    nav_data = navigation_factory.from_files(i_nav_files)
    for i_path, o_path in zip(i_paths, o_paths):
        # Copy input file to output path
        if not overwrite and os.path.exists(o_path):
            logger.warning("File %s already exists, skipping it.", o_path)
            continue
        if o_path != i_path:
            shutil.copy(i_path, o_path)

        pytechsas_apply_navigation.apply_navigation(techsas_file_path=o_path, nav=nav_data)
