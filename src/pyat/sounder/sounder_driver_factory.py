#! /usr/bin/env python3
# coding: utf-8

import os

from pyat.mbg.mbg_driver import MbgDriver
from pyat.sounder.sounder_driver import SounderDriver
from pyat.xsf.xsf_driver import XsfDriver


def open_sounder(file_path: str, mode: str = "r"):
    """
    Define a With Statement Context Managers for a SounderDriver
    Allow opening a SounderDriver in a With Statement
    """
    driver = get_sounder_driver(file_path)

    class ContextManager:
        def __enter__(self):
            if not driver is None:
                driver.open(mode)
            return driver

        def __exit__(self, exc_type, exc_value, traceback):
            if not driver is None:
                driver.close()

    return ContextManager()


def get_sounder_driver(file_path: str) -> SounderDriver:
    """
    Instanciates a SounderDriver suitable for the specified file
    """
    file_extension = os.path.splitext(file_path)[1]

    if file_extension == ".mbg":
        return MbgDriver(file_path)
    elif file_extension in (".nc", ".xsf"):
        return XsfDriver(file_path)

    raise ValueError(f"Unsupported sounder file : {file_path}.")
