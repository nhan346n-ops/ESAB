#! /usr/bin/env python3
# coding: utf-8

import logging
import os
from pathlib import Path
import shutil
import tempfile
from collections import namedtuple
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import numpy.testing as npt
import sonar_netcdf.sonar_groups as sg

import pyat.xsf.xsf_driver as xd
from pyat.xsf.xsf_upgrader_from_mbg import XsfUpgrader
from tests.file_test_installer import get_test_path

TEST_FILES_DIR = get_test_path() / "mbg2xsf"

VARS_TO_TEST = {
    sg.DynamicDraughtGrp.TIME(): 0,
    xd.DELTA_DRAUGHT: 2,
    sg.TideGrp.TIME(): 0,
    xd.TIDE_INDICATIVE: 2,
    xd.PING_TIME: 0,
    xd.PLATFORM_VERTICAL_OFFSET: 0,
    xd.WATERLINE_TO_CHART_DATUM: 2,
    xd.TX_TRANSDUCER_DEPTH: 1,
    xd.PLATFORM_HEADING: 2,
    xd.PLATFORM_PITCH: 2,
    xd.PLATFORM_ROLL: 2,
    xd.PLATFORM_LATITUDE: 7,
    xd.PLATFORM_LONGITUDE: 7,
    xd.DETECTION_X: 0,
    xd.DETECTION_Y: 0,
    xd.DETECTION_Z: 0,
    xd.DETECTION_LONGITUDE: 5,  # equivalent to a few centimetres
    xd.DETECTION_LATITUDE: 5,  # equivalent to a few centimetres
}

logger = logging.getLogger()


def _test_upgrade_xsf():
    """ """
    logger.info(f"Execution of {__name__}")
    files_in_error = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        for test_file in _browse_test_files(TEST_FILES_DIR):
            try:
                logger.debug(f"Processing {test_file.mbg_file_name}")
                _upgrade_and_compare(TEST_FILES_DIR, test_file, tmp_dir)
                logger.info(f"{test_file.mbg_file_name} processed with success")
            except AssertionError as ae:
                logger.error(f"{test_file.mbg_file_name} processed with error")
                logger.warning(ae)
                files_in_error += 1

    assert files_in_error == 0, f"Number of files in error : {files_in_error}. See logs"
    logger.info("All tests passed !")


class TestFiles(NamedTuple):
    # Reference MBG. Its data will be transferred into the raw XSF
    mbg_file_name: str
    # Reference XSF with the same corrections than the MBG
    xsf_file_name: str
    # Original XSF, before any correction
    raw_xsf_file_name: str


def _upgrade_and_compare(test_file_dir: str | Path, test_files: TestFiles, tmp_dir: str):
    # Upgraded XSF file is based of the raw XSF
    raw_xsf_file = os.path.join(test_file_dir, test_files.raw_xsf_file_name)
    upgraded_xsf_path = shutil.copy(raw_xsf_file, os.path.join(tmp_dir, os.path.basename(raw_xsf_file)))

    # Launch the upgrade : Raw XSF + Ref MBG = Upgraded XSF
    mbg_file = os.path.join(test_file_dir, test_files.mbg_file_name)
    upgrader = XsfUpgrader(
        i_paths=[raw_xsf_file],
        i_mbg=[mbg_file],
        out_dir=tmp_dir,
        overwrite=True,
    )
    # Perform upgrade only (no cut file generation and ALL conversion)
    upgrader._upgrade(xsf_path=upgraded_xsf_path, mbg_path=mbg_file)
    # shutil.copy(upgraded_xsf_path, os.path.join(r"d:\temp", "upg_" + test_files.xsf_file_name))

    # Upgraded XSF and Ref XSF must be equivalent
    ref_xsf_path = os.path.join(test_file_dir, test_files.xsf_file_name)
    _compare_xsf(os.path.join(test_file_dir, ref_xsf_path), upgraded_xsf_path, VARS_TO_TEST)


def _browse_test_files(test_file_dir: str | Path) -> List[TestFiles]:
    test_files = os.listdir(path=test_file_dir)
    test_files.sort()  # Ensures that reference files precede test files

    mbg_xsf_pairs: Dict[str, TestFiles] = {}
    # test files work in pairs
    # Buils the list of pair (xsf and mbg file)
    mbg_files = [mbg_file for mbg_file in test_files if mbg_file.endswith(".mbg")]
    for mbg_file in mbg_files:
        basename = mbg_file[:-4]
        xsf_file = basename + ".xsf.nc"
        if xsf_file in test_files:
            # We have a pair of mbg/xsf.
            # Search the raw xsf (same file name without suffix)
            raw_xsf_file = xsf_file
            for other_basename, other_test_files in mbg_xsf_pairs.items():
                if basename.startswith(other_basename):
                    raw_xsf_file = other_test_files.raw_xsf_file_name
                    break
            mbg_xsf_pairs[basename] = TestFiles(mbg_file, xsf_file, raw_xsf_file)

    return list(mbg_xsf_pairs.values())


def _compare_xsf(ref_xsf_path: str, test_xsf_path: str, vars_to_test: Dict[str, int]):
    """Open the both XSF, browse and check variables listed in vars_to_test"""
    with xd.open_xsf(ref_xsf_path) as xsf_ref, xd.open_xsf(test_xsf_path) as xsf_upg:
        _compare_status(xsf_ref, xsf_upg)
        for var_to_test, decimal in vars_to_test.items():
            _compare_variable(xsf_ref, xsf_upg, var_to_test, decimal)


def _compare_status(xsf_ref: xd.XsfDriver, xsf_upg: xd.XsfDriver):
    """Check whether invalid status are still invalid"""

    ref_status = xsf_ref[xd.STATUS][:]
    ref_status_detail = xsf_ref[xd.STATUS_DETAIL][:]
    upg_status = xsf_upg[xd.STATUS][:]
    upg_status_detail = xsf_upg[xd.STATUS_DETAIL][:]

    # Invalid status in reference XSF must be any invalid status in upgraded XSF
    upg_status = np.where((ref_status == 2) & (upg_status != 0), ref_status, upg_status)
    # Change the detail in the same way
    upg_status_detail = np.where((ref_status == 2) & (upg_status != 0), ref_status_detail, upg_status_detail)

    # Trace differences in debug mode only
    if logger.isEnabledFor(logging.DEBUG) and not np.array_equal(ref_status, upg_status):
        _trace_difference_status(ref_status, upg_status, xd.STATUS)

    # Status must be strictly equals
    npt.assert_array_equal(
        ref_status,
        upg_status,
        err_msg=f"Difference of validity in '{xd.STATUS}'",
        strict=True,
        verbose=True,
    )

    logger.debug(f"Comparing variable {xd.STATUS_DETAIL}")

    # Trace differences in debug mode only
    if logger.isEnabledFor(logging.DEBUG) and not np.array_equal(ref_status, upg_status):
        _trace_difference_status(ref_status, upg_status, xd.STATUS_DETAIL)

    npt.assert_array_equal(
        ref_status_detail,
        upg_status_detail,
        err_msg=f"Difference in '{xd.STATUS_DETAIL}' values",
        strict=True,
        verbose=True,
    )


def _compare_variable(xsf_ref: xd.XsfDriver, xsf_upg: xd.XsfDriver, var_name: str, decimal: int):
    """Read the variable in the both files and compare the values"""
    ref_var = xsf_ref[var_name][:]
    upg_var = xsf_upg[var_name][:]

    # Manage precision for float32 or float64
    if ref_var.dtype in [np.float32, np.float64] and decimal >= 0:
        logger.debug(f"Comparing variable {var_name}, precision {decimal} decimals")
        if logger.isEnabledFor(logging.DEBUG):
            _trace_difference(ref_var, upg_var, var_name, decimal)

        npt.assert_array_almost_equal(
            ref_var, upg_var, err_msg=f"Difference in '{var_name}' values", decimal=decimal, verbose=True
        )
    else:
        logger.debug(f"Comparing variable {var_name} strictly")
        npt.assert_array_equal(
            ref_var, upg_var, err_msg=f"Difference in '{var_name}' values", strict=True, verbose=True
        )


def _trace_difference_status(ref_status: np.ndarray, upg_status: np.ndarray, var_name: str):
    """Trace statistics of differences in status values"""
    logger.debug(f"Differences in {var_name} :")
    unique, counts = np.unique(ref_status, return_counts=True)
    logger.debug(" - Ref XSF :" + repr(dict(zip(unique, counts))))
    unique, counts = np.unique(upg_status, return_counts=True)
    logger.debug(" - UPG XSF :" + repr(dict(zip(unique, counts))))

    all_diff_coords = np.nonzero(ref_status - upg_status)
    ref_status = ref_status[all_diff_coords]
    upg_status = upg_status[all_diff_coords]

    # status : tuple of int, ie (status in reference xsf, status in upgraded xsf)
    # count : number of sounds that have changed status in this way
    # coords : indice of the first sound found
    Diff = namedtuple("Diff", ["status", "count", "coords"])
    diffs: Dict[Tuple, Diff] = {}
    for coords_1, coords_2, ref_s, upg_s in np.nditer([all_diff_coords[0], all_diff_coords[1], ref_status, upg_status]):
        status_pair = (int(ref_s), int(upg_s))
        if status_pair in diffs:
            d = diffs[status_pair]
            diffs[status_pair] = Diff(status=d.status, count=d.count + 1, coords=d.coords)
        else:
            diffs[status_pair] = Diff(status=status_pair, count=1, coords=(str(coords_1), str(coords_2)))

    for diff in diffs.values():
        logger.debug(
            f" From Ref status {diff.status[0]} to UPG status {diff.status[1]} : {diff.count}. (ex : {diff.coords})"
        )


def _trace_difference(ref_var: np.ndarray, upg_var: np.ndarray, var_name: str, decimal: float):
    diffs = abs(ref_var - upg_var)
    max_diff = np.max(diffs)
    if max_diff > pow(10, -decimal):
        max_coords = np.nonzero(diffs == max_diff)
        # Creates a tuple of coordinates of the max difference
        max_coord = tuple(c[0] for c in max_coords)
        logger.debug(f"Max difference ({max_diff}) in {var_name} at {repr(max_coord)} ")
        logger.debug(f" - Ref XSF : {ref_var[max_coord]}")
        logger.debug(f" - UPG XSF : {upg_var[max_coord]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s", force=True)
    _test_upgrade_xsf()
