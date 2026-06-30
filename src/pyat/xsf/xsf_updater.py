#! /usr/bin/env python3
# coding: utf-8

import datetime
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Union

import numpy as np
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
import pyat.xsf.xsf_driver as xsf_driver
from pyat.xsf.migration.history_updater import HistoryUpdater

# List of all XSF variable to transfer from reference file to target file
VARS_TO_TRANSFER = [
    xsf_driver.STATUS,
    xsf_driver.STATUS_DETAIL,
    xsf_driver.DELTA_DRAUGHT,
    xsf_driver.TIDE_INDICATIVE,
    xsf_driver.PLATFORM_VERTICAL_OFFSET,
    xsf_driver.WATERLINE_TO_CHART_DATUM,
    xsf_driver.TX_TRANSDUCER_DEPTH,
    xsf_driver.PLATFORM_HEADING,
    xsf_driver.PLATFORM_PITCH,
    xsf_driver.PLATFORM_ROLL,
    xsf_driver.PLATFORM_LATITUDE,
    xsf_driver.PLATFORM_LONGITUDE,
    xsf_driver.DETECTION_X,
    xsf_driver.DETECTION_Y,
    xsf_driver.DETECTION_Z,
    xsf_driver.DETECTION_LONGITUDE,
    xsf_driver.DETECTION_LATITUDE,
    xsf_driver.DETECTION_BACKSCATTER_R,
]

MINIMUM_XSF_VERSION = 0.2


class XsfUpdater:
    """
    Callable used by pyat/app to launch an upgrade of XSF files.
    This class aim to report validity flags and bias corrections on a set of xsf files.

    Update depends on version on reference and input files.
    When a new version of XSF is available, the following dicts can be completed to bring the possible conversions
     - __check_variables_funcs : which variables to update
     - __update_history_funcs : how to update history
     - __transfer_variables : how to transfer the variables
    """

    def __call__(self) -> None:
        """Run method."""
        self.monitor.set_work_remaining(len(self.i_paths))
        begin = datetime.datetime.now()
        file_in_error = []

        for i_xsf, o_xsf, i_ref in zip(self.i_paths, self.o_paths, self.i_refs):
            try:
                self.logger.info(f"Starting to migrate {i_xsf}")
                self.logger.info(f"\tto {o_xsf}")
                self.logger.info(f"\tusing reference file {i_ref}")

                if not os.path.exists(i_ref):
                    self.logger.warning("Unknown reference file. Migration aborted.")
                    file_in_error.append(i_xsf)
                    continue

                if not self.overwrite and os.path.exists(o_xsf):
                    self.logger.warning("File exists and overwrite is not allowed. Migration aborted.")
                    file_in_error.append(i_xsf)
                    continue

                # input file == output file => this is an update
                updating_i_xsf = os.path.exists(o_xsf) and os.path.samefile(i_xsf, o_xsf)
                if not updating_i_xsf:
                    # Copy input xsf to o_xsf before migrating
                    shutil.copy(i_xsf, o_xsf)

                now = datetime.datetime.now()
                with xsf_driver.open_xsf(i_ref) as i_xsf_driver, xsf_driver.open_xsf(o_xsf, "r+") as o_xsf_driver:
                    ref_version, out_version = self.__check_version(i_xsf_driver, o_xsf_driver)

                    # Get the variables to transfer
                    check_variables_func = self._get_suitable_function(ref_version, self.__check_variables_funcs)
                    self.logger.debug(f"Checking variables with {str(check_variables_func)}()")
                    vars_to_transfer = check_variables_func(i_xsf_driver, o_xsf_driver)

                    # transfer History
                    update_history_func = self._get_suitable_function(ref_version, self.__update_history_funcs)
                    self.logger.debug(f"Process history with {update_history_func.__name__}()")
                    update_history_func(i_xsf_driver, o_xsf_driver)

                    # add new history line for the update process
                    o_xsf_driver.append_history_line(
                        f"Updated to XSF format {out_version} with {self.__class__.__name__}"
                    )

                    # transfer processing status
                    self._transfer_proc_status(i_xsf_driver, o_xsf_driver)

                    # Transfer tide group attributes
                    self._transfer_group_attributes(i_xsf_driver, o_xsf_driver, xsf_driver.TIDE_GROUP)

                    # Process transfer
                    transfer_variables_func = self._get_suitable_function(ref_version, self.__transfer_variables_funcs)
                    self.logger.debug(f"Transferring variables with {transfer_variables_func.__name__}()")
                    transfer_variables_func(i_xsf_driver, o_xsf_driver, vars_to_transfer)

                self.monitor.worked(1)
                self.logger.info(f"End of migration of {i_xsf} : {datetime.datetime.now() - now} time elapsed\n")

            except ValueError as e:
                file_in_error.append(i_xsf)
                self.logger.error(str(e))
            except Exception as e:
                file_in_error.append(i_xsf)
                self.logger.error(f"An exception was thrown : {str(e)}", exc_info=True, stack_info=True)

        self.monitor.done()
        process_util.log_result(self.logger, begin, file_in_error)

    def __check_version(self, ref_xsf: xsf_driver.XsfDriver, o_xsf: xsf_driver.XsfDriver) -> tuple[float, float]:
        """
        Check version of XSF files
        Return a float value representing the version of the reference file
        """
        try:
            ref_version = float(ref_xsf.dataset.xsf_convention_version)
        except ValueError as e:
            raise ValueError(f"Unsupported XSF version for the reference file") from e
        try:
            o_version = float(o_xsf.dataset.xsf_convention_version)
        except ValueError as e:
            raise ValueError(f"Unsupported XSF version for the input file") from e

        if ref_version < MINIMUM_XSF_VERSION:
            raise ValueError(f"The version must be at least {MINIMUM_XSF_VERSION} for the reference file")
        if o_version < MINIMUM_XSF_VERSION:
            raise ValueError(f"The version must be at least {MINIMUM_XSF_VERSION} for file to update")

        # Can't migrate data from a newer version of file
        if o_version < ref_version:
            raise ValueError(f"Downgrading XSF is not intended")

        self.logger.info(f"Processing update from XSF version {ref_version} to {o_version}")

        return ref_version, o_version

    def __check_variables(self, ref_xsf: xsf_driver.XsfDriver, o_xsf: xsf_driver.XsfDriver) -> List[str]:
        """
        Check the dimensions variable to transfer. Must be the same of both files
        Return the list of correct variables to transfer
        """
        self.logger.info(f"Checking variables...")

        result = []
        for var_to_check in VARS_TO_TRANSFER:
            self.logger.debug(f"Process variable {var_to_check}")
            i_var = ref_xsf.get_layer(var_to_check)
            if i_var is None:
                self.logger.warning(f"Variable {var_to_check} skipped : not present in the reference file")
                continue
            o_var = o_xsf.get_layer(var_to_check)
            if o_var is None:
                self.logger.warning(f"Variable {var_to_check} skipped : not present in the target file")
                continue

            # Check dimensions
            i_dims = {dim.name: dim.size for dim in i_var.get_dims()}
            o_dims = {dim.name: dim.size for dim in o_var.get_dims()}
            if i_dims != o_dims:
                self.logger.warning(f"Variable {var_to_check} skipped : not the same dimension definition")
                continue

            result.append(var_to_check)

        if not result:
            raise ValueError("All layers were rejected by the control process : transfer aborted")

        return result

    def __transfer_variables(
        self, ref_xsf: xsf_driver.XsfDriver, o_xsf: xsf_driver.XsfDriver, vars_to_transfer: List[str]
    ) -> None:
        """
        transfer variables from reference file to output files.
        No transformation is applied on values
        """
        self.logger.info("Transferring variables...")
        for var_to_transfer in vars_to_transfer:
            self.logger.debug(f"Process variable {var_to_transfer}")
            i_var = ref_xsf.get_layer(var_to_transfer)
            o_var = o_xsf.get_layer(var_to_transfer)
            o_var[:] = i_var[:]
            self.logger.debug(f"{var_to_transfer} transferred")

    def _get_suitable_function(self, version: float, functions: Dict[float, Callable]):
        """
        Method responsible for finding the suitable function to run for the given version number
        """
        if len(functions) == 1:
            # Only one function : this is the default one
            return next(iter(functions.values()))

        versions = np.array(list(functions.keys()))
        greater_or_equal_version = np.max(np.ma.masked_greater(versions, version))
        if np.ma.is_masked(greater_or_equal_version):
            raise ValueError("No suitable updating function found : transfer aborted")

        return functions[greater_or_equal_version]

    def _remove_xsf_suffix(self, path: str) -> str:
        filename = os.path.basename(path)
        filename = filename[:-3] if filename.lower().endswith(".nc") else filename
        filename = filename[:-4] if filename.lower().endswith(".xsf") else filename
        return filename

    def _transfer_proc_status(self, ref_xsf: xsf_driver.XsfDriver, o_xsf: xsf_driver.XsfDriver) -> None:
        ref_processing_status = ref_xsf.get_processing_status()
        o_xsf.update_processing_status(status_dict=ref_processing_status)

    def _transfer_group_attributes(
        self, ref_xsf: xsf_driver.XsfDriver, o_xsf: xsf_driver.XsfDriver, group_path: str
    ) -> None:
        # Transfer group attributes only
        ref_group = ref_xsf.get_group(group_path)
        o_group = o_xsf.get_group(group_path)
        if ref_group is None or o_group is None:
            self.logger.warning(f"Group {group_path} not found in reference or target file : group skipped")
            return
        for attr_name in ref_group.ncattrs():
            o_group.setncattr(attr_name, ref_group.getncattr(attr_name))

    def __init__(
        self,
        i_paths: List[str],
        o_paths: List[str],
        i_ref: Union[str, List[str]],
        overwrite: bool = False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor
        i_ref may by
         - a forder (str). In this case, the reference files are searched there
         - the list of reference files to use. in that case, len(i_paths) == len(o_paths) == len(i_ref)
        """
        self.logger = log.logging.getLogger(self.__class__.__name__)
        self.monitor = monitor

        # Parsing parameters
        self.i_paths = arg_util.parse_list_of_files("i_paths", i_paths, True)
        self.overwrite = overwrite
        self.o_paths = arg_util.parse_list_of_files("o_paths", o_paths, False)

        self.i_refs: List[str] = []
        if isinstance(i_ref, list):
            self.i_refs = i_ref
        elif os.path.isdir(i_ref):  # Forder as reference base
            for i_xsf, o_xsf in zip(self.i_paths, self.o_paths):
                # Search reference file (same name than the output file)
                ref_xsf = os.path.join(i_ref, os.path.basename(o_xsf))
                if not os.path.exists(ref_xsf):
                    # Search reference file (same name than the input file)
                    ref_xsf = os.path.join(i_ref, os.path.basename(i_xsf))
                if not os.path.exists(ref_xsf):
                    # Search reference file with any prefix / suffix
                    ref_basename = self._remove_xsf_suffix(i_xsf)
                    for one_ref_file in Path(i_ref).iterdir():
                        if not one_ref_file.is_dir() and ref_basename in one_ref_file.name:
                            ref_xsf = one_ref_file
                            break
                self.i_refs.append(ref_xsf)
        else:
            raise ValueError(f"Invalid value for argument {i_ref} : not a folder")

        # Map all checking variable functions. Designates the function to call from a starting version number
        self.__check_variables_funcs = {
            MINIMUM_XSF_VERSION: self.__check_variables,  # Default behavior
        }
        # Map all functions to transfer variables. Designates the function to call from a starting version number
        self.__transfer_variables_funcs = {
            MINIMUM_XSF_VERSION: self.__transfer_variables,  # Default behavior
        }
        # Map all functions to update history metadata. Designates the function to call from a starting version number
        historyUpdater = HistoryUpdater()
        self.__update_history_funcs = {
            0.2: historyUpdater.update_history_from_0_2,  # Manage history from version 0.2
            0.3: historyUpdater.update_history_from_0_3,  # Manage history from version 0.3
        }
