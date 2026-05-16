#! /usr/bin/env python3
# coding: utf-8

import locale

from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.dtm.export import cython_dtm2ascii_export as p


class VariableParser:
    def __init__(self, i_dtm_driver: dtm_driver.DtmDriver, layer_name: str):
        self.variable = None
        self.line_values = None
        if layer_name in i_dtm_driver:
            self.variable = i_dtm_driver[layer_name]
            self.variable.set_auto_mask(False)

    def data(self):
        """return dataset if not empty, None otherwise"""
        if self.variable is not None:
            return self.variable[:]
        return None


def filter_cdi(x: str):
    if "CDI" in x:
        before, sep, after = x.rpartition(":")
        return after
    return ""


def filter_cprd(x: str):
    if "CPRD" in x:
        before, sep, after = x.rpartition(":")
        return after
    elif "interpolated" in x:  # EMODNET demands to write int instead of interpolate (file space purpose)
        return "INT"
    return ""


class Dtm2Ascii:
    """ "
    Export a dtm to ascii file in emo format
    Projection settings are kept, ie if the dtm is projected, its coordinates are in projection Crs
    """

    def __init__(
        self,
        i_paths: list,
        o_paths: list = None,
        export_missing_values: bool = False,
        overwrite: bool = False,
        column_separator: str = ";",
        other_separator: str = ";",
        decimal_separator: str = "dot",
        column_order: str = "XYZ",
        monitor=DefaultMonitor,
    ):
        """Init method."""
        self.i_paths = i_paths
        self.o_paths = o_paths
        # tell if we export data where bathymetry is invalid
        self.export_missing_values = export_missing_values
        self.overwrite = overwrite

        if self.is_exporting_to_xyz():
            seps = {"semicolon": ";", "comma": ",", "space": " ", "tabulation": "\t", "other": "O"}
            if column_separator.lower() in seps:
                self.column_separator = seps[column_separator.lower()]
                if self.column_separator == "O":
                    self.column_separator = other_separator
            else:
                raise ValueError(
                    f"Bad value for the parameter column_separator : '{column_separator}'. Expecting one of (semicolon, comma, space, tabulation, other)"
                )

            try:
                self.column_order = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"].index(column_order)
            except ValueError as exc:
                raise ValueError(
                    f"Bad value for the parameter column_order : '{column_order}'. Expecting one of (XYZ, XZY, YXZ, YZX, ZXY, ZYX)",
                ) from exc

            if decimal_separator.lower() == "comma":
                # Change local for using comma in cython fprintf
                locale.setlocale(locale.LC_ALL, "fr_FR")
            elif decimal_separator.lower() != "dot":
                raise ValueError(
                    f"Bad value for the parameter decimal_separator : '{decimal_separator}'. Expecting one of (dot, comma)"
                )

        self.monitor = monitor

        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __process_data(self, i_dtm_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor) -> None:

        # Projected DTM can't be exported as EMO
        if not self.is_exporting_to_xyz() and i_dtm_driver.dtm_file.spatial_reference.IsProjected():
            raise ValueError(f"Unable to export a projected file ({i_dtm_driver.dtm_file.file_path})")

        ind = self.i_paths.index(i_dtm_driver.dtm_file.file_path)
        o_path = arg_util.create_output_path(
            i_dtm_driver.dtm_file.file_path,
            extension=".xyz" if self.is_exporting_to_xyz() else ".emo",
            overwrite=self.overwrite,
            o_path=(None if not self.o_paths else self.o_paths[ind]),
        )

        self.logger.info(f"Creating file {o_path}")
        x_variable = i_dtm_driver.get_y_axis()
        x_variable.set_auto_mask(False)
        x_axis = x_variable[:]
        y_variable = i_dtm_driver.get_x_axis()
        y_variable.set_auto_mask(False)
        y_axis = y_variable[:]

        elevation_variable = VariableParser(i_dtm_driver, DtmConstants.ELEVATION_NAME)
        elevation_min_variable = VariableParser(i_dtm_driver, DtmConstants.ELEVATION_MIN)
        elevation_max_variable = VariableParser(i_dtm_driver, DtmConstants.ELEVATION_MAX)
        stdev_variable = VariableParser(i_dtm_driver, DtmConstants.STDEV)

        cdi_array = []
        if DtmConstants.CDI in i_dtm_driver:
            cdi_array = i_dtm_driver[DtmConstants.CDI][:]
            cdi_array = cdi_util.trim_string_array(cdi_array)

        value_count_variable = VariableParser(i_dtm_driver, DtmConstants.VALUE_COUNT)

        interpolation_flag_variable = VariableParser(i_dtm_driver, DtmConstants.INTERPOLATION_FLAG)
        smoothed_depth = VariableParser(i_dtm_driver, DtmConstants.ELEVATION_SMOOTHED_NAME)

        path = o_path.encode("utf-8")
        if self.is_exporting_to_xyz():
            p.export_xyz(
                path,
                self.export_missing_values,
                y_axis,
                x_axis,
                elevation_variable.data(),
                self.column_separator,
                self.column_order,
            )
        else:
            # export in emodnet file format
            # compute CDI indexes

            cdi_only_array = [filter_cdi(v) for v in cdi_array]
            cprd_array = [filter_cprd(v) for v in cdi_array]

            p.export_emo(
                path,
                self.export_missing_values,
                y_axis,
                x_axis,
                elevation_variable.data(),
                elevation_min_variable.data(),
                elevation_max_variable.data(),
                stdev_variable.data(),
                value_count_variable.data(),
                interpolation_flag_variable.data(),
                smoothed_depth.data(),
                VariableParser(i_dtm_driver, DtmConstants.CDI_INDEX).data(),
                cdi_only_array,
                cprd_array,
            )
        monitor.done()

    def is_exporting_to_xyz(self):
        """
        Return True to generate a xyz file. False for an emo file
        """
        return True

    def __call__(self) -> None:
        process_util.process_each_input_file_in_read_mode(
            self.i_paths,
            self.__class__.__name__,
            self.logger,
            self.monitor,
            self.__process_data,
        )
