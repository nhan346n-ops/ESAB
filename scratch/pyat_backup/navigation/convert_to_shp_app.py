#! /usr/bin/env python3
# coding: utf-8
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from pynvi.legacy.exporter import Convert2Shp, ConvertNvi2Shp, ShapeExportType
from pynvi.version_2.exporter import ConvertNviV2ToShp

import pyat.utils.application_utils as app_util
from pyat.utils import path_utils
import pyat.utils.pyat_logger as log
from pyat.utils.exceptions.exception_list import BadParameter
from pyat.xsf.xsf_driver import open_xsf
from pyat.mbg.mbg_driver import open_mbg


class ConvertMbg2Shp(Convert2Shp):
    @staticmethod
    def __get_utc_date(julian_date, julian_time):
        """
        Converts julian date to UTC
        """
        epoch = (julian_date - 2440588) * 24 * 3600 + (julian_time / 1000)
        return datetime.utcfromtimestamp(epoch)

    def get_navigation_data_values(
        self, filename
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, datetime, datetime]:
        """
        Reads an MBG file, and fill the provided polyline and feature.
        """
        self.logger.info(f"Input file (mbg) :{filename}")
        with open_mbg(filename) as mbg:
            lon = mbg.read_platform_longitudes()
            lat = mbg.read_platform_latitudes()
            flag_variable = mbg.read_c_flag()  # read the flags variable
            # flag_variable.set_auto_chartostring(False)  # disable char to string autoconversion
            # flag_variable.set_auto_maskandscale(False)  # do not try to compare byte to invalid value as int
            validity_flag = flag_variable[:]  # read the flags
            validity_flag = validity_flag == 0x02  # convert validity to boolean values
            start_date = self.__get_utc_date(mbg.dataset.getncattr("mbStartDate"), mbg.dataset.getncattr("mbStartTime"))
            end_date = self.__get_utc_date(mbg.dataset.getncattr("mbEndDate"), mbg.dataset.getncattr("mbEndTime"))

            time = mbg.read_ping_times()

            if mbg.sounder_file.antenna_count == 2:  # 2 antennas
                # if first antenna is invalid, copy value from second antenna
                validity_flag = np.logical_or(validity_flag[:, 0], validity_flag[:, 1])  # compute validity flag
            else:  # only 1 antenna dimension (only 1 or 2 antennas allowed, if more than that we use only the first
                validity_flag = validity_flag[:, 0]

            return (lon, lat, validity_flag, time, start_date, end_date)


class ConvertXsf2Shp(Convert2Shp):
    @staticmethod
    def __get_utc_date(date: np.datetime64) -> datetime:
        """
        Converts datetime64 to UTC datetime
        """
        return pd.to_datetime(date).to_pydatetime(warn=False)

    def get_navigation_data_values(self, filename) -> Tuple[np.ndarray, np.ndarray, np.ndarray, datetime, datetime]:
        """
        Reads an XSF file, and fill the provided polyline and feature.
        """
        self.logger.info("Input file (xsf) :{filename}")
        with open_xsf(filename) as xsf:
            lon = xsf.read_platform_longitudes()
            lat = xsf.read_platform_latitudes()
            validity_flag = np.any(xsf.read_validity_flags(0, int(xsf.sounder_file.swath_count)), axis=1)
            time = xsf.read_ping_times()

            start_date = self.__get_utc_date(time[0])
            end_date = self.__get_utc_date(time[-1])

            return (lon, lat, validity_flag, time, start_date, end_date)


class Convert2ShpApp:
    """
    This class provides methods to convert XSF/MBG/NVI to shape files.
    """

    def __init__(self, **params):
        """
        Initialize parameters.
        """
        if "i_paths" in params:
            self.input_files = params["i_paths"]
        if "o_path" in params:
            self.output_file = params["o_path"]

        self.campaign = None
        if "campaign" in params:
            self.campaign = params["campaign"]
        self.campaign_number = None
        if "campNum" in params:
            self.campaign_number = params["campNum"]
        self.navigation = None
        if "navigation" in params:
            self.navigation = params["navigation"]
        self.tool = None
        if "tool" in params:
            self.tool = params["tool"]
        if "export_type" in params:
            if params["export_type"] == "points":
                self.export_type = ShapeExportType.POINT
            else:
                self.export_type = ShapeExportType.POLYLINE
        else:
            self.export_type = ShapeExportType.POLYLINE

        self.overwrite = bool(params["overwrite"]) if "overwrite" in params else False
        self.logger = log.logging.getLogger("Convert2Shp")

    def __call__(self):
        # check input file extension
        extension = []
        for f in self.input_files:
            ext = path_utils.ext_of_fname(f)
            if ext not in extension:
                extension.append(ext)
        if len(extension) > 1:
            msg = f"Got several extension mixed {extension}, unsupported case "
            self.logger.error(msg)
            raise BadParameter(msg)

        if len(extension) == 0:
            raise BadParameter(f"No input files {self.input_files}")

        ext = extension[0]
        if "mbg" in ext:
            converter = ConvertMbg2Shp(
                input_files=self.input_files,
                output_file=self.output_file,
                overwrite=self.overwrite,
                campaign=self.campaign,
                campaign_number=self.campaign_number,
                navigation=self.navigation,
                tool=self.tool,
                logger=self.logger,
                export_type=self.export_type,
            )
            converter()
        elif ext.endswith("nvi"):
            converter = ConvertNvi2Shp(
                input_files=self.input_files,
                output_file=self.output_file,
                overwrite=self.overwrite,
                campaign=self.campaign,
                campaign_number=self.campaign_number,
                navigation=self.navigation,
                tool=self.tool,
                logger=self.logger,
                export_type=self.export_type,
            )
            converter()
        elif ext.endswith("xsf.nc"):
            converter = ConvertXsf2Shp(
                input_files=self.input_files,
                output_file=self.output_file,
                overwrite=self.overwrite,
                campaign=self.campaign,
                campaign_number=self.campaign_number,
                navigation=self.navigation,
                tool=self.tool,
                logger=self.logger,
                export_type=self.export_type,
            )
            converter()
        elif ext.endswith("nvi.nc"):
            converter = ConvertNviV2ToShp(
                input_files=self.input_files,
                output_file=self.output_file,
                overwrite=self.overwrite,
                campaign=self.campaign,
                campaign_number=self.campaign_number,
                navigation=self.navigation,
                tool=self.tool,
                logger=self.logger,
                export_type=self.export_type,
            )
            converter()
        else:
            raise BadParameter(f"Unsupported file format extension {ext}")


if __name__ == "__main__":
    # Main method (entry point)
    app_util.launch_application(app_util.get_json_configuration_file(__file__), Convert2ShpApp)
