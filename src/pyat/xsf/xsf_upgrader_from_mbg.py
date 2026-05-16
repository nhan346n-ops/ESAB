#! /usr/bin/env python3
# coding: utf-8
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import netCDF4 as nc
import numpy as np
import pygws.service.execution_context as exec_ctx
import sonar_netcdf.process.sonar_file_merger as sfm
import sonar_netcdf.sonar_groups as sg
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from sonar_netcdf.utils import nc_merger as nc_m

import pyat.dtm.utils.process_utils as process_util
import pyat.mbg.mbg_driver as md
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
import pyat.xsf.xsf_driver as xd
from pyat.mbg.mbg_to_cut import CutMbg
from pyat.utils.path_utils import basename_of_fname
from pyat.xsf.netcdf_merger_bridge import NcMergerBridge


class XsfUpgrader:
    """
    Callable used by pyat/app to launch an upgrade of XSF files from MBG files.
    This class aim to report validity flags and corrections on a set of XSF files.
    First, XSF files are cut in time to match MBG files.
    Then cut XSF files are upgraded with flags and corrections retrieved in MBG files.
    """

    def __init__(
        self,
        i_paths: List[str],
        out_dir: str,
        i_mbg: List[str],
        overwrite: bool = False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor
        """
        self.logger = log.logging.getLogger(self.__class__.__name__)
        self.logger.info("Preparing upgrade...")

        self.monitor = monitor

        # Parsing parameters
        self.i_paths = arg_util.parse_list_of_files("i_paths", i_paths, True)
        self.overwrite = overwrite

        self.out_dir = Path(out_dir)
        if not self.out_dir.exists():
            os.makedirs(self.out_dir)
        if not self.out_dir.is_dir():
            raise ValueError(f"{self.out_dir} : is not a directory. Process aborted")

        self.i_mbg = arg_util.parse_list_of_files("i_mbg", i_mbg, True)

    def __call__(self) -> None:
        """Run method."""
        self.logger.info("Start upgrading...")

        self.monitor.set_work_remaining(len(self.i_paths) + 1)
        begin = datetime.now()

        mbg_timelines = self._generate_line_from_mbg()
        if len(mbg_timelines) != len(self.i_mbg):
            self.logger.error(
                f"Cut line number ({len(mbg_timelines)}) differs from number of MBG files ({len(self.i_mbg)}). Process aborted"
            )
            return

        cut_xsf = self._cut_xsf(mbg_timelines)
        # the above merging/cutting step can produce several xsf files for one mbg file (xx.mbg --> [xxx_1.xf.nc, xxx_2.xsf.nc, ...]).
        # logic to handle 1 mbg --> n xsf files
        cut_xsf_timelines = nc_m.time_sort_files(cut_xsf)
        for cut_xsf_timeline in cut_xsf_timelines:
            cut_xsf_basename = basename_of_fname(cut_xsf_timeline[0])
            for mbg in self.i_mbg:
                mbg_basename = basename_of_fname(mbg)
                if cut_xsf_basename.startswith(mbg_basename):
                    # upgrade XSF with the corresponding time interval of MBG
                    self._upgrade(
                        cut_xsf_timeline[0], mbg, mbg_start_time=cut_xsf_timeline[1], mbg_stop_time=cut_xsf_timeline[2]
                    )
                    break

        self.monitor.done()
        process_util.log_result(self.logger, begin, [])

        # Using rsocket (if present) to send the result
        rsocket_msg_emitter = exec_ctx.get_rsocket_msg_emitter()
        if rsocket_msg_emitter is not None:
            rsocket_msg_emitter.emit_files(cut_xsf)

    def _upgrade(
        self, xsf_path: str, mbg_path: str, mbg_start_time: datetime = None, mbg_stop_time: datetime = None
    ) -> None:
        with (
            xd.open_xsf(xsf_path, mode="a") as xsf_file,
            md.open_mbg(mbg_path) as mbg_file,
        ):
            # retrieve mbg swath indexes from start/stop time interval
            if mbg_start_time is not None and mbg_stop_time is not None:
                mbg_swath_indexes = mbg_file.get_swath_indexes_from_time(
                    np.datetime64(mbg_start_time.replace(tzinfo=None)),
                    np.datetime64(mbg_stop_time.replace(tzinfo=None)),
                )
            else:
                mbg_swath_indexes = np.arange(xsf_file.sounder_file.swath_count)

            mbg_antenna_indexes = self._get_antenna_indexes(mbg_file, mbg_swath_indexes)

            if mbg_file.has_tide_correction() or mbg_file.has_draught_correction():
                self._apply_tide_draught_correction(xsf_file, mbg_file, mbg_swath_indexes, mbg_antenna_indexes)

            if (
                mbg_file.has_tide_correction()
                or mbg_file.has_draught_correction()
                or mbg_file.has_bias_correction()
                or mbg_file.has_position_correction()
            ):
                self._apply_depth_correction(xsf_file, mbg_file, mbg_swath_indexes, mbg_antenna_indexes)

            if mbg_file.has_automatic_cleaning() or mbg_file.has_manual_cleaning():
                self._report_status(xsf_file, mbg_file, mbg_swath_indexes)
            self._report_attitude(xsf_file, mbg_file, mbg_swath_indexes, mbg_antenna_indexes)

            if mbg_file.has_bias_correction() or mbg_file.has_velocity_correction():
                self._report_detection_x_y_z(xsf_file, mbg_file, mbg_swath_indexes, mbg_antenna_indexes)

            if (
                mbg_file.has_bias_correction()
                or mbg_file.has_velocity_correction()
                or mbg_file.has_position_correction()
            ):
                self._report_detection_lon_lat(xsf_file, mbg_file, mbg_swath_indexes)

            if mbg_file.has_position_correction():
                self._report_navigation(xsf_file, mbg_file, mbg_swath_indexes)

            self._report_history(xsf_file, mbg_file)
            self._report_correction_flags(xsf_file, mbg_file)

    def _generate_line_from_mbg(self) -> List[nc_m.Timeline]:
        """
        Return one cut line for each mbg file.
        """
        self.logger.info("Compute lines from MBG files...")
        cut_mbg = CutMbg(i_paths=self.i_mbg, monitor=self.monitor.split(1))
        lines = cut_mbg.cut_input_files()

        result = [
            nc_m.Timeline(
                name=Path(mbg).stem,
                # We remove 1ms to get around the problem of rounding hours to the millisecond in MBG file.
                start=line[0] - timedelta(milliseconds=1),
                stop=line[1],
            )
            for line, mbg in zip(lines, self.i_mbg)
        ]
        for timeline in result:
            self.logger.info(f"line {timeline.name} from {timeline.start} to {timeline.stop}")
        return result

    def _cut_xsf(self, timelines: List[nc_m.Timeline]) -> List[str]:
        """Cut the XSF files in input."""
        self.logger.info("Cutting XSF files...")
        cut_xsf_files = [os.path.join(self.out_dir, (timeline.name + ".xsf.nc")) for timeline in timelines]

        xsf_cutter = NcMergerBridge(
            nc_merger_class=sfm.SNMerger,
            i_paths=self.i_paths,
            o_paths=cut_xsf_files,
            timelines=timelines,
            monitor=self.monitor.split(1),
            overwrite=self.overwrite,
        )
        cut_xsf_files = xsf_cutter()

        return cut_xsf_files["outfile"]

    def _apply_tide_draught_correction(
        self,
        xsf_file: xd.XsfDriver,
        mbg_file: md.MbgDriver,
        mbg_swath_indexes: np.ndarray,
        mbg_antenna_indexes: np.ndarray,
    ) -> None:
        """Reports tide and draught corrections from MBG to XSF"""
        tide = mbg_file.read_tide()[mbg_swath_indexes, mbg_antenna_indexes]
        xsf_file[xd.TIDE_INDICATIVE][:] = tide
        xsf_file[xd.WATERLINE_TO_CHART_DATUM][:] = tide

        draught = mbg_file.read_dynamic_draught()[mbg_swath_indexes, mbg_antenna_indexes]
        xsf_file[xd.DELTA_DRAUGHT][:] = draught

    def _apply_depth_correction(
        self,
        xsf_file: xd.XsfDriver,
        mbg_file: md.MbgDriver,
        mbg_swath_indexes: np.ndarray,
        mbg_antenna_indexes: np.ndarray,
    ) -> None:
        """Reports layer update in tide, draught, Bias or Navigation corrections from MBG to XSF"""

        platform_vertical_offset = mbg_file.read_platform_vertical_offsets()[mbg_swath_indexes]
        xsf_file[xd.PLATFORM_VERTICAL_OFFSET][:] = platform_vertical_offset

        transducter_depth = mbg_file.read_reference_depth()[mbg_swath_indexes, mbg_antenna_indexes]
        xsf_file[xd.TX_TRANSDUCER_DEPTH][:] = transducter_depth

    def _report_status(self, xsf_file: xd.XsfDriver, mbg_file: md.MbgDriver, mbg_swath_indexes: np.ndarray) -> None:
        """Compute status from MBG flag layers and report them"""
        mbg_status, mbg_details = mbg_file.translate_flags_to_xsf_status_and_details(
            mbg_swath_indexes[0], mbg_swath_indexes[-1] + 1
        )

        xsf_status, xsf_details = xsf_file[xd.STATUS][:], xsf_file[xd.STATUS_DETAIL][:]
        # Make sure to keep INVALID_ACQUISITION in the XSF
        xsf_status = np.where(xsf_status == xd.STATUS_INVALID_ACQUIS, xsf_status, mbg_status)
        xsf_details = np.where(xsf_status == xd.STATUS_INVALID_ACQUIS, xsf_details, mbg_details)

        xsf_file[xd.STATUS][:] = xsf_status[:]
        xsf_file[xd.STATUS_DETAIL][:] = xsf_details[:]

    def _report_attitude(
        self,
        xsf_file: xd.XsfDriver,
        mbg_file: md.MbgDriver,
        mbg_swath_indexes: np.ndarray,
        mbg_antenna_indexes: np.ndarray,
    ) -> None:
        """Report platform heading, pitch and roll"""
        heading = mbg_file.read_heading()[mbg_swath_indexes, mbg_antenna_indexes]
        xsf_file[xd.PLATFORM_HEADING][:] = heading

        pitch = mbg_file.read_pitch()[mbg_swath_indexes, mbg_antenna_indexes]
        xsf_file[xd.PLATFORM_PITCH][:] = pitch

        roll = mbg_file.read_roll()[mbg_swath_indexes, mbg_antenna_indexes]
        xsf_file[xd.PLATFORM_ROLL][:] = roll

    def _report_detection_x_y_z(
        self,
        xsf_file: xd.XsfDriver,
        mbg_file: md.MbgDriver,
        mbg_swath_indexes: np.ndarray,
        mbg_antenna_indexes: np.ndarray,
    ) -> None:
        """Report across/along distances/depth of detections"""
        detection_x = mbg_file.read_along_distances(mbg_swath_indexes[0], mbg_swath_indexes[-1] + 1)
        xsf_file[xd.DETECTION_X][:] = detection_x

        detection_y = mbg_file.read_across_distances(mbg_swath_indexes[0], mbg_swath_indexes[-1] + 1)
        xsf_file[xd.DETECTION_Y][:] = detection_y

        depth = mbg_file.read_depth(mbg_swath_indexes[0], mbg_swath_indexes[-1] + 1)
        platform_vertical_offset = mbg_file.read_platform_vertical_offsets()[mbg_swath_indexes]
        platform_vertical_offset = platform_vertical_offset.reshape((xsf_file.sounder_file.swath_count, 1))
        tide = mbg_file.read_tide()[mbg_swath_indexes, mbg_antenna_indexes]
        tide = tide.reshape((xsf_file.sounder_file.swath_count, 1))
        detection_z = depth + platform_vertical_offset + tide
        xsf_file[xd.DETECTION_Z][:] = detection_z[:]

    def _report_detection_lon_lat(
        self, xsf_file: xd.XsfDriver, mbg_file: md.MbgDriver, mbg_swath_indexes: np.ndarray
    ) -> None:
        """Report latitude and longitude of detections"""
        xsf_file[xd.DETECTION_LONGITUDE][:] = mbg_file.read_detection_longitude()[mbg_swath_indexes, :]
        xsf_file[xd.DETECTION_LATITUDE][:] = mbg_file.read_detection_latitude()[mbg_swath_indexes, :]

    def _report_navigation(self, xsf_file: xd.XsfDriver, mbg_file: md.MbgDriver, mbg_swath_indexes: np.ndarray) -> None:
        """Report latitude and longitude of navigation"""
        xsf_file[xd.PLATFORM_LATITUDE][:] = mbg_file.read_platform_latitudes()[mbg_swath_indexes]
        xsf_file[xd.PLATFORM_LONGITUDE][:] = mbg_file.read_platform_longitudes()[mbg_swath_indexes]

    def _report_history(self, xsf_file: xd.XsfDriver, mbg_file: md.MbgDriver) -> None:
        """Report the MBG history and complete with this upgrade process"""

        hist_julian_date = mbg_file.dataset[md.HIST_DATE]
        hist_time_in_ms = mbg_file.dataset[md.HIST_TIME]
        mbg_hist_autor = mbg_file.dataset[md.HIST_AUTOR]

        xsf_history = []
        for hist_index in range(mbg_file.dataset.mbNbrHistoryRec):
            hist_datetime = datetime.fromtimestamp(
                (hist_julian_date[hist_index] - 2440588) * 24 * 3600 + (hist_time_in_ms[hist_index] / 1000),
                timezone.utc,
            )
            xsf_history.append(
                f"{hist_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')} {str(nc.chartostring(mbg_hist_autor[hist_index]))}"
            )

            xsf_history.append(
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} Upgrade from {mbg_file.get_file_path()}"
            )
        provenance_grp = xsf_file[sg.ProvenanceGrp.get_group_path()]
        provenance_grp.history = xsf_history

    def _report_correction_flags(self, xsf_file: xd.XsfDriver, mbg_file: md.MbgDriver) -> None:
        """Report the MBG correction flags"""

        flag_mapping = {
            xd.ATT_PROCESSING_STATUS_AUTOMATIC_CLEANING: mbg_file.has_automatic_cleaning(),
            xd.ATT_PROCESSING_STATUS_BIAS_CORRECTION: mbg_file.has_bias_correction(),
            xd.ATT_PROCESSING_STATUS_DRAUGHT_CORRECTION: mbg_file.has_draught_correction(),
            xd.ATT_PROCESSING_STATUS_MANUAL_CLEANING: mbg_file.has_manual_cleaning(),
            xd.ATT_PROCESSING_STATUS_POSITION_CORRECTION: mbg_file.has_position_correction(),
            xd.ATT_PROCESSING_STATUS_TIDE_CORRECTION: mbg_file.has_tide_correction(),
            xd.ATT_PROCESSING_STATUS_VELOCITY_CORRECTION: mbg_file.has_velocity_correction(),
        }

        for xsf_flag, mbg_flag in flag_mapping.items():
            self._update_correction_flag(xsf_file, xsf_flag, mbg_flag)

    def _update_correction_flag(self, xsf_file: xd.XsfDriver, xsf_flag, mbg_flag: bool) -> None:
        xsf_flag_value = xd.ATT_PROCESSING_STATUS_FLAG_ON if mbg_flag else xd.ATT_PROCESSING_STATUS_FLAG_OFF
        xsf_file.update_processing_status({xsf_flag: xsf_flag_value})

    def _get_antenna_indexes(self, mbg: md.MbgDriver, mbg_swath_indexes: np.ndarray) -> np.ndarray:
        """Return the array with the first valid antenna index for each swath, or 0 if no one valid"""
        cycle_flags = mbg.read_c_flag()[mbg_swath_indexes]
        for antenna_idx in range(mbg.sounder_file.antenna_count):
            cycle_of_antenna = cycle_flags[:, antenna_idx]
            # Set the antenna index where cycle flag is valid
            cycle_flags[:, antenna_idx] = np.where(cycle_of_antenna == 2, antenna_idx, mbg.sounder_file.antenna_count)

        # Get the minimum index of the antenne (where cycle flag is valid)
        result = np.min(cycle_flags, axis=1)
        # When no antenna is valid, take the first one
        result[result >= mbg.sounder_file.antenna_count] = 0

        return result
