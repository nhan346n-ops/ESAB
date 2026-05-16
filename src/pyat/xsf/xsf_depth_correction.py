#! /usr/bin/env python3
# coding: utf-8

import datetime as dt
import logging
import os
import shutil
from types import EllipsisType
from typing import List, NamedTuple, Tuple

import numpy as np
import pandas as pd
import pytide.driver.tide_driver as td
import sonar_netcdf.sonar_groups as sg
from sonar_netcdf.utils import nc_merger as nc_m

import pyat.utils.argument_utils as arg_util
import pyat.utils.cut_file_utils as cut_util
import pyat.xsf.xsf_driver as xd

__logger = logging.getLogger("Tide/Draught correction")


class CorrectionArgs(NamedTuple):
    """
    Class representing all arguments for configuring the process
    """

    # Input XSF files
    i_paths: List[str]

    # Output XSF files
    o_paths: List[str]
    overwrite: bool = False
    # NONE, CORRECTION_FILE, RESET
    tide_correction: str = "NONE"
    # Tide file (Gauge observations / Prediction)
    tide_file: str | None = None

    # UNKNWON, PREDICTED, MEASURED, GNSS
    tide_type: str = "UNKNOWN"
    # UNKNWON, TIDE_GAUGE, GNSS, MODEL
    tide_source: str = "UNKNOWN"

    # NONE, CORRECTION_FILE, RESET
    draught_correction: str = "NONE"
    # Draught correction file
    draught_file: str | None = None

    cut_file: str | None = None
    geo_mask_file: str | None = None
    reverse_geo_mask: bool = False
    start_date: str | None = None
    end_date: str | None = None


def process(**kwargs) -> None:
    """
    Function accepting all arguments of the process as a dict. Possible arguments are listed in "CorrectionArgs" class
    """
    process_with_Args(CorrectionArgs(**kwargs))


def process_with_Args(args: CorrectionArgs) -> None:
    """
    Main function
    Use arguments to apply the corrections
    """
    timelines = _compute_timelines(args)

    for o_path, i_path in zip(args.o_paths, args.i_paths):
        if not os.path.exists(o_path) or args.overwrite:
            _process_correction(args, timelines, i_path, o_path)
        else:
            __logger.warning(f"{o_path} exists and cannot be overwritten")


def _process_correction(args: CorrectionArgs, timelines: List[nc_m.Timeline], i_path, o_path) -> None:
    """
    Prepare output file by copying XSF input file
    Browse all the timelines and make the corrections indicated in the arguments for each one.
    Update history and processing status
    """
    __logger.info(f"Processing {i_path}")
    if not os.path.exists(o_path) or not os.path.samefile(i_path, o_path):
        shutil.copy(i_path, o_path)

    with xd.open_xsf(o_path, mode="r+") as xsf_file:
        if timelines:
            for timeline in timelines:
                __logger.info(f"Working on timeline :  from {timeline.start} to {timeline.stop}")
                _process_correction_on_timeline(args, timeline, xsf_file)
        else:
            # Using a fake timeline covering all times
            _process_correction_on_timeline(args, None, xsf_file)

        __update_history(args, xsf_file)


def _process_correction_on_timeline(args: CorrectionArgs, timeline: nc_m.Timeline, xsf_file: xd.XsfDriver) -> None:
    """
    Perform corrections indicated in the arguments to the XSF file.
    Only values belonging to the timeline are affected
    """

    # Apply draught correction before tide correction to have coherent waterline
    if args.draught_correction == "CORRECTION_FILE" and args.draught_file is not None:
        _process_draught(xsf_file, args.draught_file, timeline)
    elif args.draught_correction == "RESET":
        _process_reset_draught(xsf_file, timeline)

    # Apply tide correction
    if args.tide_correction == "CORRECTION_FILE" and args.tide_file is not None:
        if args.tide_source == "GNSS":
            _process_gnss_tide(xsf_file, args.tide_file, timeline)
        else:
            _process_tide(xsf_file, args.tide_file, timeline)
    elif args.tide_source == "RESET":
        _process_reset_tide(xsf_file, timeline)


def _read_ttb_file(ttb_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse the TTB file"""
    tides = pd.read_csv(
        ttb_path,
        delimiter="\t",
        names=["correction_dates", "correction_value"],
        dtype={"correction_value": float},
    )

    correction_dates = pd.to_datetime(tides["correction_dates"], format="%d/%m/%Y %H:%M:%S.%f", utc=True)
    correction_dates = correction_dates.to_numpy(dtype=np.uint64)
    correction_values = tides["correction_value"].to_numpy()
    return correction_dates, correction_values


def _process_tide(xsf_file: xd.XsfDriver, tide_correction_path: str, timeline: nc_m.Timeline | None):
    """
    Perform a tide correction in the XSF file.
    Only values belonging to the timeline are affected
    """
    __logger.info(f"Processing tide correction with {tide_correction_path}")

    # Read correction file
    with td.open_tide(tide_correction_path) as tide_driver:
        tide_correction_dates = tide_driver.get_times()
        tide_correction_values = tide_driver.get_tides()

        # Compute tide_indicative layer
        tide_time = xsf_file[sg.TideGrp.TIME()][:]
        tide_indicative_interp = np.interp(
            tide_time, tide_correction_dates, tide_correction_values, left=0.0, right=0.0
        )
        tide_slice = _compute_slice_from_timeline(tide_time, timeline)
        xsf_file[sg.TideGrp.TIDE_INDICATIVE()][tide_slice] = tide_indicative_interp[tide_slice]

        # Compute waterline_to_chart_datum layer
        ping_time = xsf_file[xd.PING_TIME][:]
        waterline_interp = np.interp(ping_time, tide_correction_dates, tide_correction_values, left=0.0, right=0.0)
        waterline_slice = _compute_slice_from_timeline(ping_time, timeline)
        xsf_file[xd.WATERLINE_TO_CHART_DATUM][waterline_slice] = waterline_interp[waterline_slice]
        xsf_file[sg.TideGrp.get_group_path()].setncatts(tide_driver.get_metadata())


def _process_gnss_tide(xsf_file: xd.XsfDriver, tide_correction_path: str, timeline: nc_m.Timeline | None):
    __logger.info(f"Processing tide correction with gnss tide file {tide_correction_path}")

    # Read layers
    ping_time = xsf_file[xd.PING_TIME][:]
    tide_time = xsf_file[sg.TideGrp.TIME()][:]
    draught_time = xsf_file[sg.DynamicDraughtGrp.TIME()][:]
    dynamic_draught = xsf_file[sg.DynamicDraughtGrp.DELTA_DRAUGHT()][:]

    # Draught/Tide are not necessary aligned with swath. In that case values have to be interpolated.
    dynamic_draught_on_swath = np.interp(ping_time, draught_time, dynamic_draught)

    # Read correction file
    with td.open_tide(tide_correction_path) as tide_driver:
        tide_correction_dates = tide_driver.get_times()
        tide_correction_values = tide_driver.get_tides()

        tide_correction_on_swath = np.interp(
            ping_time, tide_correction_dates, tide_correction_values, left=0.0, right=0.0
        )

        # Compute new values
        waterline_on_swath = tide_correction_on_swath - dynamic_draught_on_swath
        ping_time_slice = _compute_slice_from_timeline(ping_time, timeline)
        xsf_file[xd.WATERLINE_TO_CHART_DATUM][ping_time_slice] = waterline_on_swath[ping_time_slice]

        tide_indicative = np.interp(tide_time, ping_time, tide_correction_on_swath)
        tide_indicative_slice = _compute_slice_from_timeline(tide_time, timeline)
        xsf_file[sg.TideGrp.TIDE_INDICATIVE()][tide_indicative_slice] = tide_indicative[tide_indicative_slice]
        xsf_file[sg.TideGrp.get_group_path()].setncatts(tide_driver.get_metadata())


def _process_reset_tide(xsf_file: xd.XsfDriver, timeline: nc_m.Timeline | None):
    """
    Undo a tide correction in the XSF file.
    Only values belonging to the timeline are affected
    """
    __logger.info(f"Reseting tide correction")

    tide_time = xsf_file[sg.TideGrp.TIME()][:]
    tide_slice = _compute_slice_from_timeline(tide_time, timeline)
    xsf_file[sg.TideGrp.TIDE_INDICATIVE()][tide_slice] = 0.0
    for key in xsf_file[sg.TideGrp.get_group_path()].ncattrs():
        xsf_file[sg.TideGrp.get_group_path()].delncattr(key)

    waterline_time = xsf_file[xd.PING_TIME][:]
    waterline_slice = _compute_slice_from_timeline(waterline_time, timeline)
    xsf_file[xd.WATERLINE_TO_CHART_DATUM][waterline_slice] = 0.0


def _process_draught(xsf_file: xd.XsfDriver, draught_correction_path: str, timeline: nc_m.Timeline | None):
    """
    Perform a draught correction in the XSF file.
    Only values belonging to the timeline are affected
    """
    __logger.info(f"Processing draught correction with {draught_correction_path}")

    # Read correction file
    draught_correction_dates, draught_correction_values = _read_ttb_file(draught_correction_path)

    # Interpolate draught
    delta_draught_time = xsf_file[sg.DynamicDraughtGrp.TIME()][:]
    delta_draught_interp = np.interp(delta_draught_time, draught_correction_dates, draught_correction_values)

    # Save previous values
    previous_delta_draught = xsf_file[sg.DynamicDraughtGrp.DELTA_DRAUGHT()][:]
    # Set new values to delta_draught layer
    delta_draught_slice = _compute_slice_from_timeline(delta_draught_time, timeline)
    xsf_file[sg.DynamicDraughtGrp.DELTA_DRAUGHT()][delta_draught_slice] = delta_draught_interp[delta_draught_slice]

    # Apply difference between previous and new values
    diff_delta_draught = previous_delta_draught - delta_draught_interp
    xsf_file[xd.PLATFORM_VERTICAL_OFFSET][delta_draught_slice] -= diff_delta_draught[delta_draught_slice]
    xsf_file[xd.TX_TRANSDUCER_DEPTH][delta_draught_slice] += diff_delta_draught[delta_draught_slice]


def _process_reset_draught(xsf_file: xd.XsfDriver, timeline: nc_m.Timeline | None):
    """
    Undo a draught correction in the XSF file.
    Only values belonging to the timeline are affected
    """
    __logger.info("Reseting draught correction")

    # Interpolate draught
    delta_draught_time = xsf_file[sg.DynamicDraughtGrp.TIME()][:]

    # Save previous values
    previous_delta_draught = xsf_file[sg.DynamicDraughtGrp.DELTA_DRAUGHT()][:]
    # Reset values
    delta_draught_slice = _compute_slice_from_timeline(delta_draught_time, timeline)
    xsf_file[sg.DynamicDraughtGrp.DELTA_DRAUGHT()][delta_draught_slice] = 0.0

    # Undo corrections
    xsf_file[xd.PLATFORM_VERTICAL_OFFSET][delta_draught_slice] -= previous_delta_draught[delta_draught_slice]
    xsf_file[xd.TX_TRANSDUCER_DEPTH][delta_draught_slice] += previous_delta_draught[delta_draught_slice]


def __update_history(args: CorrectionArgs, xsf_file: xd.XsfDriver) -> None:
    """
    Complete the history of the XSF
    """
    if args.tide_correction == "CORRECTION_FILE" and args.tide_file is not None:
        xsf_file.append_history_line(
            f"Tide correction (type: {args.tide_type}; source: {args.tide_source}; ref: {os.path.basename(args.tide_file)}) with PyAT/Ifremer"
        )
        xsf_file.update_processing_status({xd.ATT_PROCESSING_STATUS_TIDE_CORRECTION: xd.ATT_PROCESSING_STATUS_FLAG_ON})

    elif args.tide_correction == "RESET":
        xsf_file.append_history_line(f"Reset tide correction with PyAT/Ifremer")
        xsf_file.update_processing_status({xd.ATT_PROCESSING_STATUS_TIDE_CORRECTION: xd.ATT_PROCESSING_STATUS_FLAG_OFF})

    if args.draught_correction == "CORRECTION_FILE" and args.draught_file is not None:
        xsf_file.append_history_line(
            f"Draught correction (ref: {os.path.basename(args.draught_file)}) with PyAT/Ifremer"
        )
        xsf_file.update_processing_status(
            {xd.ATT_PROCESSING_STATUS_DRAUGHT_CORRECTION: xd.ATT_PROCESSING_STATUS_FLAG_ON}
        )

    elif args.draught_correction == "RESET":
        xsf_file.append_history_line(f"Reset draught correction with PyAT/Ifremer")
        xsf_file.update_processing_status(
            {xd.ATT_PROCESSING_STATUS_DRAUGHT_CORRECTION: xd.ATT_PROCESSING_STATUS_FLAG_OFF}
        )


def _compute_timelines(args: CorrectionArgs) -> List[nc_m.Timeline]:
    """
    Computing timelines from the specified arguments
    """

    start_date = arg_util.parse_datetime(args.start_date)
    end_date = arg_util.parse_datetime(args.end_date)

    computed_timelines: List[nc_m.Timeline] = []
    if args.cut_file is not None:
        __logger.info("Using cut_file argument to determine the cutting time intervals")
        computed_timelines = cut_util.parse_cut_file(args.cut_file, __logger)
        if len(computed_timelines) == 0:
            __logger.info(f"No cut line found in cut file. Merge abort")

    elif args.geo_mask_file is not None:
        __logger.info(f"Compute cut lines from geographic mask (reverse = {args.reverse_geo_mask}).")
        computed_timelines = cut_util.create_cut_lines_from_files(
            i_paths=args.i_paths,
            o_paths=args.i_paths,
            i_geo_mask_path=args.geo_mask_file,
            reverse_geo_mask=args.reverse_geo_mask,
        )
        if len(computed_timelines) == 0:
            __logger.info("Geographic mask does not cut input files.")

    elif start_date is not None and end_date is not None:
        __logger.info(f"Apply custom time interval : from {start_date} to {end_date}.")
        computed_timelines.append(nc_m.Timeline("Single", start_date, end_date))

    if len(computed_timelines) > 1:
        __logger.info(f"Apply {len(computed_timelines)} cut lines : ")
        for timeline in computed_timelines:
            __logger.info(f"{timeline.name} :  from {timeline.start} to {timeline.stop}")
    else:
        __logger.info("No cutting time interval specified.")

    return computed_timelines


def _compute_slice_from_timeline(
    time_variable: np.ndarray, timeline: nc_m.Timeline | None
) -> np.ndarray | EllipsisType:
    """
    Returns the slice to be applied to the layer to respect the timeline
    """
    if timeline is None:
        return Ellipsis  # All indexes

    from_date = int(timeline.start.replace(tzinfo=dt.timezone.utc).timestamp() * 1e9)
    to_date = int(timeline.stop.replace(tzinfo=dt.timezone.utc).timestamp() * 1e9)

    indexes = np.argwhere((time_variable >= from_date) & (time_variable <= to_date)).ravel()

    if len(indexes) > 0:
        __logger.debug(f"Slicing on [{indexes[0]}:{indexes[-1]}]")

    return indexes


if __name__ == "__main__":
    process(
        i_paths=[r"E:\temp\tide\0132_20120607_070030_ShipName.xsf.nc"],
        o_paths=[r"E:\temp\tide\0132_20120607_070030_ShipName_tide-python.xsf.nc"],
        overwrite=True,
        tide_type="PREDICTED",
        tide_source="CORRECTION_FILE",
        tide_ttb=r"E:\temp\tide\python_prediction.ttb",
    )
