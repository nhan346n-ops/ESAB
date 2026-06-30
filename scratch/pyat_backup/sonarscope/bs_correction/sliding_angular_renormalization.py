import os
import shutil
import tempfile
from typing import List, Optional, Union

import dask
import numpy as np
import xarray as xr
from pygws.service.progress_monitor import ProgressMonitor
from scipy.stats import binned_statistic

from ...utils import numpy_utils
from ...utils.multiple_entry_dict import MultipleEntryDict
from ...utils.netcdf_utils import DEFAULT_COMPRESSION_LEVEL
from ...xsf import xsf_driver
from ...xsf.xsf_driver import XsfDriver
from ..common.configuration import IntegrationMethod, default_config
from ..model.constants import VariableKeys as Key
from ..model.signal.ping_detection_signal import PingDetectionSignal
from ..model.signal.ping_signal import PingSignal
from ..model.sonar_factories import ModeComputerFactory
from ..model.sounder_lib import SounderType
from ..model.sounder_mode.sounder_modes_computer import (
    get_invalid_key_indices,
    get_valid_key_indices,
)
from .bs_computer import BSComputer
from .dtm_angles_computer import DtmAnglesComputer
from .file_data import FileDataStore
from .global_data import GlobalDataModel
from .mean_bs_model import (
    BackscatterCurve,
    BackscatterCurveByIncidenceByPing,
    BackscatterCurveByTransmissionByPing,
    SlidingMeanBSModel,
)
from .seafloor_bs_angular_model import SlidingModel


def merge_backscatter_curve(input_netcdf_files: List[str], output_netcdf_file: str):
    encodings = {}
    input_ds = []
    try:
        with dask.config.set(
            {
                "array.slicing.split_large_chunks": True,
            }
        ):
            dataset_by_ping = xr.Dataset()
            for f in input_netcdf_files:
                ds = xr.open_dataset(
                    f,
                    chunks={BackscatterCurve.PING_TIME: 1000},
                )
                # retrieve compression parameters from input datasets
                for key, var in ds.data_vars.items():
                    if key not in encodings:
                        encodings[key] = {}
                    encodings[key]["zlib"] = var.encoding["zlib"]
                    encodings[key]["complevel"] = DEFAULT_COMPRESSION_LEVEL
                    if "chunksizes" in encodings[key]:
                        encodings[key]["chunksizes"] = max(encodings[key]["chunksizes"], var.encoding["chunksizes"])
                    else:
                        encodings[key]["chunksizes"] = var.encoding["chunksizes"]
                input_ds.append(ds)
                dataset_by_ping = xr.merge([dataset_by_ping, ds])
            delayed_ds = dataset_by_ping.to_netcdf(
                output_netcdf_file, encoding=encodings, engine="netcdf4", compute=False
            )
            delayed_ds.compute()
    finally:
        for d in input_ds:
            d.close()


def interpolate_bs_by_incidence(
    curve_by_incidence_by_ping: BackscatterCurveByIncidenceByPing, ping_time: np.ndarray, incidence_angles: np.ndarray
) -> np.ndarray:
    bs_incidence = np.full_like(incidence_angles, fill_value=np.nan)
    curve_incidence_angles = curve_by_incidence_by_ping.ds[BackscatterCurve.ANGLE].compute()
    curve_incidence_values = curve_by_incidence_by_ping.ds[BackscatterCurve.MEAN_BS].sel(ping_time=ping_time).compute()
    curve_incidence_mask = np.isnan(curve_incidence_values).compute()

    for i in range(len(ping_time)):
        # retrieve corresponding bs by incidence angle in bs model (extrapolate with left and right values)
        if curve_incidence_mask[i].all():
            continue
        bs_incidence[i] = np.interp(
            incidence_angles[i],
            curve_incidence_angles[~curve_incidence_mask[i]],
            curve_incidence_values[i][~curve_incidence_mask[i]],
        )
    return bs_incidence
import pandas as pd

def safe_pandas_rolling(data_or_da, window, center=True, method="sum"):
    """
    Perform a pandas time-based rolling calculation safely, even if the index is
    non-monotonic or contains duplicate values.
    """
    if hasattr(data_or_da, "to_pandas"):
        data = data_or_da.to_pandas()
    else:
        data = data_or_da

    orig_idx = data.index
    orig_order = np.arange(len(data))

    is_series = isinstance(data, pd.Series)
    if is_series:
        df = pd.DataFrame({"value": data, "orig_order": orig_order})
    else:
        df = data.copy()
        df["orig_order"] = orig_order

    df_sorted = df.sort_index()
    cols_to_roll = [c for c in df_sorted.columns if c != "orig_order"]

    rolled_values = df_sorted[cols_to_roll].rolling(window, center=center)
    if method == "sum":
        rolled_df = rolled_values.sum()
    elif method == "median":
        rolled_df = rolled_values.median()
    else:
        raise ValueError(f"Unknown rolling method {method}")

    rolled_df["orig_order"] = df_sorted["orig_order"]
    rolled_df_restored = rolled_df.sort_values("orig_order")
    rolled_df_restored.index = orig_idx

    if is_series:
        return rolled_df_restored["value"]
    else:
        return rolled_df_restored.drop(columns=["orig_order"])


class SlidingMeanBSComputer:
    def __init__(
        self,
        sounder_type=None,
        sliding_window_min: int = 10,
    ):
        self.sounder_type = sounder_type
        self.sliding_window_min = sliding_window_min
        self.short_sliding_window_min = max(sliding_window_min // 10, 1)
        numpy_utils.disable_warning()

    def compute(
        self, input_files: List[str], global_data: GlobalDataModel, dtm_angles_computer: DtmAnglesComputer = None
    ):
        """
        Compute list of mode for input file and compute mean values for backscatter detection values.
        Statistics are computed in two steps : the first one per mode and per file, then mean values are concatenated along files.
        """
        global_monitor = default_config.monitor.split(1)
        global_monitor.begin_task("Compute backscatter sliding mean", 4)

        default_config.logger.info(f"Compute list of available mode")
        mode_computer = ModeComputerFactory.create_mode_computer(self.sounder_type)
        key_dict, mode_ids_dict = mode_computer.compute(input_files)

        # set config
        default_config.setup(self.sounder_type)

        valid_indices = get_valid_key_indices(key_dict=key_dict)
        valid_modes_count = len(valid_indices)
        invalid_indices = get_invalid_key_indices(key_dict=key_dict)
        invalid_modes_count = len(invalid_indices)
        default_config.logger.info(f"Found {valid_modes_count} valid modes, {invalid_modes_count} invalid mode ")
        # create a dictionary with mode as a key and a list of xarray data set perfile containing statistics
        curves_by_incidence_per_mode_and_file = MultipleEntryDict()  # curve dictionary per mode
        curves_by_transmission_per_mode_and_file = MultipleEntryDict()  # curve dictionary per mode

        curve_by_incidence_per_mode = {}  #
        curve_by_transmission_per_mode = {}  #

        incidence_angle_range, incidence_bin_width, incidence_bin_count, incidence_bin_centers = (
            default_config.incidence_angles.angle_range,
            default_config.incidence_angles.bin_width,
            default_config.incidence_angles.bin_count,
            default_config.incidence_angles.bin_centers,
        )
        transmission_angle_range, transmission_bin_width, transmission_bin_count, transmission_bin_centers = (
            default_config.transmission_angles.angle_range,
            default_config.transmission_angles.bin_width,
            default_config.transmission_angles.bin_count,
            default_config.transmission_angles.bin_centers,
        )

        rx_antenna_count = None
        rx_antenna_index = None

        #
        # START PROCESS PER FILE BY INCIDENCE
        #
        monitor = global_monitor.split(1)
        monitor.begin_task("Compute mode and stats by incidence angle", len(input_files))
        for file in input_files:
            file = str(file)  # need to convert to str which is the id used in dictionary
            default_config.logger.info(f"------------")
            default_config.logger.info(f"Processing file {file}")
            modes_indexes = mode_ids_dict[file]  # index of active modes in files
            unique_modes = np.intersect1d(modes_indexes, valid_indices)  # retrieve list of valid modes defined in file
            # get beam angles values
            # get backscatter values
            with xsf_driver.open_xsf(file_path=file, mode="r") as xsf:
                # check minimal xsf version
                default_config.check_version(xsf_dataset=xsf)

                default_config.logger.info(f"Build ping detection model")
                ping_model = PingSignal(xsf_dataset=xsf)
                ping_model.read(Key.PING_TIME)
                ping_time = ping_model.xr_dataset[Key.PING_TIME].values
                ping_count = ping_time.shape[0]

                ping_detection_model = PingDetectionSignal(xsf_dataset=xsf)
                ping_detection_model.read([Key.BATHYMETRY_STATUS])

                status = ping_detection_model.xr_dataset[Key.BATHYMETRY_STATUS].values
                status_mask = status == 0

                default_config.logger.info(f"Compute backscatter values")
                bs_value, incidence_angles = BSComputer.compute_bs(
                    ping_dataset=ping_model,
                    ping_detection_dataset=ping_detection_model,
                    dtm_angles_computer=dtm_angles_computer,
                )

                # store filedata in global data for future use
                global_data.file_data[file] = FileDataStore(
                    file=file, bs_value=bs_value, incidence_angle=incidence_angles
                )

                # remove non valid data
                status_mask[~np.isfinite(bs_value)] = False
                status_mask[~np.isfinite(incidence_angles)] = False

                for current_mode_idx in unique_modes:
                    mode = [k for k, v in key_dict.items() if v == current_mode_idx][0]
                    # ignore invalid mode
                    if not mode.is_valid():
                        continue
                    default_config.logger.info(f"Processing mode {mode}")

                    # resample data in order to be able to work mode per mode
                    # keep only data matching the current mode
                    mode_mask = modes_indexes == current_mode_idx

                    # compute stats by incidence angle
                    # remove data not valid
                    detection_mask = status_mask.copy()
                    # remove data not matching mode
                    detection_mask[~mode_mask] = False

                    incidence_angles_masked = np.ma.MaskedArray(incidence_angles, ~detection_mask)
                    bs_masked = np.ma.MaskedArray(bs_value, ~detection_mask)
                    if default_config.integration_method is IntegrationMethod.MEAN:
                        bs_masked = default_config.db_to_linear(bs_masked)
                        mean_method = "mean"
                    else:
                        mean_method = "median"
                    mean_values_by_incidence = np.full(
                        shape=(ping_count, incidence_bin_count),
                        fill_value=np.nan,
                    )
                    value_counts_by_incidence = np.full(
                        shape=(ping_count, incidence_bin_count),
                        fill_value=np.nan,
                    )

                    for i in range(len(ping_time)):
                        if ~mode_mask[i] or ~detection_mask[i].any():
                            continue
                        angles_compressed = incidence_angles_masked[i].compressed()
                        bs_compressed = bs_masked[i].compressed()
                        stat_count, _, _ = binned_statistic(
                            x=angles_compressed,
                            values=bs_compressed,
                            statistic="count",
                            bins=incidence_bin_count,
                            range=incidence_angle_range,
                        )
                        stat_mean, _, _ = binned_statistic(
                            x=angles_compressed,
                            values=bs_compressed,
                            statistic=mean_method,
                            bins=incidence_bin_count,
                            range=incidence_angle_range,
                        )
                        mean_values_by_incidence[i, :] = stat_mean
                        value_counts_by_incidence[i, :] = stat_count

                    if default_config.integration_method is IntegrationMethod.MEAN:
                        mean_values_by_incidence = default_config.linear_to_db(mean_values_by_incidence)

                    # create curve by incidence
                    curve_by_incidence = BackscatterCurveByIncidenceByPing.build(
                        mean_values=mean_values_by_incidence,
                        count=value_counts_by_incidence,
                        bin_centers=incidence_bin_centers,
                        ping_time=ping_time[:],
                        mode=np.where(mode_mask, current_mode_idx, np.nan),
                        origin=file,
                    )
                    tmp_curvefile = tempfile.mktemp()
                    curve_by_incidence.to_netcdf(tmp_curvefile)
                    # recompute mean of each file
                    # retain mean and per mode
                    curves_by_incidence_per_mode_and_file.add(key=mode, obj=tmp_curvefile)
            monitor.worked(1)
        monitor.done()
        #
        # END PROCESS PER FILE PER INCIDENCE
        #

        #
        # Compute  synthesis per incidence
        #
        monitor = global_monitor.split(1)
        monitor.begin_task("Merge stats by incidence angle from all files", valid_modes_count + 1)

        # for each mode, recompute all means
        for mode in curves_by_incidence_per_mode_and_file.keys():
            if not mode.is_valid():
                continue

            default_config.logger.info(f"Processing mode {mode}")
            ds_incidence_files = curves_by_incidence_per_mode_and_file.get(mode)
            merged_incidence_file = tempfile.mktemp()
            merge_backscatter_curve(ds_incidence_files, merged_incidence_file)

            with xr.open_dataset(merged_incidence_file, chunks={BackscatterCurve.PING_TIME: 1000}) as ds_incidence:
                # get mean values for this mode
                mean_values_incidence = ds_incidence[BackscatterCurve.MEAN_BS]
                count_incidence = ds_incidence[BackscatterCurve.VALUE_COUNT]
                rolled_count = safe_pandas_rolling(
                    count_incidence, f"{self.sliding_window_min}min", center=True, method="sum"
                )

                if default_config.integration_method is IntegrationMethod.MEAN:
                    mean_values_incidence_linear = default_config.db_to_linear(mean_values_incidence)
                    weighted_mean_values = mean_values_incidence_linear * count_incidence
                    rolled_sum = safe_pandas_rolling(
                        weighted_mean_values, f"{self.sliding_window_min}min", center=True, method="sum"
                    )
                    rolled_mean_values = default_config.linear_to_db(rolled_sum / rolled_count)
                else:  # default_config.integration_method is IntegrationMethod.MEDIAN
                    rolled_mean_values = safe_pandas_rolling(
                        mean_values_incidence, f"{self.sliding_window_min}min", center=True, method="median"
                    )

                # clear empty pings
                mask_array = mean_values_incidence.isnull().values
                rolled_mean_values[mask_array] = np.nan
                rolled_count[mask_array] = np.nan  # Use Nan to be able to merge DataArray

                curve_by_incidence = BackscatterCurveByIncidenceByPing.build(
                    mean_values=rolled_mean_values[:],
                    count=rolled_count[:],
                    bin_centers=incidence_bin_centers,
                    ping_time=ds_incidence[BackscatterCurve.PING_TIME].values,
                    mode=ds_incidence[BackscatterCurve.MODE].values,
                    origin=None,
                )
                tmp_curvefile = tempfile.mktemp()
                curve_by_incidence.to_netcdf(tmp_curvefile)
                curve_by_incidence_per_mode[mode] = tmp_curvefile
            # clean temporary files
            os.remove(merged_incidence_file)
            for f in ds_incidence_files:
                os.remove(f)
            monitor.worked(1)

        global_data.incidence_curve_file = tempfile.mktemp()

        default_config.logger.info(f"Merging backscatter incidence curves")
        merge_backscatter_curve(curve_by_incidence_per_mode.values(), global_data.incidence_curve_file)

        # clean temporary files
        for f in curve_by_incidence_per_mode.values():
            os.remove(f)
        monitor.done()

        #
        # START PROCESS PER FILE BY TRANSMISSION
        #
        monitor = global_monitor.split(1)
        monitor.begin_task("Compute mode and stats by transmission angle", len(input_files))
        with xr.open_dataset(
            global_data.incidence_curve_file, chunks={BackscatterCurve.PING_TIME: 1000}
        ) as dataset_by_incidence_by_ping:
            curve_by_incidence_by_ping = BackscatterCurveByIncidenceByPing(xr_dataset=dataset_by_incidence_by_ping)
            for file in input_files:
                file = str(file)  # need to convert to str which is the id used in dictionary
                default_config.logger.info(f"------------")
                default_config.logger.info(f"Processing file {file}")
                modes_indexes = mode_ids_dict[file]  # index of active modes in files
                unique_modes = np.intersect1d(
                    modes_indexes, valid_indices
                )  # retrieve list of valid modes defined in file
                # get beam angles values
                # get backscatter values
                with xsf_driver.open_xsf(file_path=file, mode="r") as xsf:
                    # check minimal xsf version
                    default_config.check_version(xsf_dataset=xsf)

                    default_config.logger.info(f"Build ping detection model")
                    ping_model = PingSignal(xsf_dataset=xsf)
                    ping_model.read(Key.PING_TIME)
                    ping_time = ping_model.xr_dataset[Key.PING_TIME].values
                    ping_count = ping_time.shape[0]

                    ping_detection_model = PingDetectionSignal(xsf_dataset=xsf)
                    ping_detection_model.read(
                        [
                            Key.DETECTION_TX_BEAM_INDEX,
                            Key.DETECTION_RX_TRANSDUCER_INDEX,
                            Key.BATHYMETRY_STATUS,
                            Key.DETECTION_BEAM_POINTING_ANGLE_REF_PLATFORM,
                        ]
                    )

                    status = ping_detection_model.xr_dataset[Key.BATHYMETRY_STATUS].values
                    status_mask = status == 0
                    detection_tx_beam = ping_detection_model.xr_dataset[Key.DETECTION_TX_BEAM_INDEX].values
                    detection_rx_transducer = ping_detection_model.xr_dataset[Key.DETECTION_RX_TRANSDUCER_INDEX].values
                    detection_beam_pointing_angle_ref_platform = ping_detection_model.xr_dataset[
                        Key.DETECTION_BEAM_POINTING_ANGLE_REF_PLATFORM
                    ].values

                    file_antenna_index = xsf.get_rx_transducers()
                    file_antenna_count = len(file_antenna_index) if file_antenna_index is not None else 0
                    if rx_antenna_count is None:
                        rx_antenna_count = file_antenna_count
                        rx_antenna_index = file_antenna_index
                    elif rx_antenna_count != file_antenna_count:
                        default_config.logger.warning(f"The rx antenna count does not match in this file.")

                    default_config.logger.info(f"Compute backscatter values")
                    # reuse data computed in previous pass
                    bs_value = global_data.file_data[file].bs_value
                    # bs_linear = default_config.db_to_linear(bs_value)

                    incidence_angles = global_data.file_data[file].incidence_angle
                    # interpolate bs values by incidence
                    bs_incidence = interpolate_bs_by_incidence(curve_by_incidence_by_ping, ping_time, incidence_angles)
                    bs_diff = bs_value - bs_incidence

                    if default_config.integration_method is IntegrationMethod.MEAN:
                        bs_value = default_config.db_to_linear(bs_value)
                        bs_diff = default_config.db_to_linear(bs_diff)
                        mean_method = "mean"
                    else:
                        mean_method = "median"

                    transmission_angles = detection_beam_pointing_angle_ref_platform

                    # remove non valid data
                    status_mask[~np.isfinite(bs_value)] = False
                    status_mask[~np.isfinite(incidence_angles)] = False
                    status_mask[~np.isfinite(transmission_angles)] = False

                    for current_mode_idx in unique_modes:
                        mode = [k for k, v in key_dict.items() if v == current_mode_idx][0]
                        # ignore invalid mode
                        if not mode.is_valid():
                            continue
                        default_config.logger.info(f"Processing mode {mode} with {mode.get_tx_beam_count()} tx beams")

                        # resample data in order to be able to work mode per mode
                        # keep only data matching the current mode
                        mode_mask = modes_indexes == current_mode_idx

                        mean_values_per_tx = np.full(
                            shape=(rx_antenna_count, mode.get_tx_beam_count(), ping_count, transmission_bin_count),
                            fill_value=np.nan,
                        )
                        mean_diff_values_per_tx = np.full(
                            shape=(rx_antenna_count, mode.get_tx_beam_count(), ping_count, transmission_bin_count),
                            fill_value=np.nan,
                        )
                        value_counts_per_tx = np.full(
                            shape=(rx_antenna_count, mode.get_tx_beam_count(), ping_count, transmission_bin_count),
                            fill_value=np.nan,
                        )

                        for rx_antenna in range(rx_antenna_count):
                            rx_mask = detection_rx_transducer == rx_antenna_index[rx_antenna]
                            if not np.any(rx_mask):
                                continue
                            for tx_beam in range(mode.get_tx_beam_count()):
                                detection_mask = status_mask.copy()
                                # filter to keep data for this tx_beam
                                detection_mask[detection_tx_beam != tx_beam] = False
                                # remove data not matching antenna
                                detection_mask[~rx_mask] = False
                                # remove data not matching mode
                                detection_mask[~mode_mask] = False

                                transmission_angles_masked = np.ma.MaskedArray(transmission_angles, ~detection_mask)
                                bs_value_masked = np.ma.MaskedArray(bs_value, ~detection_mask)
                                bs_diff_masked = np.ma.MaskedArray(bs_diff, ~detection_mask)

                                for i in range(ping_count):
                                    if ~mode_mask[i] or ~detection_mask[i].any():
                                        continue
                                    # Compute stats
                                    tx_angles_compressed = transmission_angles_masked[i].compressed()
                                    bs_value_compressed = bs_value_masked[i].compressed()
                                    bs_diff_compressed = bs_diff_masked[i].compressed()
                                    stat_count, _, _ = binned_statistic(
                                        x=tx_angles_compressed,
                                        values=bs_diff_compressed,
                                        statistic="count",
                                        bins=transmission_bin_count,
                                        range=transmission_angle_range,
                                    )
                                    stat_count = stat_count.astype(int)

                                    (stat_mean_value, stat_mean_diff), _, _ = binned_statistic(
                                        x=tx_angles_compressed,
                                        values=(bs_value_compressed, bs_diff_compressed),
                                        statistic=mean_method,
                                        bins=transmission_bin_count,
                                        range=transmission_angle_range,
                                    )
                                    mean_values_per_tx[rx_antenna][tx_beam][i] = stat_mean_value
                                    mean_diff_values_per_tx[rx_antenna][tx_beam][i] = stat_mean_diff
                                    value_counts_per_tx[rx_antenna][tx_beam][i] = stat_count

                        if default_config.integration_method is IntegrationMethod.MEAN:
                            mean_values_per_tx = default_config.linear_to_db(mean_values_per_tx)
                            mean_diff_values_per_tx = default_config.linear_to_db(mean_diff_values_per_tx)

                        # create curve for all sectors
                        curve_by_transmission = BackscatterCurveByTransmissionByPing.build(
                            rx_antenna_count=rx_antenna_count,
                            tx_beam_count=mode.get_tx_beam_count(),
                            mean_values=mean_values_per_tx,
                            mean_residual_values=mean_diff_values_per_tx,
                            count=value_counts_per_tx,
                            bin_centers=transmission_bin_centers,
                            ping_time=ping_time,
                            origin=file,
                        )

                        tmp_curvefile = tempfile.mktemp()
                        curve_by_transmission.to_netcdf(tmp_curvefile)
                        # recompute mean of each file
                        # retain mean and per mode
                        curves_by_transmission_per_mode_and_file.add(key=mode, obj=tmp_curvefile)
                monitor.worked(1)
            monitor.done()
        #
        # END PROCESS PER FILE BY TRANSMISSION
        #

        #
        # Compute transmission synthesis
        #
        monitor = global_monitor.split(1)
        monitor.begin_task("Merge stats by transmission angle from all files", valid_modes_count + 1)
        # for each mode, recompute all means
        for mode in curves_by_transmission_per_mode_and_file.keys():
            if not mode.is_valid():
                continue

            default_config.logger.info(f"Processing mode {mode}")
            ds_transmission_files = curves_by_transmission_per_mode_and_file.get(mode)
            merged_transmission_file = tempfile.mktemp()
            merge_backscatter_curve(ds_transmission_files, merged_transmission_file)

            with xr.open_dataset(
                merged_transmission_file, chunks={BackscatterCurve.PING_TIME: 1000}
            ) as ds_transmission:
                # get mean values per file for this mode
                mean_transmission = ds_transmission[BackscatterCurve.MEAN_BS]
                mean_residual_transmission = ds_transmission[BackscatterCurve.MEAN_RESIDUAL_BS]
                count_transmission = ds_transmission[BackscatterCurve.VALUE_COUNT]
                rolled_count = np.empty_like(count_transmission)
                rolled_mean_residual = np.empty_like(mean_residual_transmission)
                rolled_mean = np.empty_like(mean_transmission)

                if default_config.integration_method is IntegrationMethod.MEAN:
                    # convert to linear scale
                    mean_residual_transmission_linear = default_config.db_to_linear(mean_residual_transmission)
                    weighted_mean_residual = mean_residual_transmission_linear * count_transmission

                for i in range(rolled_count.shape[0]):
                    for j in range(rolled_count.shape[1]):
                        rolled_count[i][j] = safe_pandas_rolling(
                            count_transmission[i][j],
                            f"{self.sliding_window_min}min",
                            center=True,
                            method="sum",
                        )
                        if default_config.integration_method is IntegrationMethod.MEAN:
                            rolled_mean_residual_sum = safe_pandas_rolling(
                                weighted_mean_residual[i][j],
                                f"{self.sliding_window_min}min",
                                center=True,
                                method="sum",
                            )
                            rolled_mean_residual[i][j] = default_config.linear_to_db(
                                rolled_mean_residual_sum / rolled_count[i][j]
                            )
                        else:  # default_config.integration_method is IntegrationMethod.MEDIAN
                            rolled_mean_residual[i][j] = safe_pandas_rolling(
                                mean_residual_transmission[i][j],
                                f"{self.sliding_window_min}min",
                                center=True,
                                method="median",
                            )
                        # short term stats always use median
                        rolled_mean[i][j] = safe_pandas_rolling(
                            mean_transmission[i][j],
                            f"{self.short_sliding_window_min}min",
                            center=True,
                            method="median",
                        )

                # clear empty pings
                mask_array = mean_residual_transmission.isnull().values
                rolled_mean_residual[mask_array] = np.nan
                rolled_count[mask_array] = np.nan
                mask_array = mean_transmission.isnull().values
                rolled_mean[mask_array] = np.nan

                curve_by_transmission = BackscatterCurveByTransmissionByPing.build(
                    rx_antenna_count=rolled_mean_residual.shape[0],
                    tx_beam_count=rolled_mean_residual.shape[1],
                    mean_values=rolled_mean[:],
                    mean_residual_values=rolled_mean_residual[:],
                    count=rolled_count[:],
                    bin_centers=transmission_bin_centers,
                    ping_time=ds_transmission[BackscatterCurve.PING_TIME].values,
                    origin=None,
                )
                tmp_curvefile = tempfile.mktemp()
                curve_by_transmission.to_netcdf(tmp_curvefile)
                curve_by_transmission_per_mode[mode] = tmp_curvefile
                monitor.worked(1)

            # clean temporary files
            os.remove(merged_transmission_file)
            for f in ds_transmission_files:
                os.remove(f)

        global_data.transmission_curve_file = tempfile.mktemp()
        default_config.logger.info(f"Merging backscatter transmission curves")
        merge_backscatter_curve(curve_by_transmission_per_mode.values(), global_data.transmission_curve_file)

        # clean temporary files
        for f in curve_by_transmission_per_mode.values():
            os.remove(f)

        #
        # WRITE BSAR MODEL FILE
        #
        monitor = global_monitor.split(1)
        monitor.begin_task("Write sliding bsar file", 1)
        with (
            xr.open_dataset(global_data.incidence_curve_file, chunks={}) as dataset_by_incidence_by_ping,
            xr.open_dataset(global_data.transmission_curve_file, chunks={}) as dataset_by_transmission_by_ping,
        ):
            curve_by_incidence_by_ping = BackscatterCurveByIncidenceByPing(xr_dataset=dataset_by_incidence_by_ping)
            curve_by_transmission_by_ping = BackscatterCurveByTransmissionByPing(
                xr_dataset=dataset_by_transmission_by_ping
            )
            # set temp file if output bsar file is not provided
            if global_data.model_file is None:
                global_data.model_file = tempfile.mktemp(suffix=".bsar.nc")

            model = SlidingMeanBSModel(
                sounder_type=self.sounder_type,
                model_curves=(curve_by_incidence_by_ping, curve_by_transmission_by_ping),
            )
            model.save_to_netcdf(output_file=global_data.model_file, overwrite=True)

        # clean temporary files
        os.remove(global_data.incidence_curve_file)
        os.remove(global_data.transmission_curve_file)
        monitor.done()
        global_monitor.done()


class SlidingAngleNormalizer:
    """
    Process to normalize backscatter given a mean angular bs model(bs stats and a reference value)
    """

    def __init__(self, sounder_type: str, avg_model: SlidingModel, global_data: GlobalDataModel):
        self.sounder_type = sounder_type
        self.mode_computer = ModeComputerFactory.create_mode_computer(sounder_type)
        self.avg_model = avg_model
        self.global_data = global_data

    def apply_on_file(
        self,
        input_filepath: Union[str, XsfDriver],
        output_file: Union[str, XsfDriver],
        dtm_angles_computer: DtmAnglesComputer = None,
    ) -> np.ndarray:
        """Apply an averaged compensation on a single file"""
        need_to_close = False
        xsf = output_file
        try:
            if not isinstance(output_file, XsfDriver):
                xsf = xsf_driver.XsfDriver(file_path=output_file)
                need_to_close = True

            xsf.open(mode="r+")

            # create a model for data storage
            default_config.logger.info(f"Build ping detection model")
            ping_model = PingSignal(xsf_dataset=xsf)
            ping_model.read([Key.PING_TIME])
            ping_detection_model = PingDetectionSignal(xsf_dataset=xsf)
            ping_detection_model.read(
                [
                    Key.DETECTION_TX_BEAM_INDEX,
                    Key.DETECTION_RX_TRANSDUCER_INDEX,
                    Key.DETECTION_BEAM_POINTING_ANGLE_REF_PLATFORM,
                ]
            )

            ping_time = ping_model.xr_dataset[Key.PING_TIME].values
            detection_tx_beam = ping_detection_model.xr_dataset[Key.DETECTION_TX_BEAM_INDEX].values
            detection_rx_transducer = ping_detection_model.xr_dataset[Key.DETECTION_RX_TRANSDUCER_INDEX].values
            detection_beam_pointing_angle_ref_platform = ping_detection_model.xr_dataset[
                Key.DETECTION_BEAM_POINTING_ANGLE_REF_PLATFORM
            ].values

            rx_antenna_index = xsf.get_rx_transducers()

            # compute backscatter
            if self.global_data is not None and input_filepath in self.global_data.file_data.keys():
                bs_value = self.global_data.file_data[input_filepath].bs_value
                incidence_angles = self.global_data.file_data[input_filepath].incidence_angle
            else:
                bs_value, incidence_angles = BSComputer.compute_bs(
                    ping_dataset=ping_model,
                    ping_detection_dataset=ping_detection_model,
                    dtm_angles_computer=dtm_angles_computer,
                )

            transmission_angles = detection_beam_pointing_angle_ref_platform

            # initialize output dataset
            bs_corrected = np.full_like(bs_value, fill_value=np.nan)

            default_config.logger.info(f"Apply normalisation")

            lut_incidence_table, lut_incidence_angles = self.avg_model.get_avg_incidence_lut(ping_time)
            lut_transmission_table, lut_transmission_angles = self.avg_model.get_avg_residual_transmission_lut(
                ping_time
            )

            for i in range(len(ping_time)):
                for rx_antenna in range(rx_antenna_index.shape[0]):
                    rx_mask = detection_rx_transducer[i] == rx_antenna_index[rx_antenna]
                    if not np.any(rx_mask):
                        continue
                    for tx_beam in range(lut_transmission_table.shape[1]):
                        # need to take into account for txsectors
                        detection_mask = detection_tx_beam[i] == tx_beam
                        # remove data not matching rx_antenna
                        detection_mask[~rx_mask] = False
                        if not detection_mask.any():
                            continue
                        # need to interpol correction by angles
                        partial_incidence_angles = incidence_angles[i, detection_mask]
                        partial_transmission_angles = transmission_angles[i, detection_mask]
                        partial_bs_values = bs_value[i, detection_mask]

                        # need to remove nan values from lut
                        lut_incidence_mask = ~np.isnan(lut_incidence_table[i])
                        lut_transmission_mask = ~np.isnan(lut_transmission_table[rx_antenna][tx_beam][i])
                        partial_lut_incidence_angles = lut_incidence_angles[lut_incidence_mask]
                        partial_lut_incidence_values = lut_incidence_table[i, lut_incidence_mask]
                        partial_lut_transmission_angles = lut_transmission_angles[lut_transmission_mask]
                        partial_lut_transmission_values = lut_transmission_table[rx_antenna][tx_beam][i][
                            lut_transmission_mask
                        ]

                        # apply
                        if len(partial_lut_incidence_angles) > 0 and len(partial_lut_transmission_angles) > 0:
                            bs_incidence_correction = np.interp(
                                partial_incidence_angles.flatten(),
                                partial_lut_incidence_angles,
                                partial_lut_incidence_values,
                            ).reshape(partial_incidence_angles.shape)
                            bs_transmission_correction = np.interp(
                                partial_transmission_angles.flatten(),
                                partial_lut_transmission_angles,
                                partial_lut_transmission_values,
                            ).reshape(partial_transmission_angles.shape)

                            partial_bs_corrected = (
                                partial_bs_values + bs_incidence_correction + bs_transmission_correction
                            )
                            # copy bs_corrected values to source data
                            bs_corrected[i, detection_mask] = partial_bs_corrected

            detection_variable = xsf.dataset[xsf_driver.DETECTION_BACKSCATTER_R]
            detection_variable[:] = bs_corrected

            # Update metadata
            xsf.update_processing_status(
                {xsf_driver.ATT_PROCESSING_STATUS_BACKSCATTER_CORRECTION: xsf_driver.ATT_PROCESSING_STATUS_FLAG_ON}
            )
            xsf.append_history_line(f"Backscatter sliding angular renormalization with PyAT")

            return bs_corrected
        except Exception as e:
            default_config.logger.error(f"An exception was thrown while computing : {str(e)}")
            raise e
        finally:
            # close the file
            if need_to_close:
                xsf.close()


def xsf_sliding_process(
    i_paths: List[str],
    o_paths: List[str],
    overwrite: bool = False,
    sounder_type: str = SounderType.AUTO,
    i_dtm: Optional[str] = None,
    o_bsar: Optional[str] = None,
    use_svp: bool = True,
    use_snippets: bool = True,
    use_insonified_area: bool = True,
    remove_calibration: bool = True,
    sliding_window: int = 10,
    ref_angle_min: float = 40,
    ref_angle_max: float = 60,
    monitor: ProgressMonitor = ProgressMonitor(),
):
    """
    Compute mean backscatter model of input files
    @param sounder_type : type from sounder_lib.SounderType
    @param i_paths : input file paths
    @param o_path : output file path
    @param overwrite : True to overwrite output files if needed
    @param i_dtm : input DTM used to compute seafloor incidence angles (optional)
    @param mask : geographic mask to limit extend of data used for stats
    @param use_svp : True to use sound velocity profile registered in input files
    @param use_snippets : True to recompute detection mean bs from snippets
    @param use_insonified_area : True to recompute insonified area from incidence seafloor angles
    @param remove_calibration : True to remove sounder calibration (BSCorr)
    @param sliding_window : Size of sliding window centered on current ping (in minutes)
    @param ref_angle_min : Lower bound of reference incidence angles range
    @param ref_angle_max : Upper bound of reference incidence angles range
    @param monitor : progress monitor
    """

    default_config.check_files_version(input_files=i_paths)
    default_config.set_use_snippets(use_snippets=use_snippets)
    default_config.set_use_svp(use_svp=use_svp)
    default_config.set_use_insonified_area(use_insonified_area=use_insonified_area)
    default_config.set_remove_calibration(remove_calibration=remove_calibration)
    default_config.set_integration_method(integration_method=IntegrationMethod.MEDIAN)
    default_config.monitor = monitor
    sounder_type = default_config.check_files_soundertype(input_files=i_paths, sounder_type=sounder_type)

    # Initialize storage for global data cache
    global_data = GlobalDataModel()
    # set output bsar filepath if provided
    global_data.model_file = o_bsar
    # prepare reference dtm for insonified area corrections and incidence angles
    dtm_angles_computer = DtmAnglesComputer(ref_path=i_dtm) if i_dtm is not None else None

    computer = SlidingMeanBSComputer(sounder_type=sounder_type, sliding_window_min=sliding_window)

    default_config.monitor.begin_task("Backscatter sliding angular renormalization", 2)
    # compute measured mean values per mode. store it in global_data.
    computer.compute(input_files=i_paths, dtm_angles_computer=dtm_angles_computer, global_data=global_data)

    split_monitor = default_config.monitor.split(1)
    split_monitor.begin_task("Renormalization", len(i_paths))

    bs_model = SlidingMeanBSModel.read_from_netcdf(input_file=global_data.model_file)

    # retrieve statistic data and renorm input_files
    default_config.logger.info(f"Initialize sliding angle normalizer")
    normalizer = SlidingAngleNormalizer(
        sounder_type=sounder_type,
        avg_model=SlidingModel(
            mean_bs=bs_model,
            ref_angles=(ref_angle_min, ref_angle_max),
            sliding_window_min=sliding_window,
        ),
        global_data=global_data,
    )

    for f, output_file in zip(i_paths, o_paths):
        # read bs to retrieve non-corrected values
        default_config.logger.info(f"Processing {f} to {output_file}")
        tmp_outputfile = tempfile.mktemp(suffix=os.path.basename(f))
        # Copy input file to output path"
        if not overwrite and os.path.exists(output_file):
            default_config.logger.warning(
                f"File {output_file} already exists and overwrite is not allowed, skipping it"
            )
            continue

        # create temp file
        default_config.logger.info(f"Using {tmp_outputfile} as a temporary outputfile")
        shutil.copy(f, tmp_outputfile)

        # apply corrected backscatter values
        normalizer.apply_on_file(input_filepath=f, output_file=tmp_outputfile, dtm_angles_computer=dtm_angles_computer)

        # everything went well, copy the result
        shutil.move(tmp_outputfile, output_file)

        split_monitor.worked(1)

    # release model
    bs_model = None
    # clean temporary model file
    if o_bsar is None:
        os.remove(global_data.model_file)

    split_monitor.done()
    default_config.monitor.done()
