from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.stats import binned_statistic

from pyat.xsf import xsf_driver
from ...utils import argument_utils, numpy_utils
from ...utils.multiple_entry_dict import MultipleEntryDict
from ..common.configuration import IntegrationMethod, InterpolationMethod, LinearScale, default_config
from ..common.mask import compute_geo_mask_from_lon_lat
from ..model.constants import VariableKeys as Key
from ..model.signal.ping_detection_signal import PingDetectionSignal
from ..model.signal.ping_signal import PingSignal
from ..model.sonar_factories import ModeComputerFactory
from ..model.sounder_lib import SounderType
from ..model.sounder_mode.sounder_modes_computer import get_invalid_key_indices, get_valid_key_indices
from .bs_computer import BSComputer
from .dtm_angles_computer import DtmAnglesComputer
from .file_data import FileDataStore
from .global_data import GlobalDataModel
from .mean_bs_model import BackscatterCurve, BackscatterCurveByIncidence, BackscatterCurveByTransmission, MeanBSModel
from .mean_bs_utils import merge_incidence_curves, merge_transmission_curves


class MeanBSComputer:
    def __init__(self, sounder_type: Optional[str] = None):
        self.sounder_type = sounder_type
        numpy_utils.disable_warning()

    def _compute_by_incidence(
        self,
        input_files: List[str],
        global_data: GlobalDataModel,
        dtm_angles_computer: Optional[DtmAnglesComputer],
        mask_files: Optional[List[str]] = None,
    ) -> MeanBSModel:
        mode_curves = {}
        # create a dictionary with mode as a key and a list of xarray data set perfile containing statistics
        curves_by_incidence_per_mode_and_file = MultipleEntryDict()  # curve dictionary per mode

        valid_indices = get_valid_key_indices(key_dict=global_data.keymode_dict)

        rx_antenna_count = None
        rx_antenna_index = None

        #
        # START PROCESS PER FILE BY INCIDENCE
        #
        for file in input_files:
            file = str(file)  # need to convert to str which is the id used in dictionary
            default_config.logger.info(f"Compute mode and stats by incidence angle for {file}")
            modes_indexes = global_data.file_data[file].mode_indices  # index of active modes in files
            unique_modes = np.intersect1d(modes_indexes, valid_indices)  # retrieve list of valid modes defined in file
            # get beam angles values
            # get backscatter values
            with xsf_driver.open_xsf(file_path=file, mode="r") as xsf:
                # check minimal xsf version
                default_config.check_version(xsf_dataset=xsf)

                default_config.logger.info("Build ping detection model")
                ping_model = PingSignal(xsf_dataset=xsf)
                ping_detection_model = PingDetectionSignal(xsf_dataset=xsf)

                ping_detection_model.read(
                    [
                        Key.BATHYMETRY_STATUS,
                        Key.DETECTION_LONGITUDE,
                        Key.DETECTION_LATITUDE,
                        Key.DETECTION_TX_BEAM_INDEX,
                        Key.DETECTION_RX_TRANSDUCER_INDEX,
                    ]
                )

                status = ping_detection_model.xr_dataset[Key.BATHYMETRY_STATUS].data
                status_mask = status == 0
                detection_longitude = ping_detection_model.xr_dataset[Key.DETECTION_LONGITUDE].data
                detection_latitude = ping_detection_model.xr_dataset[Key.DETECTION_LATITUDE].data
                detection_tx_beam = ping_detection_model.xr_dataset[Key.DETECTION_TX_BEAM_INDEX].data
                detection_rx_transducer = ping_detection_model.xr_dataset[Key.DETECTION_RX_TRANSDUCER_INDEX].data

                geo_mask = np.full_like(status_mask, fill_value=True)
                if mask_files is not None and len(mask_files) > 0:
                    default_config.logger.info("Apply geographic mask")
                    geo_mask = compute_geo_mask_from_lon_lat(detection_longitude, detection_latitude, mask_files)
                    if not geo_mask.any():
                        default_config.logger.info("File outside of mask. Skipping it.")
                        continue

                file_antenna_index = xsf.get_rx_transducers()
                file_antenna_count = len(file_antenna_index) if file_antenna_index is not None else 0
                if rx_antenna_count is None:
                    rx_antenna_count = file_antenna_count
                    rx_antenna_index = file_antenna_index
                elif rx_antenna_count != file_antenna_count:
                    default_config.logger.warning("The rx antenna count does not match in this file.")

                bs_value, incidence_angles = BSComputer.compute_bs(
                    ping_dataset=ping_model,
                    ping_detection_dataset=ping_detection_model,
                    dtm_angles_computer=dtm_angles_computer,
                )
                global_data.file_data[file].bs_value = bs_value
                global_data.file_data[file].incidence_angle = incidence_angles

                for current_mode_idx in unique_modes:
                    mode = [k for k, v in global_data.keymode_dict.items() if v == current_mode_idx][0]
                    # ignore invalid mode
                    if not mode.is_valid():
                        continue
                    default_config.logger.info(f"Processing mode {mode}")

                    # compute stats by tx sector
                    mean_values_per_sector = np.full(
                        shape=(
                            rx_antenna_count,
                            mode.get_tx_beam_count(),
                            default_config.incidence_angles.bin_count,
                        ),
                        dtype=np.float64,
                        fill_value=np.nan,
                    )
                    value_counts_per_sector = np.full(
                        shape=(
                            rx_antenna_count,
                            mode.get_tx_beam_count(),
                            default_config.incidence_angles.bin_count,
                        ),
                        dtype=np.int32,
                        fill_value=0,
                    )

                    # resample data in order to be able to work mode per mode
                    # keep only data matching the current mode
                    mode_mask = modes_indexes == current_mode_idx

                    for rx_antenna in range(rx_antenna_count):
                        rx_mask = detection_rx_transducer == rx_antenna_index[rx_antenna]
                        if not np.any(rx_mask):
                            continue
                        for tx_beam in range(mode.get_tx_beam_count()):
                            # filter to keep data for this tx_beam
                            detection_mask = detection_tx_beam == tx_beam
                            # remove data not matching antenna
                            detection_mask[~rx_mask] = False
                            # remove data not matching mode
                            detection_mask[~mode_mask] = False
                            # remove data not valid
                            detection_mask[~status_mask] = False
                            # remove data outside geo_mask
                            detection_mask[~geo_mask] = False

                            incidence_angles_masked = incidence_angles[detection_mask]
                            bs_masked = bs_value[detection_mask]
                            flat_incidence_angles = incidence_angles_masked.ravel()
                            flat_bs = bs_masked.ravel()
                            flat_bs_linear = default_config.db_to_linear(bs_masked).ravel()
                            # remove Nan value, this can happen when missing datagram for example
                            values_to_remove = np.isnan(flat_bs_linear) | np.isnan(flat_incidence_angles)

                            flat_incidence_angles = flat_incidence_angles[~values_to_remove]
                            flat_bs_linear = flat_bs_linear[~values_to_remove]
                            flat_bs = flat_bs[~values_to_remove]

                            if default_config.integration_method is IntegrationMethod.MEAN:
                                mean_method = "mean"
                            else:
                                mean_method = "median"

                            stat_count, bin_edges, _ = binned_statistic(
                                x=flat_incidence_angles,
                                values=flat_bs,
                                statistic="count",
                                bins=default_config.incidence_angles.bin_count,
                                range=default_config.incidence_angles.angle_range,
                            )
                            stat_count = stat_count.astype(np.int32)

                            stat_mean_linear, _, _ = binned_statistic(
                                x=flat_incidence_angles,
                                values=flat_bs_linear,
                                statistic=mean_method,
                                bins=default_config.incidence_angles.bin_count,
                                range=default_config.incidence_angles.angle_range,
                            )
                            stat_mean = default_config.linear_to_db(stat_mean_linear)

                            mean_values_per_sector[rx_antenna][tx_beam] = stat_mean
                            value_counts_per_sector[rx_antenna][tx_beam] = stat_count

                    # create curve by incidence
                    curve_by_incidence = BackscatterCurveByIncidence.build(
                        mean_values=mean_values_per_sector,
                        count=value_counts_per_sector,
                        bin_centers=default_config.incidence_angles.bin_centers,
                        origin=file,
                    )

                    # recompute mean of each file
                    # retain mean and per mode
                    curves_by_incidence_per_mode_and_file.add(key=mode, obj=curve_by_incidence)
        #
        # END PROCESS PER FILE PER INCIDENCE
        #

        #
        # Compute synthesis per incidence
        #

        # for each mode, recompute all means
        for mode in curves_by_incidence_per_mode_and_file.keys():
            if not mode.is_valid():
                continue
            values = curves_by_incidence_per_mode_and_file.get(mode)
            merged_curve = merge_incidence_curves(input_curves=values, apply_filtering=True)
            mode_curves[mode] = (merged_curve, None)
        return MeanBSModel(self.sounder_type, mode_curves)

    def _compute_by_transmission(
        self,
        input_files: List[str],
        global_data: GlobalDataModel,
        incidence_meanbsmodel: MeanBSModel,
        dtm_angles_computer: Optional[DtmAnglesComputer],
        mask_files: Optional[List[str]] = None,
    ) -> MeanBSModel:

        # create a dictionary with mode as a key and a list of xarray data set perfile containing statistics
        curves_by_transmission_per_mode_and_file = MultipleEntryDict()  # curve dictionary per mode

        rx_antenna_count = None
        rx_antenna_index = None

        valid_indices = get_valid_key_indices(key_dict=global_data.keymode_dict)
        #
        # START PROCESS PER FILE BY TRANSMISSION
        #
        for file in input_files:
            file = str(file)  # need to convert to str which is the id used in dictionary
            default_config.logger.info(f"Compute mode and stats by transmission angle for {file}")
            modes_indexes = global_data.file_data[file].mode_indices  # index of active modes in files
            unique_modes = np.intersect1d(modes_indexes, valid_indices)  # retrieve list of valid modes defined in file
            # get beam angles values
            # get backscatter values
            with xsf_driver.open_xsf(file_path=file, mode="r") as xsf:
                # check minimal xsf version
                default_config.check_version(xsf_dataset=xsf)

                default_config.logger.info("Build ping detection model")
                ping_model = PingSignal(xsf_dataset=xsf)
                ping_detection_model = PingDetectionSignal(xsf_dataset=xsf)

                ping_detection_model.read(
                    [
                        Key.DETECTION_LONGITUDE,
                        Key.DETECTION_LATITUDE,
                        Key.DETECTION_TX_BEAM_INDEX,
                        Key.DETECTION_RX_TRANSDUCER_INDEX,
                        Key.BATHYMETRY_STATUS,
                        Key.DETECTION_BEAM_POINTING_ANGLE_REF_PLATFORM,
                    ]
                )

                status = ping_detection_model.xr_dataset[Key.BATHYMETRY_STATUS].data
                status_mask = status == 0
                detection_longitude = ping_detection_model.xr_dataset[Key.DETECTION_LONGITUDE].data
                detection_latitude = ping_detection_model.xr_dataset[Key.DETECTION_LATITUDE].data
                detection_tx_beam = ping_detection_model.xr_dataset[Key.DETECTION_TX_BEAM_INDEX].data
                detection_rx_transducer = ping_detection_model.xr_dataset[Key.DETECTION_RX_TRANSDUCER_INDEX].data
                detection_beam_pointing_angle_ref_platform = ping_detection_model.xr_dataset[
                    Key.DETECTION_BEAM_POINTING_ANGLE_REF_PLATFORM
                ].data

                geo_mask = np.full_like(status_mask, fill_value=True)
                if mask_files is not None and len(mask_files) > 0:
                    default_config.logger.info("Apply geographic mask")
                    geo_mask = compute_geo_mask_from_lon_lat(detection_longitude, detection_latitude, mask_files)
                    if not geo_mask.any():
                        default_config.logger.info("File outside of mask. Skipping it.")
                        continue

                file_antenna_index = xsf.get_rx_transducers()
                file_antenna_count = len(file_antenna_index) if file_antenna_index is not None else 0
                if rx_antenna_count is None:
                    rx_antenna_count = file_antenna_count
                    rx_antenna_index = file_antenna_index
                elif rx_antenna_count != file_antenna_count:
                    default_config.logger.warning("The rx antenna count does not match in this file.")

                # compute backscatter
                if len(global_data.file_data[file].bs_value) and len(global_data.file_data[file].incidence_angle):
                    bs_value = global_data.file_data[file].bs_value
                    incidence_angles = global_data.file_data[file].incidence_angle
                else:
                    bs_value, incidence_angles = BSComputer.compute_bs(
                        ping_dataset=ping_model,
                        ping_detection_dataset=ping_detection_model,
                        dtm_angles_computer=dtm_angles_computer,
                    )

                transmission_angles = detection_beam_pointing_angle_ref_platform

                for current_mode_idx in unique_modes:
                    mode = [k for k, v in global_data.keymode_dict.items() if v == current_mode_idx][0]
                    # ignore invalid mode
                    if not mode.is_valid():
                        continue
                    default_config.logger.info(f"Processing mode {mode} with {mode.get_tx_beam_count()} tx beams")

                    # retrieve curve by incidence
                    curve_by_incidence = incidence_meanbsmodel.get_curve_by_incidence(mode)
                    if not curve_by_incidence:
                        default_config.logger.warning(
                            f"No matching mode {mode} in input meanbsmodel for file {file}, ignoring mode"
                        )
                        continue

                    # compute stats by tx angle
                    mean_values_per_tx = np.full(
                        shape=(
                            rx_antenna_count,
                            mode.get_tx_beam_count(),
                            default_config.transmission_angles.bin_count,
                        ),
                        dtype=np.float64,
                        fill_value=np.nan,
                    )
                    mean_diff_values_per_tx = np.full(
                        shape=(
                            rx_antenna_count,
                            mode.get_tx_beam_count(),
                            default_config.transmission_angles.bin_count,
                        ),
                        dtype=np.float64,
                        fill_value=np.nan,
                    )
                    value_counts_per_tx = np.full(
                        shape=(
                            rx_antenna_count,
                            mode.get_tx_beam_count(),
                            default_config.transmission_angles.bin_count,
                        ),
                        dtype=np.int32,
                        fill_value=0,
                    )

                    # resample data in order to be able to work mode per mode
                    # keep only data matching the current mode
                    mode_mask = modes_indexes == current_mode_idx

                    for rx_antenna in range(rx_antenna_count):
                        rx_mask = detection_rx_transducer == rx_antenna_index[rx_antenna]
                        if not np.any(rx_mask):
                            continue
                        for tx_beam in range(mode.get_tx_beam_count()):
                            # filter to keep data for this tx_beam
                            detection_mask = detection_tx_beam == tx_beam
                            # remove data not matching antenna
                            detection_mask[~rx_mask] = False
                            # remove data not matching mode
                            detection_mask[~mode_mask] = False
                            # remove data not valid
                            detection_mask[~status_mask] = False
                            # remove data outside geo_mask
                            detection_mask[~geo_mask] = False

                            transmission_angles_masked = transmission_angles[detection_mask]
                            incidence_angles_masked = incidence_angles[detection_mask]

                            bs_masked = bs_value[detection_mask]
                            # HERE FILTER BY TX ANGLES

                            flat_transmission_angles = transmission_angles_masked.ravel()
                            flat_incidence_angles = incidence_angles_masked.ravel()

                            flat_bs = bs_masked.ravel()
                            flat_bs_linear = default_config.db_to_linear(bs_masked).ravel()

                            # retrieve corresponding bs by incidence angle in bs model (extrapolate with left and right values)
                            source_mean_bs_by_incidence = curve_by_incidence.ds[BackscatterCurve.MEAN_BS]
                            if source_mean_bs_by_incidence.shape[0] == 1:
                                # if only one rx antenna, use the same curve for all rx antenna
                                source_rx_antenna = 0
                            else:
                                source_rx_antenna = rx_antenna
                            if source_mean_bs_by_incidence.shape[1] == 1:
                                # if only one tx beam, use the same curve for all sectors
                                source_tx_beam = 0
                            else:
                                source_tx_beam = tx_beam
                            source_x = curve_by_incidence.ds[BackscatterCurve.ANGLE][:]
                            source_y = curve_by_incidence.ds[BackscatterCurve.MEAN_BS][
                                source_rx_antenna, source_tx_beam, :
                            ]
                            source_mask = np.isnan(source_y)
                            if source_mask.all():
                                continue
                            flat_bs_incidence = np.interp(
                                flat_incidence_angles, source_x[~source_mask], source_y[~source_mask]
                            )

                            flat_bs_diff = flat_bs - flat_bs_incidence
                            flat_bs_diff_linear = default_config.db_to_linear(flat_bs_diff)

                            # remove Nan value, this can happen when missing datagram for example
                            values_to_remove = (
                                np.isnan(flat_bs_linear)
                                | np.isnan(flat_transmission_angles)
                                | np.isnan(flat_incidence_angles)
                                | np.isnan(flat_bs_diff_linear)
                            )
                            if values_to_remove.all():
                                continue

                            flat_transmission_angles = flat_transmission_angles[~values_to_remove]
                            flat_bs_linear = flat_bs_linear[~values_to_remove]
                            flat_bs = flat_bs[~values_to_remove]
                            flat_bs_diff_linear = flat_bs_diff_linear[~values_to_remove]

                            # Compute stats
                            if default_config.integration_method is IntegrationMethod.MEAN:
                                mean_method = "mean"
                            else:
                                mean_method = "median"
                            stat_count, _, _ = binned_statistic(
                                x=flat_transmission_angles,
                                values=flat_bs,
                                statistic="count",
                                bins=default_config.transmission_angles.bin_count,
                                range=default_config.transmission_angles.angle_range,
                            )
                            stat_count = stat_count.astype(np.int32)
                            stat_mean_linear, _, _ = binned_statistic(
                                x=flat_transmission_angles,
                                values=flat_bs_linear,
                                statistic=mean_method,
                                bins=default_config.transmission_angles.bin_count,
                                range=default_config.transmission_angles.angle_range,
                            )
                            stat_mean_diff_linear, _, _ = binned_statistic(
                                x=flat_transmission_angles,
                                values=flat_bs_diff_linear,
                                statistic=mean_method,
                                bins=default_config.transmission_angles.bin_count,
                                range=default_config.transmission_angles.angle_range,
                            )
                            stat_mean = default_config.linear_to_db(stat_mean_linear)
                            stat_mean_diff = default_config.linear_to_db(stat_mean_diff_linear)

                            mean_values_per_tx[rx_antenna][tx_beam] = stat_mean
                            mean_diff_values_per_tx[rx_antenna][tx_beam] = stat_mean_diff
                            value_counts_per_tx[rx_antenna][tx_beam] = stat_count

                    # create curve for all
                    curve_by_transmission = BackscatterCurveByTransmission.build(
                        mean_values=mean_values_per_tx,
                        mean_residual_values=mean_diff_values_per_tx,
                        count=value_counts_per_tx,
                        bin_centers=default_config.transmission_angles.bin_centers,
                        origin=file,
                    )

                    # recompute mean of each file
                    # retain mean and per mode
                    curves_by_transmission_per_mode_and_file.add(key=mode, obj=curve_by_transmission)
        #
        # END PROCESS PER FILE BY TRANSMISSION
        #

        curve_per_mode = {}
        #
        # Compute  synthesis
        #
        # for each mode, recompute all means
        for mode in curves_by_transmission_per_mode_and_file.keys():
            if not mode.is_valid():
                continue
            values = curves_by_transmission_per_mode_and_file.get(mode)
            curve_by_transmission = merge_transmission_curves(input_curves=values, apply_filtering=True)

            # retrieve curve by incidence
            curve_by_incidence = incidence_meanbsmodel.get_curve_by_incidence(mode)
            if not curve_by_incidence:
                default_config.logger.warning(f"No matching mode {mode} in input meanbsmodel, ignoring mode")
                continue

            curve_per_mode[mode] = (curve_by_incidence, curve_by_transmission)

        # for each mode, catenate incidence and transmission curves
        # for mode, curve_by_transmission in curve_by_transmission_per_mode.items():
        #     curve_per_mode[mode] = (curve_by_incidence_per_mode[mode], curve_by_transmission)
        return MeanBSModel(sounder_type=self.sounder_type, mode_curves=curve_per_mode)

    def compute(
        self,
        input_files: List[str],
        input_dtm: Optional[str] = None,
        input_meanmodel: Optional[MeanBSModel] = None,
        mask_files: Optional[List[str]] = None,
    ) -> MeanBSModel:
        """
        Compute list of mode for input file and compute mean values for backscatter detection values.
        Statistics are computed in two steps : the first one per mode and per file, then mean values are concatenated along files.

        """
        default_config.check_files_version(input_files=input_files)
        default_config.logger.info("Compute list of available mode")
        mode_computer = ModeComputerFactory.create_mode_computer(self.sounder_type)
        key_dict, mode_ids_dict = mode_computer.compute(input_files)

        # set config
        default_config.setup(self.sounder_type)

        valid_indices = get_valid_key_indices(key_dict=key_dict)
        valid_count = len(valid_indices)
        invalid_indices = get_invalid_key_indices(key_dict=key_dict)
        invalid_count = len(invalid_indices)
        default_config.logger.info(f"Found {valid_count} valid modes, {invalid_count} invalid mode ")

        # prepare reference dtm for insonified area corrections
        dtm_angles_computer = DtmAnglesComputer(ref_path=input_dtm) if input_dtm is not None else None

        # Initialize storage for global data cache
        global_data = GlobalDataModel()
        global_data.keymode_dict = key_dict
        for file in input_files:
            filedatastore = FileDataStore(file, mode_indices=mode_ids_dict[file])
            global_data.file_data[file] = filedatastore

        if input_meanmodel:
            incidence_meanbsmodel = input_meanmodel
        else:
            incidence_meanbsmodel = self._compute_by_incidence(
                input_files=input_files,
                dtm_angles_computer=dtm_angles_computer,
                global_data=global_data,
                mask_files=mask_files,
            )

        return self._compute_by_transmission(
            input_files=input_files,
            dtm_angles_computer=dtm_angles_computer,
            incidence_meanbsmodel=incidence_meanbsmodel,
            global_data=global_data,
            mask_files=mask_files,
        )


def compute_mean_model(
    i_paths: List[str],
    o_path: str,
    overwrite: bool = False,
    sounder_type: str = SounderType.AUTO,
    i_dtm: Optional[str] = None,
    i_meanmodel: Optional[str] = None,
    mask: Optional[List[str]] = None,
    use_svp: bool = True,
    use_snippets: bool = True,
    use_insonified_area: bool = True,
    use_reference_by_sector: bool = True,
    remove_calibration: bool = True,
    remove_compensation: bool = True,
):
    """
    Compute mean backscatter model of input files
    @param sounder_type : type from sounder_lib.SounderType
    @param i_paths : input file paths
    @param o_path : output file path
    @param overwrite : True to overwrite output files if needed
    @param i_dtm : input DTM used to compute seafloor incidence angles (optional)
    @param i_meanmodel : input MeanBSModel(.bsar) used to fix bs model by incidence angle (optional)
    @param mask : geographic mask to limit extend of data used for stats
    @param use_svp : True to use sound velocity profile registered in input files
    @param use_snippets : True to recompute detection mean bs from snippets
    @param use_insonified_area : True to recompute insonified area from incidence seafloor angles
    @param use_reference_by_sector : True to use a reference incidence curves by sector (for calibration)
    @param remove_calibration : True to remove sounder calibration (BSCorr from kmall only)
    @param remove_compensation : True to remove angular backscatter compensation (Lambert + specular)
    """
    mask_files = argument_utils.parse_list_of_files("mask", mask)
    default_config.check_files_version(input_files=i_paths)
    default_config.set_use_snippets(use_snippets=use_snippets)
    default_config.set_use_svp(use_svp=use_svp)
    default_config.set_use_insonified_area(use_insonified_area=use_insonified_area)
    default_config.set_remove_calibration(remove_calibration=remove_calibration)
    default_config.set_remove_compensation(remove_compensation=remove_compensation)
    default_config.set_use_reference_by_sector(use_reference_by_sector=use_reference_by_sector)

    sounder_type = default_config.check_files_soundertype(input_files=i_paths, sounder_type=sounder_type)

    if i_meanmodel:
        incidence_meanmodel = MeanBSModel.read_from_netcdf(input_file=i_meanmodel, apply_conf=False)
    else:
        incidence_meanmodel = None

    computer = MeanBSComputer(sounder_type=sounder_type)
    # compute measured mean values per mode
    mean_model = computer.compute(
        input_files=i_paths, input_dtm=i_dtm, input_meanmodel=incidence_meanmodel, mask_files=mask_files
    )
    # retrieve statistic data and compute all curves per mode
    mean_model.save_to_netcdf(output_file=o_path, overwrite=overwrite)
    return mean_model


def compute_mean_model_process(
    i_paths: List[str],
    o_path: str,
    overwrite: bool = False,
    sounder_type: str = SounderType.AUTO,
    i_dtm: Optional[str] = None,
    i_meanmodel: Optional[str] = None,
    mask: Optional[List[str]] = None,
    use_svp: bool = True,
    use_snippets: bool = True,
    use_insonified_area: bool = True,
    remove_compensation: bool = True,
    remove_calibration: bool = True,
    integration_method: str = IntegrationMethod.MEAN.name,
    linear_scale: str = LinearScale.ENERGY.name,
    frequency_interpolation_method: str = InterpolationMethod.LINEAR.name,
    use_reference_by_sector: bool = True,
) -> None:
    """
    Compute mean backscatter model of input files
    @param sounder_type : type from sounder_lib.SounderType
    @param i_paths : input file paths
    @param o_path : output file path
    @param overwrite : True to overwrite output files if needed
    @param i_dtm : input DTM used to compute seafloor incidence angles (optional)
    @param i_meanmodel : input MeanBSModel(.bsar) used to fix bs model by incidence angle (optional)
    @param mask : geographic mask to limit extend of data used for stats
    @param use_svp : True to use sound velocity profile registered in input files
    @param use_snippets : True to recompute detection mean bs from snippets
    @param use_insonified_area : True to recompute insonified area from incidence seafloor angles
    @param remove_calibration : True to remove sounder calibration (BSCorr from kmall only)
    @param integration_method : method to use to integrate mean values (MEAN or MEDIAN)
    @param linear_scale : scale to use for mean value integration (ENERGY or AMPLITUDE)
    @param frequency_interpolation_method : method to use for interpolating frequencies from reference incidence curves (NEAREST or LINEAR)
    @param use_reference_by_sector : True to use a reference incidence curves by sector
    """

    default_config.set_integration_method(integration_method=IntegrationMethod[integration_method])
    default_config.set_linear_scale(working_scale=LinearScale[linear_scale])
    default_config.set_frequency_interpolation_method(
        interpolation_method=InterpolationMethod[frequency_interpolation_method]
    )
    compute_mean_model(
        sounder_type=sounder_type,
        i_paths=i_paths,
        o_path=o_path,
        overwrite=overwrite,
        i_dtm=i_dtm,
        i_meanmodel=i_meanmodel,
        mask=mask,
        use_svp=use_svp,
        use_snippets=use_snippets,
        use_insonified_area=use_insonified_area,
        remove_compensation=remove_compensation,
        remove_calibration=remove_calibration,
        use_reference_by_sector=use_reference_by_sector,
    )
