import os.path
import shutil
import tempfile
from typing import Union, List

import numpy as np

from pyat.sonarscope.bs_correction.bs_computer import BSComputer
from pyat.sonarscope.bs_correction.dtm_angles_computer import DtmAnglesComputer
from pyat.sonarscope.bs_correction.seafloor_bs_angular_model import ConstantModel
from pyat.sonarscope.bs_correction.stats_computer import MeanBSModel
from pyat.sonarscope.common.configuration import default_config
from pyat.sonarscope.model.constants import VariableKeys as Key
from pyat.sonarscope.model.signal.ping_detection_signal import PingDetectionSignal
from pyat.sonarscope.model.signal.ping_signal import PingSignal
from pyat.sonarscope.model.sonar_factories import ModeComputerFactory
from pyat.sonarscope.model.sounder_mode.sounder_modes_computer import remove_invalid_key
from pyat.utils import numpy_utils
from pyat.xsf import xsf_driver
from pyat.xsf.xsf_driver import XsfDriver


class AngleNormalizer:
    """
    Process to normalize backscatter given a mean angular bs model(bs stats and a reference value)
    """

    def __init__(self, sounder_type: str, avg_model: ConstantModel):
        self.sounder_type = sounder_type
        self.mode_computer = ModeComputerFactory.create_mode_computer(sounder_type)
        self.avg_model = avg_model

    def apply_on_file(
        self,
        input_file: Union[str, XsfDriver],
        dtm_angles_computer: DtmAnglesComputer = None,
        mean_model_file=None,
        apply_compensation: bool = True,
    ) -> np.ndarray:
        """Apply an averaged compensation on a single file"""
        need_to_close = False
        xsf = input_file
        try:
            if not isinstance(input_file, XsfDriver):
                xsf = xsf_driver.XsfDriver(file_path=input_file)
                need_to_close = True

            xsf.open(mode="r+")
            # initialize stuff
            key_dict = {}
            # retrieve mode information
            key_dict, sounder_mode_array = self.mode_computer.compute_xsf(xsf=xsf, global_keys=key_dict)
            key_dict = remove_invalid_key(key_dict)

            # create a model for data storage
            default_config.logger.info(f"Build ping detection model")
            ping_model = PingSignal(xsf_dataset=xsf)
            ping_detection_model = PingDetectionSignal(xsf_dataset=xsf)
            ping_detection_model.read(
                [
                    Key.DETECTION_TX_BEAM_INDEX,
                    Key.DETECTION_RX_TRANSDUCER_INDEX,
                    Key.DETECTION_BEAM_POINTING_ANGLE_REF_PLATFORM,
                ]
            )

            detection_tx_beam = ping_detection_model.xr_dataset[Key.DETECTION_TX_BEAM_INDEX].data
            detection_rx_transducer = ping_detection_model.xr_dataset[Key.DETECTION_RX_TRANSDUCER_INDEX].data
            detection_beam_pointing_angle_ref_platform = ping_detection_model.xr_dataset[
                Key.DETECTION_BEAM_POINTING_ANGLE_REF_PLATFORM
            ].data

            rx_antenna_index = xsf.get_rx_transducers()
            rx_antenna_count = len(rx_antenna_index)

            # compute backscatter
            bs_value, incidence_angles = BSComputer.compute_bs(
                ping_dataset=ping_model,
                ping_detection_dataset=ping_detection_model,
                dtm_angles_computer=dtm_angles_computer,
            )

            transmission_angles = detection_beam_pointing_angle_ref_platform

            # initialize output dataset
            bs_corrected = np.full_like(bs_value, fill_value=np.nan)

            default_config.logger.info(f"Apply normalisation")
            for mode, indexes in key_dict.items():
                lut_incidence_table, lut_incidence_angles = self.avg_model.get_avg_incidence_lut(mode)
                lut_transmission_table, lut_transmission_angles = self.avg_model.get_avg_residual_transmission_lut(mode)
                if lut_transmission_table is None or lut_incidence_table is None:
                    continue
                mode_mask = sounder_mode_array == indexes
                for rx_antenna in range(0, rx_antenna_count):
                    rx_mask = detection_rx_transducer == rx_antenna_index[rx_antenna]
                    if not np.any(rx_mask):
                        continue
                    for tx_beam in range(0, mode.get_tx_beam_count()):
                        # need to take into account for txsectors
                        detection_mask = detection_tx_beam == tx_beam
                        # remove data not matching rx_antenna
                        detection_mask[~rx_mask] = False
                        # remove data not matching mode selection
                        detection_mask[~mode_mask] = False

                        # need to interpol correction by angles
                        partial_incidence_angles = incidence_angles[detection_mask]
                        partial_transmission_angles = transmission_angles[detection_mask]
                        partial_bs_values = bs_value[detection_mask]

                        # need to remove nan values from lut
                        lut_incidence_mask = ~np.isnan(lut_incidence_table)
                        lut_transmission_mask = ~np.isnan(lut_transmission_table[rx_antenna][tx_beam])
                        partial_lut_incidence_angles = lut_incidence_angles[lut_incidence_mask]
                        partial_lut_incidence_values = lut_incidence_table[lut_incidence_mask]
                        partial_lut_transmission_angles = lut_transmission_angles[lut_transmission_mask]
                        partial_lut_transmission_values = lut_transmission_table[rx_antenna][tx_beam][
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
                            if apply_compensation:
                                partial_bs_corrected = (
                                    partial_bs_values + bs_incidence_correction + bs_transmission_correction
                                )
                            else:
                                partial_bs_corrected = partial_bs_values + bs_transmission_correction

                            # copy bs_corrected values to source data
                            bs_corrected[detection_mask] = partial_bs_corrected
            detection_variable = xsf.dataset[xsf_driver.DETECTION_BACKSCATTER_R]
            detection_variable[:] = bs_corrected

            # Update metadata
            xsf.update_processing_status(
                {xsf_driver.ATT_PROCESSING_STATUS_BACKSCATTER_CORRECTION: xsf_driver.ATT_PROCESSING_STATUS_FLAG_ON}
            )
            xsf.append_history_line(
                f"Backscatter angular renormalization (ref:{os.path.basename(mean_model_file)}) with PyAT"
            )

            return bs_corrected
        except Exception as e:
            default_config.logger.error(f"An exception was thrown while computing : {str(e)}")
            raise e
        finally:
            # close the file
            if need_to_close:
                xsf.close()


def xsf_constant_process(
    i_paths: List[str],
    o_paths: List[str],
    mean_model_file: str,
    overwrite: bool = False,
    i_dtm: str = None,
    apply_compensation: bool = True,
    reference_level: float = -20,
    use_snippets: bool = True,
):
    """
    Normalize backscatter of input files to fit a constant output model
    @param i_paths : input file paths
    @param o_paths : output file paths
    @param mean_model_file : mean angular bs response used to compensate backscatter of input files
    @param overwrite : True to overwrite output files if needed
    @param i_dtm : input DTM used to compute incidence angles for insonified area (optional)
    @param apply_compensation : True to apply compensation and fit a constant model. If False, only apply sector corrections
    @param reference_level : expected mean backscatter value used in normalization (dB)
    @param use_snippets : True to recompute mean backscatter value from snippets
    @param use_svp : True to use sound velocity profile registered in input files
    """

    numpy_utils.disable_warning()
    default_config.logger.info("Starting angular_renormalization process")
    default_config.check_files_version(input_files=i_paths)

    mean_model = MeanBSModel.read_from_netcdf(mean_model_file, apply_conf=True)

    # apply configuration
    default_config.set_use_snippets(use_snippets=use_snippets)
    default_config.setup(sounder_type=mean_model.sounder_type)

    # prepare reference dtm for insonified area corrections and incidence angles
    dtm_angles_computer = DtmAnglesComputer(ref_path=i_dtm) if i_dtm is not None else None

    normalizer = AngleNormalizer(
        sounder_type=mean_model.sounder_type, avg_model=ConstantModel(mean_bs=mean_model, bs_value=reference_level)
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

        # get corrected backscatter values
        normalizer.apply_on_file(
            input_file=tmp_outputfile,
            dtm_angles_computer=dtm_angles_computer,
            mean_model_file=mean_model_file,
            apply_compensation=apply_compensation,
        )

        # everything went well, copy the result
        shutil.move(tmp_outputfile, output_file)
