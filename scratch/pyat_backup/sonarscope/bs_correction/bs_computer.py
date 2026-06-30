from __future__ import annotations

from typing import Tuple

import numpy as np

from .dtm_angles_computer import DtmAnglesComputer
from .generic_correction_computer import GenericCorrectionComputer
from .kongsberg_correction_computer import KongsbergCorrectionComputer
from ..common.configuration import default_config
from ..model.constants import VariableKeys as Key
from ..model.signal.ping_detection_signal import PingDetectionSignal
from ..model.signal.ping_signal import PingSignal


class BSComputer:
    @staticmethod
    def compute_bs(
        ping_dataset: PingSignal,
        ping_detection_dataset: PingDetectionSignal,
        dtm_angles_computer: DtmAnglesComputer | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute corrected backscatter detection values applying constructor corrections.
        Returns : Tuple array with
            - corrected backscatter values
            - used incidence angles
        """

        if default_config.use_snippets:
            if default_config.remove_compensation:
                default_config.logger.info("Compute uncompensated backscatter from snippets")
                ping_detection_dataset.read([Key.DETECTION_BS_SNIPPETS_MEAN_WITHOUT_LAMBERT_COMP])
            else:
                default_config.logger.info("Compute raw backscatter from snippets")
                ping_detection_dataset.read([Key.DETECTION_BS_SNIPPETS_MEAN])

        # retrieve backscatter from snippets or directly from mean variable
        if (
            default_config.use_snippets
            and default_config.remove_compensation
            and Key.DETECTION_BS_SNIPPETS_MEAN_WITHOUT_LAMBERT_COMP in ping_detection_dataset.xr_dataset
        ):
            bs = ping_detection_dataset.xr_dataset[Key.DETECTION_BS_SNIPPETS_MEAN_WITHOUT_LAMBERT_COMP].data
        elif (
            default_config.use_snippets
            and not default_config.remove_compensation
            and Key.DETECTION_BS_SNIPPETS_MEAN in ping_detection_dataset.xr_dataset
        ):
            bs = ping_detection_dataset.xr_dataset[Key.DETECTION_BS_SNIPPETS_MEAN].data
        elif default_config.remove_compensation:
            default_config.logger.info("Compute uncompensated backscatter")
            ping_detection_dataset.read([Key.DETECTION_BACKSCATTER_WITHOUT_COMP])
            bs = ping_detection_dataset.xr_dataset[Key.DETECTION_BACKSCATTER_WITHOUT_COMP].data
        else:
            default_config.logger.info("Compute raw backscatter")
            ping_detection_dataset.read([Key.DETECTION_BACKSCATTER])
            bs = ping_detection_dataset.xr_dataset[Key.DETECTION_BACKSCATTER].data

        # remove calibration if asked
        if default_config.remove_calibration:
            ping_detection_dataset.read([Key.DETECTION_BACKSCATTER_CALIBRATION])
            if Key.DETECTION_BACKSCATTER_CALIBRATION in ping_detection_dataset.xr_dataset:
                bs_calibration = ping_detection_dataset.xr_dataset[Key.DETECTION_BACKSCATTER_CALIBRATION].data
                # check real presence of bs_calibration
                if np.any(np.isfinite(bs_calibration[:])):
                    bs[:] = bs[:] + bs_calibration[:]
                else:
                    default_config.logger.warning("No calibration info")
                    default_config.set_remove_calibration(False)

        generic_computer = GenericCorrectionComputer(
            ping_timed_dataset=ping_dataset,
            ping_detection_dataset=ping_detection_dataset,
            dtm_angles_computer=dtm_angles_computer,
        )
        if default_config.use_insonified_area:
            default_config.logger.info("Apply insonified area correction")
            kongsberg_computer = KongsbergCorrectionComputer(
                ping_timed_dataset=ping_dataset, ping_detection_dataset=ping_detection_dataset
            )
            km_insonified_area = kongsberg_computer.compute_insonified_area_db()
            ifr_insonified_area = generic_computer.compute_insonified_area_db()
            bs[:] = bs[:] + km_insonified_area[:] - ifr_insonified_area[:]

        # retrieve incidence angles
        incidence_angles = generic_computer.compute_seafloor_incidence_angle()

        return bs[:], incidence_angles[:]
