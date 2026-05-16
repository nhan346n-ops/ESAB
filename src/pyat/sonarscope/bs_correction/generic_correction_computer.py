import numpy as np

from pyat.sonarscope.bs_correction.dtm_angles_computer import DtmAnglesComputer
from pyat.sonarscope.model.constants import VariableKeys as Key
from pyat.sonarscope.model.signal.ping_detection_signal import PingDetectionSignal
from pyat.sonarscope.model.signal.ping_signal import PingSignal
from ...utils import signal
from ..common.configuration import default_config


class GenericCorrectionComputer:
    def __init__(
        self,
        ping_timed_dataset: PingSignal,
        ping_detection_dataset: PingDetectionSignal,
        dtm_angles_computer: DtmAnglesComputer | None = None,
    ):
        self.ping_time_dataset = ping_timed_dataset
        self.ping_detection_dataset = ping_detection_dataset
        self.dtm_angles_computer = dtm_angles_computer
        self._across_incidence_angles = None
        self._along_incidence_angles = None
        self._incidence_angles = None

    def compute_lambert_correction(self) -> np.ndarray:
        return None

    def compute_specular_correction(self) -> np.ndarray:
        return None

    def compute_across_along_seafloor_incidence_angle(self) -> (np.ndarray, np.ndarray):
        """
        Compute seafloor incidence angles (degrees)
        """
        if self._across_incidence_angles is None or self._along_incidence_angles is None:
            if default_config.use_svp:
                self.ping_detection_dataset.read([Key.DETECTION_INCIDENCE_ANGLE])
                detection_incidence_angle = self.ping_detection_dataset.xr_dataset[Key.DETECTION_INCIDENCE_ANGLE].data
            else:
                self.ping_detection_dataset.read([Key.DETECTION_BEAM_POINTING_ANGLE_REF_VERTICAL])
                detection_incidence_angle = self.ping_detection_dataset.xr_dataset[
                    Key.DETECTION_BEAM_POINTING_ANGLE_REF_VERTICAL
                ].data
            self.ping_detection_dataset.read([Key.TX_TILT_ANGLE_REF_VERTICAL])
            detection_tx_tilt = self.ping_detection_dataset.xr_dataset[Key.TX_TILT_ANGLE_REF_VERTICAL].data

            along_incidence_deg = detection_tx_tilt
            across_incidence_deg = detection_incidence_angle

            # Compute dtm slopes in SCS
            if self.dtm_angles_computer is not None:
                self.ping_time_dataset.read([Key.PLATFORM_HEADING])
                platform_heading = self.ping_time_dataset.xr_dataset[Key.PLATFORM_HEADING].data
                self.ping_detection_dataset.read(
                    [
                        Key.DETECTION_LONGITUDE,
                        Key.DETECTION_LATITUDE,
                    ]
                )
                detection_longitudes = self.ping_detection_dataset.xr_dataset[Key.DETECTION_LONGITUDE].data
                detection_latitudes = self.ping_detection_dataset.xr_dataset[Key.DETECTION_LATITUDE].data
                across_slope, along_slope = self.dtm_angles_computer.retrieve_across_along_slope_from_lonlat(
                    longitudes=detection_longitudes, latitudes=detection_latitudes, source_headings=platform_heading
                )
                # Compute full incidence angles
                along_incidence_deg = along_incidence_deg + along_slope
                across_incidence_deg = across_incidence_deg + across_slope

            # clip values to [-90, 90] degrees to avoid numerical issues in later computations
            self._across_incidence_angles = np.clip(across_incidence_deg, -90.0, 90.0)
            self._along_incidence_angles = np.clip(along_incidence_deg, -90.0, 90.0)

        return self._across_incidence_angles, self._along_incidence_angles

    def compute_seafloor_incidence_angle(self) -> np.ndarray:
        """
        Compute seafloor incidence angles (degrees)
        """
        if self._incidence_angles is None:
            (
                across_incidence_angles,
                along_incidence_angles,
            ) = self.compute_across_along_seafloor_incidence_angle()
            self._incidence_angles = np.rad2deg(
                np.arctan(
                    np.sqrt(
                        np.square(np.tan(np.deg2rad(across_incidence_angles)))
                        + np.square(np.tan(np.deg2rad(along_incidence_angles)))
                    )
                )
            )
        return self._incidence_angles

    def compute_insonified_area_db(self) -> np.ndarray:
        """
        Compute insonified area as used by Ifremer (dB)
        """
        self.ping_time_dataset.read([Key.SOUND_SPEED_AT_TRANSDUCER, Key.TX_BEAMWIDTH, Key.RX_BEAMWIDTH])
        sound_speed = self.ping_time_dataset.xr_dataset[Key.SOUND_SPEED_AT_TRANSDUCER].data
        along_beamwidth_deg = self.ping_time_dataset.xr_dataset[Key.TX_BEAMWIDTH].data
        across_beamwidth_deg = self.ping_time_dataset.xr_dataset[Key.RX_BEAMWIDTH].data

        self.ping_detection_dataset.read(
            [
                Key.DETECTION_SAMPLING_FREQ,
                Key.DETECTION_BEAM_POINTING_ANGLE,
                Key.DETECTION_RANGE_SAMPLE,
                Key.PULSE_LENGTH_EFFECTIVE,
            ]
        )
        sampling_frequency = self.ping_detection_dataset.xr_dataset[Key.DETECTION_SAMPLING_FREQ].data
        detection_beam_pointing_angle = self.ping_detection_dataset.xr_dataset[Key.DETECTION_BEAM_POINTING_ANGLE].data

        detection_range_sample = self.ping_detection_dataset.xr_dataset[Key.DETECTION_RANGE_SAMPLE].data
        pulse_length_effective = self.ping_detection_dataset.xr_dataset[Key.PULSE_LENGTH_EFFECTIVE].data

        # Compute full incidence angles
        across_incidence_deg, along_incidence_deg = self.compute_across_along_seafloor_incidence_angle()
        seafloor_incidence_deg = self.compute_seafloor_incidence_angle()

        across_incidence_rad = np.deg2rad(across_incidence_deg)
        along_incidence_rad = np.deg2rad(along_incidence_deg)
        seafloor_incidence_rad = np.deg2rad(seafloor_incidence_deg)
        seafloor_aspect_rad = np.arctan2(np.tan(across_incidence_rad), np.tan(along_incidence_rad))

        # sonar_aire_insonifiee_dB.m

        # First compute beam opening angles. It depends on angle relative to transducer array as a physical effect of beamforming
        cos_angles = np.cos(np.deg2rad(detection_beam_pointing_angle))
        along_beam_opening = np.tan(np.deg2rad(along_beamwidth_deg[:, None]))
        across_beam_opening = np.tan(np.deg2rad(across_beamwidth_deg[:, None])) / cos_angles

        # Estimate range
        range_meter = detection_range_sample * sound_speed[:, None] / (2 * sampling_frequency)

        # Across resolution around normal incidence
        resol_across_normal = across_beam_opening * range_meter / np.cos(across_incidence_rad)
        # Along resolution around normal incidence
        resol_along_normal = along_beam_opening * range_meter / np.cos(along_incidence_rad)

        # Across resolution limited by pulse length
        # if False:
        # method with incidence angle
        resol_across_oblique = (
            sound_speed[:, None] * pulse_length_effective / (2 * np.abs(np.sin(seafloor_incidence_rad)))
        )
        resol_along_oblique = (
            np.sqrt(
                (along_beam_opening * np.sin(seafloor_aspect_rad)) ** 2
                + (across_beam_opening * np.cos(seafloor_aspect_rad)) ** 2
            )
            * range_meter
        )
        # else:
        #     # method with across incidence angle
        #     resol_across_oblique = (
        #         sound_speed[:, None] * pulse_length_effective / (2 * np.abs(np.sin(across_incidence_rad)))
        #     )
        #     resol_along_oblique = resol_along_normal

        # Consolidated across resolution
        resol = np.minimum(resol_across_oblique * resol_along_oblique, resol_across_normal * resol_along_normal)

        # Get insonified area as the product of resolutions
        insonified_area_db = signal.energy_to_db(resol)
        return insonified_area_db
