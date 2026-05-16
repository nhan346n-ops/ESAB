import numpy as np

from pyat.sonarscope.bs_correction.kongsberg_correction import (
    lambert_correction,
    specular_correction,
    insonified_area_db,
)
from pyat.sonarscope.model.signal.ping_detection_signal import PingDetectionSignal
from pyat.sonarscope.model.signal.ping_signal import PingSignal
from pyat.sonarscope.model.constants import VariableKeys as Key


class KongsbergCorrectionComputer:
    def __init__(
        self,
        ping_timed_dataset: PingSignal,
        ping_detection_dataset: PingDetectionSignal,
    ):
        self.ping_time_dataset = ping_timed_dataset
        self.ping_detection_dataset = ping_detection_dataset

    def compute_lambert_correction(self) -> np.ndarray:
        """
        Returns the correction to apply on backscatter to compensate lamberts law (ping/detection)
        """
        self.ping_detection_dataset.read([Key.RANGE_TO_NORMAL_INCIDENCE, Key.DETECTION_RANGE_SAMPLE])

        detection_range = self.ping_detection_dataset.xr_dataset[Key.DETECTION_RANGE_SAMPLE].data
        range_to_normal_incidence = self.ping_detection_dataset.xr_dataset[Key.RANGE_TO_NORMAL_INCIDENCE].data

        return lambert_correction(detection_range=detection_range, range_to_normal_incidence=range_to_normal_incidence)

    def compute_specular_correction(self) -> np.ndarray:
        """
        Returns the correction to apply on backscatter to compensate specular effect (ping/detection)
        """
        self.ping_detection_dataset.read(
            [
                Key.RANGE_TO_NORMAL_INCIDENCE,
                Key.DETECTION_RANGE_SAMPLE,
                Key.BACKSCATTER_NORMAL_INCIDENCE_LEVEL,
                Key.BACKSCATTER_OBLIQUE_INCIDENCE_LEVEL,
                Key.TVG_LAW_CROSSOVER_ANGLE,
            ]
        )

        detection_range = self.ping_detection_dataset.xr_dataset[Key.DETECTION_RANGE_SAMPLE].data
        range_to_normal_incidence = self.ping_detection_dataset.xr_dataset[Key.RANGE_TO_NORMAL_INCIDENCE].data

        backscatter_normal_incidence_level = self.ping_detection_dataset.xr_dataset[
            Key.BACKSCATTER_NORMAL_INCIDENCE_LEVEL
        ].data
        backscatter_oblique_incidence_level = self.ping_detection_dataset.xr_dataset[
            Key.BACKSCATTER_OBLIQUE_INCIDENCE_LEVEL
        ].data
        tvg_law_crossover_angle = self.ping_detection_dataset.xr_dataset[Key.TVG_LAW_CROSSOVER_ANGLE].data

        return specular_correction(
            detection_range=detection_range,
            range_to_normal_incidence=range_to_normal_incidence,
            backscatter_normal_incidence_level=backscatter_normal_incidence_level,
            backscatter_oblique_incidence_level=backscatter_oblique_incidence_level,
            tvg_law_crossover_angle=tvg_law_crossover_angle,
        )

    def compute_insonified_area_db(self) -> np.ndarray:
        """
        Returns the insonified area (dB) as a ping/detection array as used by Kongsberg to retrieve BS values from BTS
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
                Key.RANGE_TO_NORMAL_INCIDENCE,
                Key.PULSE_LENGTH_REALTIME,
            ]
        )
        sampling_frequency = self.ping_detection_dataset.xr_dataset[Key.DETECTION_SAMPLING_FREQ].data
        detection_beam_pointing_angle = self.ping_detection_dataset.xr_dataset[Key.DETECTION_BEAM_POINTING_ANGLE].data
        detection_range_sample = self.ping_detection_dataset.xr_dataset[Key.DETECTION_RANGE_SAMPLE].data
        range_to_normal_incidence = self.ping_detection_dataset.xr_dataset[Key.RANGE_TO_NORMAL_INCIDENCE].data
        pulse_length_realtime = self.ping_detection_dataset.xr_dataset[Key.PULSE_LENGTH_REALTIME].data

        return insonified_area_db(
            detection_range_sample=detection_range_sample,
            range_to_normal_incidence=range_to_normal_incidence,
            sampling_frequency=sampling_frequency,
            detection_beam_pointing_angle=detection_beam_pointing_angle,
            pulse_length_effective=pulse_length_realtime,
            sound_speed=sound_speed,
            along_beamwidth_deg=along_beamwidth_deg,
            across_beamwidth_deg=across_beamwidth_deg,
        )
