import numpy as np

from pyat.utils import signal


def lambert_correction(detection_range: np.ndarray, range_to_normal_incidence: np.ndarray) -> np.ndarray:
    """
    Compute lambert correction to remove from snippets for BS uncompensation
    Arguments must have the same dimension

    Args:
        detection_range: array of detection ranges (same unit as second argument)
        range_to_normal_incidence: array of normal incidence ranges

    Returns: ping-detection matrix containing Lambert correction to remove to uncompensate BS from snippets

    """
    lambert_corr = signal.amplitude_to_db(np.divide(detection_range, range_to_normal_incidence))
    return lambert_corr


def specular_correction(
    detection_range: np.ndarray,
    range_to_normal_incidence: np.ndarray,
    backscatter_normal_incidence_level: np.ndarray,
    backscatter_oblique_incidence_level: np.ndarray,
    tvg_law_crossover_angle: np.ndarray,
):
    """
    Compute specular correction to remove from snippets for BS uncompensation
    Arguments must have the same dimension

    Args:
        detection_range: array of detection ranges (same unit as second argument)
        range_to_normal_incidence: array of normal incidence ranges
        backscatter_normal_incidence_level: BSN in dB
        backscatter_oblique_incidence_level: BSO in dB
        tvg_law_crossover_angle: crossover angle in degrees

    Returns:

    """
    Rn = range_to_normal_incidence
    BSN = backscatter_normal_incidence_level
    BSO = backscatter_oblique_incidence_level
    deltaBS = BSN - BSO
    Tetac0 = np.radians(tvg_law_crossover_angle)  # with old xsf, crossover had a 10 factor error
    # CorrSpec.m
    rc0 = Rn / np.cos(Tetac0)
    # remove ranges smaller than normal range
    detection_range = np.maximum(detection_range, Rn)

    # find ranges between normal range and crossover range
    subrange_mask = detection_range < rc0
    # Rn_min = np.nanmin(detection_range[:].data, axis=1, initial=np.nan, where=subrange_mask)
    # ne fonctionne pas directement, il faut faire le min par antenna
    # pas sur non plus que cela soit utile... on prend directement le Rn pour le moment

    Rn_min = Rn
    Rn_diff = rc0 - Rn_min
    range_diff = detection_range - Rn_min
    X = np.sqrt(np.divide(range_diff, Rn_diff)) - 1
    specular_corr = np.multiply(X, deltaBS)
    # mask values out of range (angles greater than crossover)
    specular_corr[~subrange_mask] = 0.0
    return specular_corr


def insonified_area_db(
    # ping_detection variables
    detection_range_sample: np.ndarray,
    range_to_normal_incidence: np.ndarray,
    sampling_frequency: np.ndarray,
    detection_beam_pointing_angle: np.ndarray,
    pulse_length_effective: np.ndarray,
    # ping_time variables
    sound_speed: np.ndarray,
    along_beamwidth_deg: np.ndarray,
    across_beamwidth_deg: np.ndarray,
):
    """
    Compute insonified area as used by KM (dB)
    """
    # sonar_InsonifiedArea_KM.m

    # First compute beam opening angles. It depends on angle relative to transducer array as a physical effect of beamforming
    cos_angles = np.cos(np.deg2rad(detection_beam_pointing_angle))
    along_beam_opening_rad = np.deg2rad(along_beamwidth_deg[:, None])
    across_beam_opening_rad = np.deg2rad(across_beamwidth_deg[:, None] / cos_angles)

    # Estimate ranges
    range_meter = detection_range_sample * sound_speed[:, None] / (2 * sampling_frequency)
    rn_meter = range_to_normal_incidence * sound_speed[:, None] / (2 * sampling_frequency)

    # remove ranges smaller than normal range
    range_meter = np.maximum(range_meter, rn_meter)

    # Compute oblique area depending on pulse length
    AO = (sound_speed[:, None] * pulse_length_effective * along_beam_opening_rad * range_meter) / (
        2 * np.sqrt(1 - (rn_meter**2 / range_meter**2))
    )
    # Compute area around normal incidence depending on across beam opening
    AN = across_beam_opening_rad * along_beam_opening_rad * (range_meter**2)

    # Get area as the minimum of computed areas
    insonified_area_db = signal.energy_to_db(np.minimum(AO, AN))

    return insonified_area_db
