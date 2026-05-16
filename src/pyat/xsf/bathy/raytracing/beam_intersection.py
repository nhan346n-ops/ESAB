"""
Module providing functions to compute FCS beam pointing vectors using Tx/Rx beam intersection algorithms:
- Virtual Concentric Cone Algorithm (VCCA)
- Non Concentric Triangle Algorithm (NCTA)
- Non Concentric Cone Algorithm (NCCA)
"""

import numba
import numpy as np
from scipy.spatial.transform import Rotation

from pyat.xsf.bathy.raytracing.beam_footprint_location import (
    init_slant_ranges,
    intersect_tx_rx_beams_analytical_nb,
    intersect_tx_rx_beams_triangles,
)
from pyat.xsf.bathy.raytracing.raytracing import raytracing_by_depth, raytracing_by_depth_nb
from pyat.utils.hyperbolas_utils import intersect_tx_rx_beams_analytical
from pyat.xsf.xsf_driver import XsfDriver


def vcca(
    tx_steer: np.ndarray, rx_steer: np.ndarray, delta: np.ndarray, rot_arf2s: Rotation
) -> tuple[np.ndarray, np.ndarray]:
    """
    VCCA : Virtual Concentric Cone Algorithm
    Compute beam pointing vector in the ARF coordinate system (all angles in radians)
    Based on Beaudoin [2004] algorithm [https://journals.lib.unb.ca/index.php/ihr/article/view/20675/23837]
    params:
    tx_steer : Tx steering angle [radians]
    rx_steer : Rx steering angle [radians]
    delta : non-orthogonality angle between Tx and Rx arrays in XY plane [radians]
    rotARF2S : rotation matrix from ARF to SCS
    """
    try:
        # temporary disable underflow error as the may arise with not yet found reason
        with np.errstate(under="ignore"):
            y1 = -np.sin(rx_steer) / np.cos(delta)
            y2 = np.sin(tx_steer) * np.tan(delta)
    except FloatingPointError as e:
        print(f"FloatingPointError during vectorized computation: {e}")
    radial = np.sqrt((y1 + y2) ** 2 + np.sin(tx_steer) ** 2)

    bp_vector = np.column_stack([np.sin(tx_steer), y1 + y2, np.sqrt(1 - radial**2)])

    # Rotate beam-pointing vector in the ideal geographic coordinate system -> geographic launch vector
    bp_launch = rot_arf2s.apply(bp_vector)

    # Extract azimuth and depression angles to later perform ray-tracing
    bp_incidence = np.rad2deg(np.arctan2(np.sqrt(bp_launch[:, 0] ** 2 + bp_launch[:, 1] ** 2), bp_launch[:, 2]))
    bp_azimuth = np.rad2deg(np.arctan2(bp_launch[:, 1], bp_launch[:, 0]))

    return bp_incidence, bp_azimuth


def ncta(
    xsf: XsfDriver,
    initial_ranges,
    delta,
    tx_steer,
    rx_steer,
    array_separation,
    tx_fcs_position,
    rx_fcs_position,
    rot_arf2s,
    tol=1e-6,
    max_iter=10,
):
    """
    NCTA : Non Concentric Triangle Algorithm
    Compute beam pointing vector in the ARF coordinate system by intersecting projected triangles on focal plane
    Based on Bu [2020] algorithm
    """
    # Get surface sound velocity
    sv_at_tx = xsf.read_sound_speed_at_transducer()
    # Get sound velocity profiles index from XSF file for each detection
    ssp_idx = xsf.get_ssp_idx().repeat(xsf.sounder_file.beam_count)
    # get two-way travel time in seconds
    two_way_travel_time = xsf.read_detection_two_way_travel_time()

    one_way_travel_time_tx = np.full_like(two_way_travel_time.ravel(), np.nan)
    bp_incidence_tx = np.full_like(two_way_travel_time.ravel(), np.nan)
    bp_azimuth_tx = np.full_like(two_way_travel_time.ravel(), np.nan)

    for i, (r, d, tx, rx, a, tx_fcs_pos, rx_fcs_pos, twtt, sv_tx, ssp_i) in enumerate(
        zip(
            initial_ranges,
            delta,
            tx_steer,
            rx_steer,
            array_separation,
            tx_fcs_position,
            rx_fcs_position,
            two_way_travel_time.ravel(),
            sv_at_tx,
            ssp_idx,
        )
    ):
        if np.isnan(twtt):
            # if no detection where made for this beam let's
            continue
        if any(np.isnan(tx_fcs_pos)):
            # if missing tx_depth value
            continue

        twtt_error = 1
        iter_count = 0

        tl, rl = init_slant_ranges(r, a)

        # get sound velocity profile for current detection
        svp_depths, svp_values = xsf.read_sound_speed_profile(ssp_i)

        # iterate until twtt error is below tol or max_iter is reached
        while abs(twtt_error) > tol and iter_count < max_iter:
            launch_vector = intersect_tx_rx_beams_triangles(tl, rl, d, tx, rx, a)
            receive_vector = launch_vector - a

            # rotate launch and receive vectors in SCS
            launch_vector_scs = rot_arf2s[i].apply(launch_vector)
            receive_vector_scs = rot_arf2s[i].apply(receive_vector)

            # derive corresponding azimuth and incidence angles for both Tx and Rx
            azimuth_tx = np.rad2deg(np.arctan2(launch_vector_scs[1], launch_vector_scs[0]))
            incidence_tx = np.rad2deg(
                np.arctan2(np.sqrt(launch_vector_scs[0] ** 2 + launch_vector_scs[1] ** 2), launch_vector_scs[2])
            )
            incidence_rx = np.rad2deg(
                np.arctan2(np.sqrt(receive_vector_scs[0] ** 2 + receive_vector_scs[1] ** 2), receive_vector_scs[2])
            )
            # calculate TWTT by raytracing from both Tx and Rx to the seafloor
            owtt_tx, sv_bottom, _ = raytracing_by_depth(
                svp_depths,
                svp_values,
                tx_fcs_pos[-1] + launch_vector_scs[-1],
                incidence_tx,
                tx_fcs_pos[-1],
                sv_at_tx,
            )
            owtt_rx, _, _ = raytracing_by_depth(
                svp_depths, svp_values, rx_fcs_pos[-1] + receive_vector_scs[-1], incidence_rx, rx_fcs_pos[-1], sv_at_tx
            )

            # adjust focal range based on TWTT error
            twtt_error = twtt - (owtt_tx + owtt_rx)
            slant_range_increment = twtt_error / 2 * sv_bottom
            tl += slant_range_increment[0]
            rl += slant_range_increment[0]
            iter_count += 1

        # populate incidence and azimuth angles and owtt from Tx
        bp_incidence_tx[i] = incidence_tx
        bp_azimuth_tx[i] = azimuth_tx
        one_way_travel_time_tx[i] = owtt_tx

    return bp_incidence_tx, bp_azimuth_tx, one_way_travel_time_tx


def ncca(
    xsf: XsfDriver,
    initial_ranges,
    delta,
    tx_steer,
    rx_steer,
    array_separation,
    tx_fcs_position,
    rx_fcs_position,
    rot_arf2s,
    tol=1e-6,
    max_iter=10,
):
    """
    NCCA : Non Concentric Cone Algorithm
    Compute beam pointing vector in the ARF coordinate system by intersecting hyperbolas on focal plane
    Based on Hamilton [2014] algorithm
    [https://hydrography.ca/wp-content/uploads/files/2014conference/20-Hamilton-et-al-Algorithm-for-non-concentric-MB-array-geometry.pdf]
    """
    # Get surface sound velocity
    sv_at_tx = xsf.read_sound_speed_at_transducer()
    # Get sound velocity profiles index from XSF file for each detection
    ssp_idx = xsf.get_ssp_idx().repeat(xsf.sounder_file.beam_count)
    # get two-way travel time in seconds
    two_way_travel_time = xsf.read_detection_two_way_travel_time()

    # initialize output arrays
    one_way_travel_time_tx = np.full_like(two_way_travel_time.ravel(), np.nan)
    bp_incidence_tx = np.full_like(two_way_travel_time.ravel(), np.nan)
    bp_azimuth_tx = np.full_like(two_way_travel_time.ravel(), np.nan)

    # loop over detections
    for i, (r, d, tx, rx, a, tx_fcs_pos, rx_fcs_pos, twtt, sv_tx, ssp_i) in enumerate(
        zip(
            initial_ranges,
            delta,
            tx_steer,
            rx_steer,
            array_separation,
            tx_fcs_position,
            rx_fcs_position,
            two_way_travel_time.ravel(),
            sv_at_tx,
            ssp_idx,
        )
    ):
        if np.isnan(twtt):
            # if no detection where made for this beam let's
            continue
        if any(np.isnan(tx_fcs_pos)):
            # if missing tx_depth value
            continue

        twtt_error = 1
        iter_count = 0
        rng = r[-1].copy()

        # get sound velocity profile for current detection
        svp_depths, svp_values = xsf.read_sound_speed_profile(ssp_i)

        # iterate until twtt error is below tol or max_iter is reached
        while abs(twtt_error) > tol and iter_count < max_iter:

            intersection = intersect_tx_rx_beams_analytical(rng, d, tx, rx, a, plot=False, render="bokeh")
            if intersection is None:
                # TODO handle this case
                print("No intersection found")
                print(f"beam {i} {iter_count=}, {rng=}")
                break

            x, y = intersection

            launch_vector = np.asarray([x, y, rng])
            receive_vector = launch_vector - a

            # rotate launch and receive vectors in SCS
            launch_vector_scs = rot_arf2s[i].apply(launch_vector)
            receive_vector_scs = rot_arf2s[i].apply(receive_vector)

            # derive corresponding azimuth and incidence angles for both Tx and Rx
            azimuth_tx = np.rad2deg(np.arctan2(launch_vector_scs[1], launch_vector_scs[0]))
            incidence_tx = np.rad2deg(
                np.arctan2(np.sqrt(launch_vector_scs[0] ** 2 + launch_vector_scs[1] ** 2), launch_vector_scs[2])
            )
            incidence_rx = np.rad2deg(
                np.arctan2(np.sqrt(receive_vector_scs[0] ** 2 + receive_vector_scs[1] ** 2), receive_vector_scs[2])
            )
            # calculate TWTT by raytracing from both Tx and Rx to the seafloor
            owtt_tx, sv_bottom, bottom_incidence_tx = raytracing_by_depth(
                svp_depths,
                svp_values,
                tx_fcs_pos[-1] + launch_vector_scs[-1],
                incidence_tx,
                tx_fcs_pos[-1],
                sv_tx,
            )
            owtt_rx, _, _ = raytracing_by_depth(
                svp_depths, svp_values, rx_fcs_pos[-1] + receive_vector_scs[-1], incidence_rx, rx_fcs_pos[-1], sv_tx
            )

            # adjust focal range based on TWTT error
            twtt_error = twtt - (owtt_tx + owtt_rx)
            range_increment = twtt_error / 2 * sv_bottom * np.cos(np.deg2rad(bottom_incidence_tx))
            rng += range_increment[0]
            iter_count += 1

        # populate incidence and azimuth angles and owtt from Tx
        bp_incidence_tx[i] = incidence_tx
        bp_azimuth_tx[i] = azimuth_tx
        one_way_travel_time_tx[i] = owtt_tx

    return bp_incidence_tx, bp_azimuth_tx, one_way_travel_time_tx


@numba.njit(parallel=True)
def ncta_parallel(
    two_way_travel_time,
    svp_depths,
    svp_values,
    sv_tx,
    initial_ranges,
    delta,
    tx_steer,
    rx_steer,
    array_separation,
    tx_FCS_position,
    rx_FCS_position,
    rotARF2S,
    tol=1e-6,
    max_iter=10,
):
    """
    NCTA : Non Concentric Triangle Algorithm
    Compute beam pointing vector in the ARF coordinate system by intersecting projected triangles on focal plane
    Based on Bu [2020] algorithm
    """
    one_way_travel_time_tx = np.full_like(two_way_travel_time, np.nan)
    BP_incidence_tx = np.full_like(two_way_travel_time, np.nan)
    BP_azimuth_tx = np.full_like(two_way_travel_time, np.nan)

    for i in numba.prange(len(two_way_travel_time)):
        twtt = two_way_travel_time[i]
        r = initial_ranges[i]
        d = delta[i]
        tx = tx_steer[i]
        rx = rx_steer[i]
        a = array_separation[i]
        tx_fcs_pos = tx_FCS_position[i]
        rx_fcs_pos = rx_FCS_position[i]
        rot_arf2s = rotARF2S[i]
        sv = sv_tx[i]

        if np.isnan(twtt):
            # if no detection where made for this beam let's
            continue
        if np.any(np.isnan(tx_fcs_pos)):
            # if missing tx_depth value
            continue

        tl, rl = init_slant_ranges(r, a)

        twtt_error = 1
        iter_count = 0

        # iterate until twtt error is below tol or max_iter is reached
        while np.abs(twtt_error) > tol and iter_count < max_iter:
            launch_vector = intersect_tx_rx_beams_triangles(tl, rl, d, tx, rx, a)
            receive_vector = launch_vector - a

            # rotate launch and receive vectors in SCS
            launch_vector_scs = rot_arf2s @ launch_vector
            receive_vector_scs = rot_arf2s @ receive_vector

            # derive corresponding azimuth and incidence angles for both Tx and Rx
            azimuth_tx = np.rad2deg(np.arctan2(launch_vector_scs[1], launch_vector_scs[0]))
            incidence_tx = np.rad2deg(
                np.arctan2(np.sqrt(launch_vector_scs[0] ** 2 + launch_vector_scs[1] ** 2), launch_vector_scs[2])
            )

            incidence_rx = np.rad2deg(
                np.arctan2(np.sqrt(receive_vector_scs[0] ** 2 + receive_vector_scs[1] ** 2), receive_vector_scs[2])
            )
            # calculate TWTT by raytracing from both Tx and Rx to the seafloor
            owtt_tx, sv_bottom, _ = raytracing_by_depth_nb(
                svp_depths,
                svp_values,
                tx_fcs_pos[-1] + launch_vector_scs[-1],
                incidence_tx,
                tx_fcs_pos[-1],
                sv,
            )
            owtt_rx, _, _ = raytracing_by_depth_nb(
                svp_depths, svp_values, rx_fcs_pos[-1] + receive_vector_scs[-1], incidence_rx, rx_fcs_pos[-1], sv
            )

            # adjust focal range based on TWTT error
            twtt_error = twtt - (owtt_tx + owtt_rx)
            slant_range_increment = twtt_error / 2 * sv_bottom
            tl += slant_range_increment
            rl += slant_range_increment
            iter_count += 1

        # populate incidence and azimuth angles and owtt from Tx
        BP_incidence_tx[i] = incidence_tx
        BP_azimuth_tx[i] = azimuth_tx
        one_way_travel_time_tx[i] = owtt_tx

    return BP_incidence_tx, BP_azimuth_tx, one_way_travel_time_tx


@numba.njit(parallel=True)
def ncca_parallel(
    two_way_travel_time,
    svp_depths,
    svp_values,
    sv_tx,
    initial_ranges,
    delta,
    tx_steer,
    rx_steer,
    array_separation,
    tx_FCS_position,
    rx_FCS_position,
    rotARF2S,
    tol=1e-6,
    max_iter=10,
):
    """
    NCCA : Non Concentric Cone Algorithm
    Compute beam pointing vector in the ARF coordinate system by intersecting hyperbolas on focal plane
    Based on Hamilton [2014] algorithm
    [https://hydrography.ca/wp-content/uploads/files/2014conference/20-Hamilton-et-al-Algorithm-for-non-concentric-MB-array-geometry.pdf]
    """
    one_way_travel_time_tx = np.full_like(two_way_travel_time, np.nan)
    BP_incidence_tx = np.full_like(two_way_travel_time, np.nan)
    BP_azimuth_tx = np.full_like(two_way_travel_time, np.nan)

    # loop over beams
    for i in numba.prange(two_way_travel_time.size):
        twtt = two_way_travel_time[i]
        r = initial_ranges[i]
        d = delta[i]
        tx = tx_steer[i]
        rx = rx_steer[i]
        a = array_separation[i]
        tx_fcs_pos = tx_FCS_position[i]
        rx_fcs_pos = rx_FCS_position[i]
        rot_arf2s = rotARF2S[i]
        sv = sv_tx[i]

        if np.isnan(twtt):
            # if no detection where made for this beam let's
            continue
        if np.any(np.isnan(tx_fcs_pos)):
            # if missing tx_depth value
            continue

        twtt_error = 1
        iter_count = 0
        rng = r[-1]  # .copy()

        # iterate until twtt error is below tol or max_iter is reached
        while np.abs(twtt_error) > tol and iter_count < max_iter:

            intersection = intersect_tx_rx_beams_analytical_nb(rng, d, tx, rx, a)

            x, y = intersection

            launch_vector = np.asarray([x, y, rng])
            receive_vector = launch_vector - a

            # rotate launch and receive vectors in SCS
            launch_vector_scs = rot_arf2s @ launch_vector
            receive_vector_scs = rot_arf2s @ receive_vector

            # derive corresponding azimuth and incidence angles for both Tx and Rx
            azimuth_tx = np.rad2deg(np.arctan2(launch_vector_scs[1], launch_vector_scs[0]))
            incidence_tx = np.rad2deg(
                np.arctan2(np.sqrt(launch_vector_scs[0] ** 2 + launch_vector_scs[1] ** 2), launch_vector_scs[2])
            )

            incidence_rx = np.rad2deg(
                np.arctan2(np.sqrt(receive_vector_scs[0] ** 2 + receive_vector_scs[1] ** 2), receive_vector_scs[2])
            )
            # calculate TWTT by raytracing from both Tx and Rx to the seafloor
            owtt_tx, sv_bottom, bottom_incidence_tx = raytracing_by_depth_nb(
                svp_depths,
                svp_values,
                tx_fcs_pos[-1] + launch_vector_scs[-1],
                incidence_tx,
                tx_fcs_pos[-1],
                sv,
            )
            owtt_rx, _, _ = raytracing_by_depth_nb(
                svp_depths, svp_values, rx_fcs_pos[-1] + receive_vector_scs[-1], incidence_rx, rx_fcs_pos[-1], sv
            )

            # adjust focal range based on TWTT error
            twtt_error = twtt - (owtt_tx + owtt_rx)
            range_increment = twtt_error / 2 * sv_bottom * np.cos(np.deg2rad(bottom_incidence_tx))
            rng += range_increment
            iter_count += 1

        # populate incidence and azimuth angles and owtt from Tx
        BP_incidence_tx[i] = incidence_tx
        BP_azimuth_tx[i] = azimuth_tx
        one_way_travel_time_tx[i] = owtt_tx

    return BP_incidence_tx, BP_azimuth_tx, one_way_travel_time_tx
