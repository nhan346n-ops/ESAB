from typing import Tuple

import netCDF4 as nc
import numpy as np
import sonar_netcdf.sonar_groups as sg
from pygws.service.progress_monitor import ProgressMonitor
from scipy.spatial.transform import Rotation

from pyat.utils.coordinates_system_utils import (
    rot_scs_to_fcs,
    rot_vcs_to_fcs,
    rot_vcs_to_scs,
)
from pyat.utils.coords import create_lonlat_to_xy_converter
from pyat.utils.netcdf_utils import get_variable
from pyat.utils.numpy_utils import linear_interp_data
from pyat.utils.proj_utils import lon_lat_to_utm_proj4
from pyat.utils.time_utils import floatsecond_tonano_array
from pyat.xsf.bathy.raytracing.beam_intersection import (
    ncca,
    ncca_parallel,
    ncta,
    ncta_parallel,
    vcca,
)
from pyat.xsf.bathy.raytracing.raytracing import compute_detection_xyz
from pyat.xsf.xsf_driver import BEAM_GROUP_NAME as BM_GRP
from pyat.xsf.xsf_driver import XsfDriver

MINIMUM_XSF_VERSION = 0.7


def compute_beam_pointing_vector_in_scs(
    xsf: XsfDriver, algo="ncca_parallel"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes beam pointing vector in suface coordinate system (SCS), as a pair of incidence and azimuth beam pointing angles [degrees] from Tx SCS location for each detection.

    :param xsf: XsfDriver object
    :param algo: algorithm to use, options are: 'vcca', 'ncca', 'ncta', 'ncca_parallel', 'ncta_parallel'
    :return: tuple of (beam_pointing_incidence, beam_pointing_azimuth, one_way_travel_time, draft, tx_SCS_location)
    """
    # check xsf version
    xsf_version = xsf.get_version()
    if xsf_version < MINIMUM_XSF_VERSION:
        raise ValueError(
            f"The version must be at least {MINIMUM_XSF_VERSION} (currently {xsf_version}) for this processing\nTry updating the XSF file first."
        )

    global_monitor = ProgressMonitor()
    global_monitor.begin_task("Computing beam launch vectors", 4 if algo.split("_")[0] in ["ncca", "ncta"] else 2)

    # compute [ping beam] transmit and receive times
    transmit_time, receive_time = get_tx_rx_times(xsf.dataset)

    # Retrieve platform orientation angles at transmit times and receive times
    transmit_hdng, transmit_pitch, transmit_roll = get_platform_orientation_at_times(xsf, transmit_time)
    receive_hdng, receive_pitch, receive_roll = get_platform_orientation_at_times(xsf, receive_time)

    # get rotations from VCS to SCS at transmit time ...
    transmit_rot_v2s = rot_vcs_to_scs(pitches=transmit_pitch.ravel(), rolls=transmit_roll.ravel())
    # ... and receive time, considering transmit time SCS orientation as origin.
    # We use rot_vcs_to_fcs, with yaw from transmit time to receive time, in place of heading.
    transmit_to_receive_yaw = receive_hdng.ravel() - transmit_hdng.ravel()
    receive_rot_v2s = rot_vcs_to_fcs(
        headings=transmit_to_receive_yaw, pitches=receive_pitch.ravel(), rolls=receive_roll.ravel()
    )

    # compute platform vertical offset at transmit and receive times
    transmit_vertical_offset = get_platform_vertical_offset_at_times(xsf, transmit_time, transmit_rot_v2s)
    receive_vertical_offset = get_platform_vertical_offset_at_times(xsf, receive_time, receive_rot_v2s)

    # computes ARF coordinate system
    rot_arf2s, delta, tx_rot_a2s, rx_rot_a2s = get_arf_to_scs(xsf.dataset, transmit_rot_v2s, receive_rot_v2s)

    # Compute draft and Tx array offsets from platform origin at transmit time
    draft, tx_scs_location = compute_draft(
        xsf.dataset, transmit_rot_v2s, receive_rot_v2s, transmit_vertical_offset, receive_vertical_offset
    )

    # Get Tx and Rx steering angles
    tx_steer, rx_steer = get_steering_angles(xsf.dataset, tx_rot_a2s, rx_rot_a2s)  # rotARF2S.inv(), delta)

    global_monitor.worked(1)
    if algo == "vcca":
        # 1st algorithm Beaudoin [2004], co-located arrays (VCCA)
        # Compute beam pointing vector in the ARF coordinate system (all angles in radians)
        bp_incidence, bp_azimuth = vcca(tx_steer, rx_steer, delta, rot_arf2s)
        # Set draft to NaN when no high freq attitude value is available to ensure that there won't be no ray tracing for the corresponding beams
        draft[np.isnan(transmit_to_receive_yaw)] = np.nan
        # get two-way travel time in seconds
        two_way_travel_time = xsf.read_detection_two_way_travel_time()
        global_monitor.done()
        return bp_incidence, bp_azimuth, two_way_travel_time / 2, draft, tx_scs_location  # , tx_draft
    else:
        # 2nd algorithm Hamilton [2014], non co-located arrays (NCCA)
        # Computes Tx and Rx arrays 3d offsets in ARF
        # rotation from SCS to ARF
        rot_s2arf = rot_arf2s.inv()
        # rotation from SCS to FCS at transmit time
        rot_arf2f = rot_scs_to_fcs(headings=transmit_hdng.ravel()) * rot_arf2s

        array_separation, tx_fcs_position, rx_fcs_position = get_array_separation_at_times(
            xsf,
            transmit_time,
            receive_time,
            transmit_vertical_offset,
            receive_vertical_offset,
            rot_vcs_to_fcs(headings=transmit_hdng.ravel(), pitches=transmit_pitch.ravel(), rolls=transmit_roll.ravel()),
            rot_vcs_to_fcs(headings=receive_hdng.ravel(), pitches=receive_pitch.ravel(), rolls=receive_roll.ravel()),
            rot_arf2f.inv(),
        )
        global_monitor.worked(1)

        # initial focal range estimation with a simple assumption of Tx and Rx steering
        initial_ranges = compute_initial_focal_range(xsf, rot_arf2s, rot_s2arf, rx_steer, tx_fcs_position)
        global_monitor.worked(1)

        if algo == "ncca":
            # compute BP angles
            bp_incidence_tx, bp_azimuth_tx, owtt_tx = ncca(
                xsf,
                initial_ranges,
                delta,
                tx_steer,
                rx_steer,
                array_separation,
                tx_fcs_position,
                rx_fcs_position,
                rot_arf2s,
            )
        elif algo == "ncta":
            bp_incidence_tx, bp_azimuth_tx, owtt_tx = ncta(
                xsf,
                initial_ranges,
                delta,
                tx_steer,
                rx_steer,
                array_separation,
                tx_fcs_position,
                rx_fcs_position,
                rot_arf2s,
            )

        else:  # parrallelized versions
            # Get surface sound velocity at transmit_time
            sv_tx = get_sv_at_transmit_times(xsf, transmit_time)
            # get two-way travel time in seconds
            two_way_travel_time = xsf.read_detection_two_way_travel_time()
            # Get first sound velocity profile from XSF file
            svp_depths, svp_values = xsf.read_sound_speed_profile(0)
            # TODO handle multiple sound velocity profiles  using ssp_idx
            # Get sound velocity profiles index from XSF file for each detection
            # ssp_idx = get_ssp_idx(xsf).repeat(xsf.sounder_file.beam_count)

            if algo == "ncca_parallel":
                # Process in chunks to monitor progress
                sub_monitor = global_monitor.split(1)
                # compute chunck size to be 10% of total detections
                n_detections = len(two_way_travel_time.ravel())
                chunk_size = max(1, n_detections // 10)
                sub_monitor.begin_task(f"processing detections with {algo}", n_detections)
                bp_incidence_tx = np.full_like(two_way_travel_time.ravel(), np.nan)
                bp_azimuth_tx = np.full_like(two_way_travel_time.ravel(), np.nan)
                owtt_tx = np.full_like(two_way_travel_time.ravel(), np.nan)

                for i in range(0, n_detections, chunk_size):
                    chunk_end = min(i + chunk_size, n_detections)
                    chunk_slice = slice(i, chunk_end)
                    bp_incidence_chunk, bp_azimuth_chunk, owtt_chunk = ncca_parallel(
                        two_way_travel_time.ravel()[chunk_slice],
                        svp_depths,
                        svp_values,
                        sv_tx.ravel()[chunk_slice],
                        initial_ranges[chunk_slice],
                        delta[chunk_slice],
                        tx_steer[chunk_slice],
                        rx_steer[chunk_slice],
                        array_separation[chunk_slice],
                        tx_fcs_position[chunk_slice],
                        rx_fcs_position[chunk_slice],
                        rot_arf2s.as_matrix()[chunk_slice],
                    )
                    bp_incidence_tx[chunk_slice] = bp_incidence_chunk
                    bp_azimuth_tx[chunk_slice] = bp_azimuth_chunk
                    owtt_tx[chunk_slice] = owtt_chunk
                    sub_monitor.worked(chunk_slice.stop - chunk_slice.start)

                # reshape to original ping x beam shape
                bp_incidence_tx = bp_incidence_tx.reshape(two_way_travel_time.shape)
                bp_azimuth_tx = bp_azimuth_tx.reshape(two_way_travel_time.shape)
                owtt_tx = owtt_tx.reshape(two_way_travel_time.shape)

            elif algo == "ncta_parallel":
                # compute BP angles
                bp_incidence_tx, bp_azimuth_tx, owtt_tx = ncta_parallel(
                    two_way_travel_time.ravel(),
                    svp_depths,
                    svp_values,
                    sv_tx.ravel(),
                    initial_ranges,
                    delta,
                    tx_steer,
                    rx_steer,
                    array_separation,
                    tx_fcs_position,
                    rx_fcs_position,
                    rot_arf2s.as_matrix(),
                )
            else:
                raise ValueError(
                    f"Unknown algorithm: {algo}. Supported algorithms are 'vcca', 'ncca' 'ncta', 'ncca_parallel' and 'ncta_parallel'."
                )
        global_monitor.done()
        return (
            bp_incidence_tx,
            bp_azimuth_tx,
            owtt_tx.reshape(transmit_time.shape),
            draft,
            tx_scs_location,
        )


def get_arf_to_scs(
    xsf: nc.Dataset, transmit_rot_v2s: Rotation, receive_rot_v2s: Rotation
) -> Tuple[Rotation, np.ndarray]:
    """
    Generates the Array Reference Frame (ARF) coordinate system from the XSF dataset.
    Returns the rotation from ARF to SCS, and the non-orthogonality angle between Tx and Rx vector in ARF (delta).

    :param xsf: XsfDriver object
    :param transmit_rot_v2s: Rotation from VCS to SCS at transmit time
    :param receive_rot_v2s: Rotation from VCS to SCS at receive time
    :return: tuple of (rot_arf2s, delta)
    """
    # get rotations from Tx and Rx ACS to VCS
    tx_rot_a2v, rx_rot_a2v = get_vcs_array_installation_angles(xsf)
    # Builds array reference frame (ARF) coordinate system from the SCS at transmit time
    # Tx vector
    tx_rot_a2s = transmit_rot_v2s * tx_rot_a2v
    tx_vector = tx_rot_a2s.apply([1, 0, 0])
    # Rx vector
    rx_rot_a2s = receive_rot_v2s * rx_rot_a2v
    rx_vector = rx_rot_a2s.apply([0, 1, 0])
    # Builds array reference frame (ARF) coordinate system
    x_arf = tx_vector
    z_arf = np.cross(tx_vector, rx_vector)
    y_arf = np.cross(z_arf, x_arf)
    # Computes non-orthogonality angle between Tx and Rx vector
    delta = np.arccos(np.sum(tx_vector * rx_vector, axis=1)) - np.pi / 2
    # Initialized associated rotations
    rot_arf2s = Rotation.from_matrix([np.column_stack([x, y, z]) for x, y, z in zip(x_arf, y_arf, z_arf)])

    return rot_arf2s, delta, tx_rot_a2s, rx_rot_a2s


def compute_initial_focal_range(
    xsf: nc.Dataset, rot_arf2s: Rotation, rot_s2arf: Rotation, rx_steer: np.ndarray, tx_fcs_position: np.ndarray
) -> np.ndarray:
    """
    Returns initial focal range (in ARF) for each beam in meters.

    :param xsf: nc.Dataset object
    :param rot_arf2s: Rotation from ARF to SCS
    :param rot_s2arf: Rotation from SCS to ARF
    :param rx_steer: Rx steering angles in radians
    :param tx_fcs_position: Tx array position in FCS at transmit time
    """
    # compute initial beam pointing launch vector in SCS
    bp_launch_init = rot_arf2s.apply(np.column_stack([np.zeros_like(rx_steer), np.sin(-rx_steer), np.cos(rx_steer)]))
    bp_incidence = np.rad2deg(
        np.arctan2(np.sqrt(bp_launch_init[:, 0] ** 2 + bp_launch_init[:, 1] ** 2), bp_launch_init[:, 2])
    )
    bp_azimuth = np.rad2deg(np.arctan2(bp_launch_init[:, 1], bp_launch_init[:, 0]))

    # Perform raytracing using beam incidence angle
    detection_x, detection_y, detection_z = compute_detection_xyz(xsf, bp_incidence, bp_azimuth, tx_fcs_position[:, -1])

    # rotate back into ARF
    return rot_s2arf.apply(np.column_stack([detection_x.ravel(), detection_y.ravel(), detection_z.ravel()]))


def compute_draft(
    xsf: nc.Dataset,
    tx_rot_v2s: Rotation,
    rx_rot_v2s: Rotation,
    transmit_vertical_offset: np.ndarray,
    receive_vertical_offset: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns transducer draft and Tx array SCS location at transmit time.

    :param xsf: nc.Dataset object
    :param tx_rot_v2s: Rotation from VCS to SCS at transmit time
    :param rx_rot_v2s: Rotation from VCS to SCS at receive time
    :param transmit_vertical_offset: platform vertical offset at transmit time
    :param receive_vertical_offset: platform vertical offset at receive time
    :return: tuple of (draft, tx_scs_location)
    """
    tx_offsets, rx_offsets = get_vcs_array_installation_offsets(xsf)

    # Apply attitude to Tx and Rx offsets for each transmit and receive time to get real transducer SCS location
    tx_scs_location = tx_rot_v2s.apply(tx_offsets)
    rx_scs_location = rx_rot_v2s.apply(rx_offsets)

    # compute draft : negate platform vertical offset to get distance from the water line -> platform origin
    tx_draft = tx_scs_location[:, 2] - transmit_vertical_offset.ravel()  # /2
    rx_draft = rx_scs_location[:, 2] - receive_vertical_offset.ravel()  # /2

    return (tx_draft + rx_draft) / 2, tx_scs_location  # , tx_draft


def get_vcs_array_installation_angles(xsf: nc.Dataset) -> Tuple[Rotation, Rotation]:
    """
    Returns installation rotations (roll, pitch, heading) of Tx and Rx arrays in VCS for each detection.

    :param xsf: nc.Dataset object
    :return: tuple of (rot_a2v_tx, rot_a2v_rx)
    """
    detection_tx_transducer_index = xsf[sg.BathymetryGrp.DETECTION_TX_TRANSDUCER_INDEX(BM_GRP)][:]
    detection_rx_transducer_index = xsf[sg.BathymetryGrp.DETECTION_RX_TRANSDUCER_INDEX(BM_GRP)][:]

    detection_rx_transducer_index = np.array(detection_rx_transducer_index)  # remove mask for invalid data
    detection_tx_transducer_index = np.array(detection_tx_transducer_index)  # remove mask for invalid data
    # modify invalid values to point to first rx transducer, does not matter since all values will be flagged as invalid
    detection_rx_transducer_index[detection_rx_transducer_index < 0] = 0
    detection_tx_transducer_index[detection_tx_transducer_index < 0] = 0

    # retrieve installation angles for Tx and Rx
    x_installation_angle = np.asarray(xsf[sg.PlatformGrp.TRANSDUCER_ROTATION_X()])
    y_installation_angle = np.asarray(xsf[sg.PlatformGrp.TRANSDUCER_ROTATION_Y()])
    z_installation_angle = np.asarray(xsf[sg.PlatformGrp.TRANSDUCER_ROTATION_Z()])

    # Compute full angles rotations from ACS to VCS
    installation_rot = Rotation.from_euler(
        "ZYX", np.column_stack([z_installation_angle, y_installation_angle, x_installation_angle]), degrees=True
    )

    rot_a2v_tx = installation_rot[detection_tx_transducer_index.ravel()]
    rot_a2v_rx = installation_rot[detection_rx_transducer_index.ravel()]

    return rot_a2v_tx, rot_a2v_rx


def get_vcs_array_installation_offsets(xsf: nc.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns installation offsets (x, y, z) of Tx and Rx arrays in VCS for each detection.
    """
    detection_tx_transducer_index = xsf[sg.BathymetryGrp.DETECTION_TX_TRANSDUCER_INDEX(BM_GRP)][:]
    detection_rx_transducer_index = xsf[sg.BathymetryGrp.DETECTION_RX_TRANSDUCER_INDEX(BM_GRP)][:]

    detection_rx_transducer_index = np.array(detection_rx_transducer_index)  # remove mask for invalid data
    detection_tx_transducer_index = np.array(detection_tx_transducer_index)  # remove mask for invalid data
    # modify invalid values to point to first rx transducer, does not matter since all values will be flagged as invalid
    detection_rx_transducer_index[detection_rx_transducer_index < 0] = 0
    detection_tx_transducer_index[detection_tx_transducer_index < 0] = 0

    # retrieve installation offset for Tx and Rx
    rx_installation_offset = np.asarray(xsf[sg.PlatformGrp.TRANSDUCER_OFFSET_X()])
    ry_installation_offset = np.asarray(xsf[sg.PlatformGrp.TRANSDUCER_OFFSET_Y()])
    rz_installation_offset = np.asarray(xsf[sg.PlatformGrp.TRANSDUCER_OFFSET_Z()])
    installation_offsets = np.column_stack([rx_installation_offset, ry_installation_offset, rz_installation_offset])

    return (
        installation_offsets[detection_tx_transducer_index.ravel(), :],
        installation_offsets[detection_rx_transducer_index.ravel(), :],
    )


def get_platform_orientation_at_times(xsf: XsfDriver, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns platform orientation angles (heading, pitch, roll) from full resolution attitudes interpolated at given times.

    :param xsf: XsfDriver object
    :param times: times in ns since unix epoch
    :return: tuple of (heading, pitch, roll)
    """
    att_times = xsf.read_attitude_times()
    heading = xsf.read_attitude_headings()
    pitch = xsf.read_attitude_pitches()
    roll = xsf.read_attitude_rolls()
    # Unwrap heading before interpolation, and take the modulo afterward
    return (
        linear_interp_data(np.unwrap(heading, period=360), att_times, times) % 360,
        linear_interp_data(pitch, att_times, times),
        linear_interp_data(roll, att_times, times),
    )


def get_platform_position_at_times(xsf: XsfDriver, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns platform position (lon, lat) in FCS at given times for each detection.

    :param xsf: XsfDriver object
    :param times: times in ns since unix epoch
    :return: tuple of (lon, lat), each ping*beam size (flatten array)
    """
    pos_times = xsf.read_position_times().view("uint64")  # convert to ns since unix epoch
    lat = xsf.read_position_latitudes()
    lon = xsf.read_position_longitudes()

    return linear_interp_data(lon, pos_times, times), linear_interp_data(lat, pos_times, times)


def get_platform_vertical_offset_at_times(xsf: XsfDriver, times: np.ndarray, rot_v2s: Rotation) -> np.ndarray:
    """
    Returns platform vertical offset from high frequency vertical offset data (MRU heave or depth pressure sensor).
    It is distance from the platform reference point to the actual water line (positive downward).
    platform_vertical_offset = water_level + vertical_offset

    :param xsf: XsfDriver object
    :param times: times in ns since unix epoch
    :param rot_v2s: Rotation from VCS to SCS at given times
    """

    # Get water level offset : distance from origin of the platform coordinate system -> nominal water level
    platform_to_nominal_water_level = xsf[sg.PlatformGrp.WATER_LEVEL()][:]

    if xsf.get_preferred_depth_subgroup_id() is not None:
        # Get platform depth values if present (AUV/ROV) and is depth (type > 0) positive downward
        # support to Depth (pressure) sensor heave : DSH='IN' or 'NI'
        # IN = the heave of an underwater vehicle is presumed to be measured by the vehicle’s depth sensor
        # and the heave sensor input is not used by system.

        # Retrieve depth sensor installation offsets
        depth_sensor_installation_offsets = np.column_stack(xsf.read_depth_sensor_offset())
        # apply platform attitude rotations to depth sensor offsets
        depth_sensor_offset = rot_v2s.apply(depth_sensor_installation_offsets)
        # finaly get vertical component
        depth_sensor_vertical_offset = depth_sensor_offset[:, 2]

        vertical_offset = xsf.read_depth_sensor_vertical_offset()
        vertical_offset_time = xsf.read_depth_sensor_times()

        return (
            platform_to_nominal_water_level
            + linear_interp_data(vertical_offset, vertical_offset_time, times)
            + depth_sensor_vertical_offset.reshape(times.shape)
        )

    else:  # Get Heave (surface vessel)
        vertical_offset = xsf.read_attitude_vertical_offsets()
        vertical_offset_time = xsf.read_attitude_times()
        # heave and orientation are @ MRU location, must add induced heave @ platform location
        platform_to_mru_offsets = xsf.read_attitude_offset()
        platform_position_ref_mru = rot_v2s.apply(-platform_to_mru_offsets)
        induced_platform_heave = platform_position_ref_mru[:, 2]
        return (
            platform_to_nominal_water_level
            + linear_interp_data(vertical_offset, vertical_offset_time, times)
            - induced_platform_heave.reshape(times.shape)
        )


def get_array_separation_at_times(
    xsf: XsfDriver,
    tx_time: np.ndarray,
    rx_time: np.ndarray,
    tx_vertical_offset: np.ndarray,
    rx_vertical_offset: np.ndarray,
    tx_rot_v2f: Rotation,
    rx_rot_v2f: Rotation,
    rot_f2arf: Rotation,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns 3-D distance vector in ARF between Tx (@ transmit time) and Rx (@ receive time) arrays for each detection,
    and Tx and Rx array positions in FCS at transmit and receive times.

    :param xsf: XsfDriver object
    :param tx_time: transmit times in ns since unix epoch
    :param rx_time: receive times in ns since unix epoch
    :param tx_vertical_offset: platform vertical offset at transmit times
    :param rx_vertical_offset: platform vertical offset at receive times
    :param tx_rot_v2f: Rotation from VCS to FCS at transmit times
    :param rx_rot_v2f: Rotation from VCS to FCS at receive times
    :param rot_f2arf: Rotation from FCS to ARF
    :return: tuple of (rx_tx_arf_offset, tx_fcs_position, rx_fcs_position)
    """
    # Retrieve platform position at transmit and receive times
    lon_trans, lat_trans = get_platform_position_at_times(xsf, tx_time)
    lon_recep, lat_recep = get_platform_position_at_times(xsf, rx_time)

    # create a local UTM projection
    dd_to_xy = create_lonlat_to_xy_converter(proj=lon_lat_to_utm_proj4(np.nanmean(lon_trans), np.nanmean(lat_trans)))

    # Compute platform UTM positions at transmit and receive times,
    # swapping X and Y-axis to match FCS definition (x pointing north, y pointing east)
    y_trans, x_trans = dd_to_xy(lon_trans.ravel(), lat_trans.ravel())
    y_recep, x_recep = dd_to_xy(lon_recep.ravel(), lat_recep.ravel())

    # Combine these with negated platform vertical offset to get 3D platform FCS positions at transmit and receive times
    platform_fcs_position_trans = np.column_stack([x_trans, y_trans, -tx_vertical_offset.ravel()])
    platform_fcs_position_recep = np.column_stack([x_recep, y_recep, -rx_vertical_offset.ravel()])

    # Compute array position in the FCS, at transmit and receive times
    tx_vcs_offset, rx_vcs_offset = get_vcs_array_installation_offsets(xsf)
    tx_fcs_position = tx_rot_v2f.apply(tx_vcs_offset) + platform_fcs_position_trans
    rx_fcs_position = rx_rot_v2f.apply(rx_vcs_offset) + platform_fcs_position_recep

    # Computes 3D distances between Tx (transmit time) and Rx (receive time) for each detection
    rx_tx_fcs_offset = rx_fcs_position - tx_fcs_position

    # rotate Rx-Tx offset in  geographic reference frame back to ARF, and return Tx and Rx FCS positions
    return rot_f2arf.apply(rx_tx_fcs_offset), tx_fcs_position, rx_fcs_position


def get_sv_at_transmit_times(xsf: XsfDriver, times: np.ndarray) -> np.ndarray:
    """
    Returns sound velocity at transducer at given times (unix timestamp, i.e. number of nanoseconds since 1970-01-01T00:00:00Z ).

    :param xsf: XsfDriver object
    :param times: times in ns since unix epoch
    :return: sound velocity at transducer at given times
    """
    sv_times = xsf.read_ping_times().view("uint64")  # convert to ns since unix epoch
    sv_values = xsf.read_sound_speed_at_transducer()

    # cast to float32 to avoid type mimatch issues with numba when inserting sv in ssp
    return linear_interp_data(sv_values, sv_times, times).astype(np.float32)


def get_tx_rx_times(xsf: nc.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    returns actual receive and transmit  (aka Rx and Tx time) for each ping/beam.
    Tx : accounts for sector firing delays
    Rx : accounts for Tx sector firing delays + 2-way travel times
    Used to interpolate high frequency attitude data.

    :param xsf: nc.Dataset object
    :return: tuple of (tx_time, rx_time), as unix timestamp, i.e. number of nanoseconds since 1970-01-01T00:00:00Z
    """
    pingtime = xsf[sg.BeamGroup1Grp.PING_TIME(BM_GRP)][:]
    detection_2wtt = xsf[sg.BathymetryGrp.DETECTION_TWO_WAY_TRAVEL_TIME(BM_GRP)][:]  # seconds

    tx_beam_size = xsf[sg.BeamGroup1Grp.get_group_path(BM_GRP)].dimensions[sg.BeamGroup1Grp.TX_BEAM_DIM_NAME].size
    if tx_beam_size > 1:
        # if there are multiple tx beams, it means that there might be sector firing delays to take into account
        detection_tx_beam = xsf[sg.BathymetryGrp.DETECTION_TX_BEAM(BM_GRP)][:]
        tx_delay = xsf[sg.BeamGroup1VendorSpecificGrp.TRANSMIT_TIME_DELAY(BM_GRP)][:]  # seconds
    else:
        detection_tx_beam = np.zeros(detection_2wtt.shape, dtype=int)
        tx_delay = np.zeros_like(detection_2wtt)

    # create a ping index
    ping_number, detection_number = detection_2wtt.shape
    detection_ping_index = np.arange(ping_number)[:, None] * np.ones((1, detection_number), dtype=int)

    # remove mask for invalid data
    detection_tx_beam = np.array(detection_tx_beam)
    # modify invalid values to point to first tx beam, does not matter since all values will be latter flagged as invalid
    detection_tx_beam[detection_tx_beam < 0] = 0

    detection_delay = tx_delay[detection_ping_index, detection_tx_beam]
    # convert to nano seconds
    detection_delay = floatsecond_tonano_array(detection_delay)
    detection_2wtt = floatsecond_tonano_array(detection_2wtt)
    # add that to pingtime
    tx_time = pingtime.reshape(-1, 1) + detection_delay
    rx_time = tx_time + detection_2wtt

    return tx_time, rx_time


def get_steering_angles(xsf: nc.Dataset, tx_rot_a2s: Rotation, rx_rot_a2s: Rotation) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns rx and tx steering angles [radians] relative to ACS for each detection.

    :param xsf: nc.Dataset object
    :param tx_rot_a2s: Rotation object from ACS to SCS for transmit
    :param rx_rot_a2s: Rotation object from ACS to SCS for receive
    :return: tuple of (tx_steer, rx_steer) in radians
    """
    # Get Rx beam angle relative to the Rx transducer array if beam are not stabilized
    angle_ref_rx = xsf[sg.BathymetryGrp.DETECTION_BEAM_POINTING_ANGLE(BM_GRP)][:].astype(np.float64)
    # angle_ref_rx = xsf[sg.BeamGroup1Grp.RX_BEAM_ROTATION_PHI(BM_GRP)][:].astype(np.float64)
    ping_number, detection_number = angle_ref_rx.shape

    rx_steer = angle_ref_rx.ravel()
    # If stabilized (ME70 for example) angles are relative to the horizontal plane and need to be transformed to be relative to the array
    beam_stab_path = sg.BathymetryGrp.DETECTION_BEAM_STABILISATION(BM_GRP)
    beam_stab_var = get_variable(i_dataset=xsf, variable_path=beam_stab_path)
    if beam_stab_var is not None:
        beam_stab_mask = beam_stab_var[:] == 1
        # expand to the shape of angle_ref_rx
        beam_stab_mask = np.broadcast_to(beam_stab_mask[:, None], (ping_number, detection_number)).ravel()
        if np.any(beam_stab_mask):
            beam_scs = np.column_stack(
                [
                    np.zeros_like(rx_steer[beam_stab_mask]),
                    np.sin(np.deg2rad(rx_steer[beam_stab_mask])),
                    np.cos(np.deg2rad(rx_steer[beam_stab_mask])),
                ]
            )
            beam_acs = rx_rot_a2s[beam_stab_mask].inv().apply(beam_scs)
            rx_beam_steer = np.degrees(np.arctan2(beam_acs[:, 1], beam_acs[:, 2]))
            rx_steer = rx_beam_steer

    # Tx steering angle
    detection_ping_index = np.arange(ping_number)[:, None] * np.ones((1, detection_number), dtype=int)

    tx_beam_size = xsf[sg.BeamGroup1Grp.get_group_path(BM_GRP)].dimensions[sg.BeamGroup1Grp.TX_BEAM_DIM_NAME].size
    if tx_beam_size > 1:
        # if there are multiple tx beams, it means that there might be sector firing delays to take into account
        detection_tx_beam = xsf[sg.BathymetryGrp.DETECTION_TX_BEAM(BM_GRP)][:]
    else:
        detection_tx_beam = np.zeros(angle_ref_rx.shape, dtype=int)

    detection_tx_beam = np.array(detection_tx_beam)
    # modify invalid values to point to first tx beam, does not matter since all values will be latter flagged as invalid
    detection_tx_beam[detection_tx_beam < 0] = 0

    # Get Tx beam angle relative to the Tx transducer array if beam are not stabilized
    if (
        sg.BeamGroup1VendorSpecificGrp.RAW_TX_BEAM_TILT_ANGLE_VNAME
        in xsf[sg.BeamGroup1VendorSpecificGrp.get_group_path(BM_GRP)].variables
    ):
        angle_ref_tx = xsf[sg.BeamGroup1VendorSpecificGrp.RAW_TX_BEAM_TILT_ANGLE(BM_GRP)][:]
        angle_ref_tx = angle_ref_tx[detection_ping_index, detection_tx_beam].astype(np.float64)
        tx_steer = angle_ref_tx.ravel()
    else:
        # use tx_beam_rotation_theta that is relative to surface if beam_stabilisation is active, so we need to transform it later
        angle_ref_tx = xsf[sg.BeamGroup1Grp.TX_BEAM_ROTATION_THETA(BM_GRP)][:]
        angle_ref_tx = angle_ref_tx[detection_ping_index, detection_tx_beam].astype(np.float64)
        tx_steer = angle_ref_tx.ravel()

        beam_stab_path = sg.BeamGroup1Grp.BEAM_STABILISATION(BM_GRP)
        beam_stab_var = get_variable(i_dataset=xsf, variable_path=beam_stab_path)
        if beam_stab_var is not None:
            beam_stab_mask = beam_stab_var[:] == 1
            # expand to the shape of angle_ref_rx
            beam_stab_mask = np.broadcast_to(beam_stab_mask[:, None], (ping_number, detection_number)).ravel()
            if np.any(beam_stab_mask):
                # beams are stabilized, they are relative to SCS (an not platform as it mentionned in the helper)
                # need to rotate angles in SCS -> ACS uising tx_rot_a2s
                tx_steer_scs = Rotation.from_euler("Y", tx_steer[beam_stab_mask], degrees=True)
                tx_steer_acs = tx_rot_a2s[beam_stab_mask].inv() * tx_steer_scs
                tx_steer = tx_steer_acs.as_euler("ZYX", degrees=True)[:, 1]

    return np.deg2rad(tx_steer), np.deg2rad(rx_steer)
