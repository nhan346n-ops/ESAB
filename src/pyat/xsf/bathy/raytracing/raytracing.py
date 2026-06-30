"""
Module providing :
- acoustic raytracing functions to compute refracted beam paths.
- functions to compute soundings position from raytracing results and platform attitude.
"""

import netCDF4 as nc
import numba
import numpy as np
import sonar_netcdf.sonar_groups as sg

from pyat.sonarscope.model.constants import DEFAULT_BEAM_GROUP_IDENT as DFLT_BM_GRP
from pyat.utils.coords import compute_detection_position
from pyat.xsf.xsf_driver import XsfDriver


def prepare_svp_for_raytracing(
    svp_depths: np.ndarray[np.float64],
    svp_values: np.ndarray[np.float64],
    tx_depth: np.ndarray[np.float64],
    sv_tx: np.ndarray[np.float64],
    n_beam: int,
    sv_precision: float = 1e-3,
) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[int]]:
    """
    Prepares SVP for raytracing by inserting transducer depth and measuredsound speed at that depth.
    Returns 2D arrays of SVP depths and values (size = (n_layers, n_beam)), as well as SVP vertical gradients and transducer depth indices in that array.

    :param svp_depths: 1D array of sound velocity profile depths (m)
    :type svp_depths: np.ndarray[np.float64]
    :param svp_values: 1D array of sound velocity profile values (m/s)
    :type svp_values: np.ndarray[np.float64]
    :param tx_depth: 1D array of transducer depths (m)
    :type tx_depth: np.ndarray[np.float64]
    :param sv_tx: 1D array of transducer sound speeds (m/s)
    :type sv_tx: np.ndarray[np.float64]
    :param n_beam: number of beams
    :type n_beam: int
    :return: tuple of 2D arrays (svp_depths (m), svp_values (m/s)), each of size = (n_layers, n_beam)
    :rtype: tuple[ndarray[float64, Any], ndarray[float64, Any]]
    """

    # create 2D svp arrays, enforcing SSP data copy to avoid leaking in-place modifications
    svp_z = np.outer(svp_depths.copy(), np.full(shape=(n_beam,), fill_value=1.0))  # np.ones_like(beam_incidence_angle))
    svp_v = np.outer(svp_values.copy(), np.full(shape=(n_beam,), fill_value=1.0))

    # find above draft sound velocity index
    i_tx_depth = np.fromiter(
        (np.maximum(np.searchsorted(svp_z[:, i], tx_depth[i]) - 1, 0) for i in range(n_beam)),
        dtype=int,
        count=n_beam,
    )

    # Create mask for valid sv_tx values (nan or <=0)
    sv_tx_invalid = np.logical_or(sv_tx <= 0, np.isnan(sv_tx))

    # Vectorized interpolation for missing sv_tx values
    sv_interpolated = np.full_like(sv_tx, fill_value=np.nan)
    for i in np.nonzero(sv_tx_invalid)[0]:
        sv_interpolated[i] = np.interp(tx_depth[i], svp_z[:, i], svp_v[:, i])

    # Use provided sv_tx where valid, interpolated values where not
    sv_to_insert = np.where(sv_tx_invalid, sv_interpolated, sv_tx)

    # Vectorized insertion of sound speed and depth at transducer level
    svp_v[i_tx_depth, np.arange(n_beam)] = sv_to_insert
    svp_z[i_tx_depth, np.arange(n_beam)] = tx_depth

    # # remove possible duplicated values in svp_z, due to insertion above
    # # if that occurs, offset values by corresponding machine precision
    # z_diff = np.diff(svp_z, axis=0)
    # same_depth_layers = np.abs(z_diff) < np.spacing(svp_z)[:-1]
    # while np.any(same_depth_layers):
    #     # add machine precision to next layer z values...
    #     same_depth_layers = np.vstack([np.zeros((1, n_beam), dtype=bool), same_depth_layers])
    #     # Use the lowest of the data type's machine precision involve in computation i.e. one_way_travel_time's float32
    #     svp_z[same_depth_layers] += np.spacing(svp_z[same_depth_layers].astype(one_way_travel_time.dtype))
    #     # ... and continue while it's not ok
    #     z_diff = np.diff(svp_z, axis=0)
    #     same_depth_layers = np.abs(z_diff) < np.spacing(svp_z)[:-1]

    # Don't allow for zero gradients (i.e. strait line propagation)
    # if that occurs, offset values by corresponding machine precision
    svp_diff = np.diff(svp_v, axis=0)
    zero_gradient_layers = np.abs(svp_diff) < sv_precision  # np.spacing(svp_v)[:-1]
    while np.any(zero_gradient_layers):
        # add machine precision to next layer SVP values...
        zero_gradient_layers = np.vstack([np.zeros((1, n_beam), dtype=bool), zero_gradient_layers])
        # Use the lowest of the data type's machine precision involve in computation i.e. one_way_travel_time's float32
        svp_v[zero_gradient_layers] += sv_precision  # np.spacing(svp_v[zero_gradient_layers])
        # ... and continue while it's not ok
        svp_diff = np.diff(svp_v, axis=0)
        zero_gradient_layers = np.abs(svp_diff) < sv_precision  # np.spacing(svp_v)[:-1]

    return svp_z, svp_v, svp_diff, i_tx_depth


def raytracing_by_time(
    svp_depths: np.ndarray[np.float64],
    svp_values: np.ndarray[np.float64],
    one_way_travel_time: np.ndarray[np.float64],
    beam_incidence_angle: np.ndarray[np.float64],
    tx_depth: np.ndarray[np.float64],
    sv_tx: np.ndarray[np.float64],
) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64]]:
    """
    Computes acoustic beams refracted path for one ping (n_beam), as a function of time, using snell-descartes law.
    Sound speed profile is assumed to be piece-wise linear between provided depth levels, allowing only circular ray paths in each layer.
    Notable exception is for null ray constant (vertical beams) where straight line propagation is allowed.
    Only downward propagation is allowed. Function is vectorized over beams.
    Returns soundings vertical and horizontal  distance (from beam launch vector origin), and bottom incidence angle (°).

    :param svp_depths: 1D array of sound velocity profile depths (m)
    :param svp_values: 1D array of sound velocity profile values (m/s)
    :param one_way_travel_time: 1D array of one way travel times (s), size = n_beam
    :param beam_incidence_angle: 1D array of beam incidence angles for each beam (°), size = n_beam
    :param tx_depth: 1D array of transducer depths (m), size = n_beam | 1
    :param sv_tx: 1D array of sound velocity measured at transducer depth (m/s), size = n_beam | 1
    :return: tuple of 3 1D arrays (depths (m), horizontal distances (m), bottom incidence angles (°)), each of size = n_beam
    """
    # size checks, ensure all passed arrays are 1D of compatible sizes
    beam_incidence_angle = np.atleast_1d(beam_incidence_angle)
    one_way_travel_time = np.atleast_1d(one_way_travel_time)
    tx_depth = np.atleast_1d(tx_depth)
    sv_tx = np.atleast_1d(sv_tx)
    n_beam = len(beam_incidence_angle)
    if len(one_way_travel_time) != n_beam:
        raise ValueError("one_way_travel_time size does not match beam_incidence_angle size")
    if len(tx_depth) not in (1, n_beam):
        raise ValueError("tx_depth size must be 1 or match beam_incidence_angle size")
    if len(sv_tx) not in (1, n_beam):
        raise ValueError("sv_tx size must be 1 or match beam_incidence_angle size")
    # size adjustments
    if len(tx_depth) == 1:
        tx_depth = np.broadcast_to(tx_depth, (n_beam,))
    if len(sv_tx) == 1:
        sv_tx = np.broadcast_to(sv_tx, (n_beam,))

    # convert to radians
    beam_incidence_angle = np.deg2rad(beam_incidence_angle)

    # Get (n_layers, n_beam) arrays of svp depths and values with inserted transducer depth and sound speed
    svp_z, svp_v, svp_diff, i_tx_depth = prepare_svp_for_raytracing(svp_depths, svp_values, tx_depth, sv_tx, n_beam)

    # SVP gradient
    svp_grad = svp_diff / np.diff(svp_z, axis=0)

    # Ray constants (Snell-Descartes law)
    ray_cst = np.sin(beam_incidence_angle) / svp_v[i_tx_depth, np.arange(n_beam)]

    # Incidence angles at layer boundaries
    sin_incidence_angles = svp_v * ray_cst
    # Prevent arcsined values outside [-1, 1], corresponding to rays reaching horizontal, thus trapped in the layer
    incidence_angles = np.arcsin(
        np.where((sin_incidence_angles >= -1) & (sin_incidence_angles <= 1), sin_incidence_angles, np.nan)
    )

    # Layers flight-through times
    # t = np.where(ray_cst == 0, np.log(svp_v), np.log(np.tan(incidence_angles / 2)))
    t = np.log(svp_v)  # Pre-allocate t array filled with the fallback value
    np.log(
        np.tan(incidence_angles / 2), out=t, where=(ray_cst != 0)
    )  # Compute log(tan) ONLY where the mask is True, writing directly into 't'

    dt = np.diff(t, axis=0) / svp_grad
    # set all above tx_depth layers flight time to 0 : only propagate downward
    layers = np.arange(dt.shape[0])[:, None]
    dt[layers < i_tx_depth] = 0
    t_total = np.vstack([np.zeros(n_beam), np.cumsum(dt, axis=0)])

    # Last layer index
    last_layer = np.sum(t_total < one_way_travel_time, axis=0) - 1

    # Last layer flight-through times
    dt_last = one_way_travel_time - t_total[last_layer, np.arange(n_beam)]

    # Last layer sound velocity gradient
    grad_last = svp_grad[last_layer, np.arange(n_beam)]

    # Bottom incidence angle
    bottom_incidence_angle = 2 * (
        np.arctan(np.tan(incidence_angles[last_layer, np.arange(n_beam)] / 2) * np.exp(grad_last * dt_last))
    )
    # Bottom sound speed
    sv_last = svp_v[last_layer, np.arange(n_beam)] * np.exp(
        grad_last * dt_last
    )  # Pre-allocate 'sv_last' array filled with the fallback value
    np.divide(
        np.sin(bottom_incidence_angle), ray_cst, out=sv_last, where=(ray_cst != 0)
    )  # divide only where ray_cst is not zero, writing directly into 'sv_last'

    # Finaly, compute depth below tx_depth...
    dz = (sv_last - svp_v[last_layer, np.arange(n_beam)]) / grad_last
    depth = svp_z[last_layer, np.arange(n_beam)] + dz - tx_depth

    # ... and horizontal distance from Tx array
    svp_grad = np.vstack([svp_grad, svp_grad[-1, np.newaxis]])
    radius = np.zeros_like(incidence_angles, dtype=float)  # Pre-allocate 'radius' array filled with the fallback value
    np.divide(
        -svp_v / svp_grad, np.sin(incidence_angles), out=radius, where=(ray_cst != 0)
    )  # divide only where ray_cst is not zero, writing directly into 'radius'

    hrz_dist = radius[:-1, :] * np.diff(np.cos(incidence_angles), axis=0)
    # set all above tx_depth layers horizontal distance to 0 : only propagate downward
    hrz_dist[layers < i_tx_depth] = 0
    hrz_dist_total = np.vstack([np.zeros(n_beam), np.cumsum(hrz_dist, axis=0)])

    horizontal_distance = np.zeros_like(
        depth, dtype=float
    )  # Pre-allocate 'horizontal_distance' array filled with the fallback value
    np.divide(
        hrz_dist_total[last_layer, np.arange(n_beam)]
        - (np.cos(bottom_incidence_angle) - np.cos(incidence_angles[last_layer, np.arange(n_beam)]))
        * svp_v[last_layer, np.arange(n_beam)]
        / grad_last,
        np.sin(incidence_angles[last_layer, np.arange(n_beam)]),
        out=horizontal_distance,
        where=(ray_cst != 0),
    )  # divide only where ray_cst is not zero, writing directly into 'horizontal_distance'

    return depth, horizontal_distance, np.rad2deg(bottom_incidence_angle)


def raytracing_by_depth(
    svp_depths: np.ndarray[np.float64],
    svp_values: np.ndarray[np.float64],
    depth: np.float64,
    beam_incidence_angle: np.float64,
    tx_depth: np.float64,
    sv_tx: np.float64,
) -> tuple[np.float64, np.float64, np.float64]:
    """
    Computes acoustic ray path one way travel time, for one beam, as a function of depth, using snell-descartes law.
    Sound speed profile is assumed to be piece-wise linear between provided depth levels, allowing only circular ray paths in each layer.
    Notable exception is for null ray constant (vertical beams) where straight line propagation is allowed.
    Only downward propagation is allowed.
    returns sounding's one way travel time (s), bottom sound speed (m/s)and incidence angle (°).
    TODO: this function could be vectorized over beams like raytracing_by_time(), and used in a MBES simulator

    :param svp_depths: 1D array of sound velocity profile depths (m), size = n_layers
    :type svp_depths: np.ndarray[np.float64]
    :param svp_values: 1D array of sound velocity values (m/s), size = n_layers
    :type svp_values: np.ndarray[np.float64]
    :param depth: 1D array of sounding depths (m)
    :type depth: np.float64
    :param beam_incidence_angle: 1D array of beam incidence angles for each beam (°)
    :type beam_incidence_angle: np.float64
    :param tx_depth: Transducer depth (m)
    :type tx_depth: np.float64
    :param sv_tx: Sound speed at transducer depth (m/s)
    :type sv_tx: np.float64
    :return: tuple of 3 1D arrays (one way travel time (s), bottom sound speed (m/s), bottom incidence angle (°))
    :rtype: tuple[np.float64, np.float64, np.float64]
    """
    beam_incidence_angle_rad = np.deg2rad(beam_incidence_angle)

    # enforce SSP data copy to avoid leaking in-place modifications
    svp_z = svp_depths.copy()
    svp_v = svp_values.copy()

    # find above draft sound velocity index
    ix_tx_depth = np.maximum(np.searchsorted(svp_z, tx_depth) - 1, 0)

    if not sv_tx and ix_tx_depth > 0:
        # interpolate ssp at transducer depth
        svp_v[ix_tx_depth] = np.interp(tx_depth, svp_z, svp_v)
    else:
        # insert sound speed value measured at transducer...
        svp_v[ix_tx_depth] = sv_tx
    # ...at transducer depth
    svp_z[ix_tx_depth] = tx_depth

    # limit sound velocity profile to z[draft : +inf]
    svp_z = svp_z[ix_tx_depth:]
    svp_v = svp_v[ix_tx_depth:]

    # remove possible duplicated values in svp_z, due to insertion above, and corresponding svp_v if any
    repeated_values_ix = np.nonzero(np.diff(svp_z) == 0)[0] + 1
    svp_z = np.delete(svp_z, repeated_values_ix)
    svp_v = np.delete(svp_v, repeated_values_ix)

    # Don't allow for zero gradients (i.e. strait line acoustic propagation)
    # if that occurs, offset values by corresponding machine precision
    svp_diff = np.diff(svp_v)
    zero_gradient_layers = np.abs(svp_diff) < np.spacing(svp_v)[:-1]
    while np.any(zero_gradient_layers):
        # add machine precision to SVP values...
        zero_gradient_layers = np.hstack([False, zero_gradient_layers])
        svp_v[zero_gradient_layers] += np.spacing(svp_v[zero_gradient_layers].astype(np.float32))
        # ... and continue while it's not ok
        svp_diff = np.diff(svp_v)
        zero_gradient_layers = np.abs(svp_diff) < np.spacing(svp_v)[:-1]

    # SVP gradient
    svp_grad = svp_diff / np.diff(svp_z)

    # Ray constants
    ray_cst = np.sin(beam_incidence_angle_rad) / svp_v[0]

    # Incidence angles at layer boundaries
    sin_incidence_angles = svp_v * ray_cst
    # Prevent arcsined values outside [-1, 1], corresponding to rays reaching horizontal, thus trapped in the layer
    incidence_angles = np.arcsin(
        np.where((sin_incidence_angles >= -1) & (sin_incidence_angles <= 1), sin_incidence_angles, np.nan)
    )

    # Layers flight-through times
    if ray_cst == 0:
        t = np.log(svp_v)
    else:
        t = np.log(np.tan(incidence_angles / 2))
    dt = np.diff(t, axis=0) / svp_grad
    t_total = np.concatenate((np.zeros((1,)), np.cumsum(dt)))

    # Last layer index according to depth
    last_layer = np.sum(svp_z < depth, axis=0) - 1
    dz = depth - svp_z[last_layer]

    # Last layer sound velocity gradient
    svp_grad = np.append(svp_grad, svp_grad[-1])
    grad_last = svp_grad[last_layer]

    # horizontal distance
    # radius = -(svp_v[:, np.newaxis] / svp_grad[:, np.newaxis]) / np.sin(incidence_angles)
    # hrz_dist = radius[:-1, :] * np.diff(np.cos(incidence_angles), axis=0)
    # hrz_dist_total = np.vstack([np.zeros(n_beam), np.cumsum(hrz_dist, axis=0)])

    # Bottom sound speed
    sv_last = svp_v[last_layer] + dz * grad_last

    # Bottom incidence angle
    bottom_incidence_angle = np.arcsin(sv_last * ray_cst)

    # Last layer flight-through times
    if ray_cst == 0:
        dt_last = np.log(sv_last / svp_v[last_layer]) / grad_last
    else:
        dt_last = np.log(np.tan(bottom_incidence_angle / 2) / np.tan(incidence_angles[last_layer] / 2)) / grad_last
    # Finaly, compute total flight time...
    on_way_travel_time = t_total[last_layer] + dt_last

    # # ... and horizontal distance from reference frame origin
    # horizontal_distance = hrz_dist_total[last_layer, np.arange(n_beam)] - (
    #     np.cos(bottom_incidence_angle) - np.cos(incidence_angles[last_layer, np.arange(n_beam)])
    # ) * svp_v[last_layer] / grad_last / np.sin(incidence_angles[last_layer, np.arange(n_beam)])

    return on_way_travel_time, sv_last, np.rad2deg(bottom_incidence_angle)  # horizontal_distance


@numba.njit(parallel=True)
def raytracing_by_depth_nb(
    svp_depths: np.ndarray[np.float64],
    svp_values: np.ndarray[np.float64],
    depth: np.float64,
    beam_incidence_angle: np.float64,
    tx_depth: np.float64,
    sv_tx: np.float64,
) -> tuple[np.float64, np.float64, np.float64]:
    """
    Computes acoustic ray path one way travel time, for one beam, as a function of depth, using snell-descartes law.
    Sound speed profile is assumed to be piece-wise linear between provided depth levels, allowing only circular ray paths in each layer.
    Notable exception is for null ray constant (vertical beams) where straight line propagation is allowed.
    Only downward propagation is allowed. Results are relative to Tx.
    returns sounding's one way travel time (s), bottom sound speed (m/s)and incidence angle (°).
    Numba njit-compiled version of raytracing_by_depth().

    :param svp_depths: 1D array of sound velocity profile depths (m), size = n_layers
    :type svp_depths: np.ndarray[np.float64]
    :param svp_values: 1D array of sound velocity values (m/s), size = n_layers
    :type svp_values: np.ndarray[np.float64]
    :param depth: 1D array of sounding depths (m)
    :type depth: np.float64
    :param beam_incidence_angle: 1D array of beam incidence angles for each beam (°)
    :type beam_incidence_angle: np.float64
    :param tx_depth: Transducer depth (m)
    :type tx_depth: np.float64
    :param sv_tx: Sound speed at transducer depth (m/s)
    :type sv_tx: np.float64
    :return: tuple of 3 1D arrays (one way travel time (s), bottom sound speed (m/s), bottom incidence angle (°))
    :rtype: tuple[np.float64, np.float64, np.float64]
    """
    beam_incidence_angle_rad = np.deg2rad(beam_incidence_angle)

    # enforce SSP data copy to avoid leaking in-place modifications
    svp_z = svp_depths.copy()
    svp_v = svp_values.copy()

    # find above draft sound velocity index
    ix_tx_depth = np.maximum(np.searchsorted(svp_z, tx_depth) - 1, 0)

    if not sv_tx and ix_tx_depth > 0:
        # interpolate ssp at transducer depth
        svp_v[ix_tx_depth] = np.interp(tx_depth, svp_z, svp_v)
    else:
        # insert sound speed value measured at transducer...
        svp_v[ix_tx_depth] = sv_tx
    # ...at transducer depth
    svp_z[ix_tx_depth] = tx_depth

    # limit sound velocity profile to z[draft : +inf]
    svp_z = svp_z[ix_tx_depth:]
    svp_v = svp_v[ix_tx_depth:]

    # remove possible duplicated values in svp_z, due to insertion above, and corresponding svp_v if any
    repeated_values_ix = np.nonzero(np.diff(svp_z) == 0)[0] + 1
    svp_z = np.delete(svp_z, repeated_values_ix)
    svp_v = np.delete(svp_v, repeated_values_ix)

    # Don't allow for zero gradients (i.e. strait line acoustic propagation)
    # if that occurs, offset values by corresponding machine precision
    svp_diff = np.diff(svp_v)
    zero_gradient_layers = np.abs(svp_diff) < np.spacing(svp_v)[:-1]
    while np.any(zero_gradient_layers):
        # add machine precision to SVP values...
        # zero_gradient_layers = np.hstack([False, zero_gradient_layers])
        zero_gradient_layers = np.concatenate((np.full((1,), False), zero_gradient_layers))
        svp_v[zero_gradient_layers] += np.spacing(svp_v[zero_gradient_layers].astype(np.float32))
        # ... and continue while it's not ok
        svp_diff = np.diff(svp_v)
        zero_gradient_layers = np.abs(svp_diff) < np.spacing(svp_v)[:-1]

    # SVP gradient
    svp_grad = svp_diff / np.diff(svp_z)

    # Ray constants
    ray_cst = np.sin(beam_incidence_angle_rad) / svp_v[0]

    # Incidence angles at layer boundaries
    sin_incidence_angles = svp_v * ray_cst
    # Prevent arcsined values outside [-1, 1], corresponding to rays reaching horizontal, thus trapped in the layer
    incidence_angles = np.arcsin(
        np.where((sin_incidence_angles >= -1) & (sin_incidence_angles <= 1), sin_incidence_angles, np.nan)
    )

    # Layers flight-through times
    if ray_cst == 0:
        t = np.log(svp_v)
    else:
        t = np.log(np.tan(incidence_angles / 2))
    dt = np.diff(t) / svp_grad
    t_total = np.concatenate((np.zeros((1,)), np.cumsum(dt)))

    # Last layer index according to depth
    last_layer = np.sum(svp_z < depth) - 1
    dz = depth - svp_z[last_layer]

    # Last layer sound velocity gradient
    svp_grad = np.append(svp_grad, svp_grad[-1])
    grad_last = svp_grad[last_layer]

    # horizontal distance (optional, not needed for now)
    # radius = -(svp_v[:, np.newaxis] / svp_grad[:, np.newaxis]) / np.sin(incidence_angles)
    # hrz_dist = radius[:-1, :] * np.diff(np.cos(incidence_angles), axis=0)
    # hrz_dist_total = np.vstack([np.zeros(n_beam), np.cumsum(hrz_dist, axis=0)])

    # Bottom sound speed
    sv_last = svp_v[last_layer] + dz * grad_last

    # Bottom incidence angle
    bottom_incidence_angle = np.arcsin(sv_last * ray_cst)

    # Last layer flight-through times
    if ray_cst == 0:
        dt_last = np.log(sv_last / svp_v[last_layer]) / grad_last
    else:
        dt_last = np.log(np.tan(bottom_incidence_angle / 2) / np.tan(incidence_angles[last_layer] / 2)) / grad_last

    # Finaly, compute total flight time...
    on_way_travel_time = t_total[last_layer] + dt_last

    # # ... and horizontal distance from reference frame origin (optional, not needed for now)
    # horizontal_distance = hrz_dist_total[last_layer, np.arange(n_beam)] - (
    #     np.cos(bottom_incidence_angle) - np.cos(incidence_angles[last_layer, np.arange(n_beam)])
    # ) * svp_v[last_layer] / grad_last / np.sin(incidence_angles[last_layer, np.arange(n_beam)])

    return on_way_travel_time, sv_last, np.rad2deg(bottom_incidence_angle)  # horizontal_distance


def compute_detection_xyz(
    xsf: XsfDriver, beam_incidence_angles: np.ndarray, beam_azimuth_angles: np.ndarray, tx_depths: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes detection depth, along and across distance, in surface coordinate system, with origin relative to Tx array position, given :
    - sound velocity profiles, and transducer measured sound velocity,
    - geographic beam launch incidence and azimuth angles,
    - platform draft.
    Accounts for acoustic beam refraction.
    """
    # get two-way travel time in seconds
    one_way_travel_times = xsf.read_detection_two_way_travel_time() / 2
    # Get sound velocity at transduducer depth
    sv_at_tx_depth = xsf.read_sound_speed_at_transducer()
    # Get sound velocity profiles index from XSF file
    ssp_idx = xsf.get_ssp_idx()

    # init output arrays
    detection_x = np.full(shape=one_way_travel_times.shape, fill_value=np.nan)
    detection_y = np.full(shape=one_way_travel_times.shape, fill_value=np.nan)
    detection_z = np.full(shape=one_way_travel_times.shape, fill_value=np.nan)

    # loop through pings
    for ping, (
        one_way_travel_time,
        beam_incidence_angle,
        beam_azimuth_angle,
        tx_depth,
        sv_tx,
        ssp_i,
    ) in enumerate(
        zip(
            one_way_travel_times,
            beam_incidence_angles.reshape(one_way_travel_times.shape),
            beam_azimuth_angles.reshape(one_way_travel_times.shape),
            tx_depths.reshape(one_way_travel_times.shape),
            sv_at_tx_depth,
            ssp_idx,
        )
    ):
        if all(np.isnan(one_way_travel_time)):
            # if no detection where made for this ping let's
            continue
        if any(np.isnan(tx_depth)) or np.ma.is_masked(tx_depth):
            # if missing tx_depth value
            continue

        # get sound velocity profile for current ping
        svp_depths, svp_values = xsf.read_sound_speed_profile(ssp_i)

        # perform ray-tracing
        detection_z[ping], horizontal_distance, _ = raytracing_by_time(
            svp_depths, svp_values, one_way_travel_time, beam_incidence_angle, tx_depth, sv_tx
        )

        # compute along and across distance, from azimuth angle and horizontal distance
        detection_x[ping] = horizontal_distance * np.cos(np.deg2rad(beam_azimuth_angle))
        detection_y[ping] = horizontal_distance * np.sin(np.deg2rad(beam_azimuth_angle))

    return detection_x, detection_y, detection_z


def offset_detection_xyz_to_origin(
    detection_x: np.ndarray, detection_y: np.ndarray, detection_z: np.ndarray, tx_off_a2o: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Offsets detection position relative to Tx to reference point origin
    """
    for ping, tx_off in enumerate(tx_off_a2o):
        # offset to reference point origin
        detection_x[ping] += tx_off[:, 0]
        detection_y[ping] += tx_off[:, 1]
        detection_z[ping] += tx_off[:, 2]

    return detection_x, detection_y, detection_z


def compute_detection_lonlat(
    detection_x: np.ndarray, detection_y: np.ndarray, headings: np.ndarray, nav_lons: np.ndarray, nav_lats: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes detection latitude and longitude from detection (x,y) position ref. platform origin, heading and navigation
    """
    detection_latitude = np.full(shape=detection_x.shape, fill_value=np.nan)
    detection_longitude = np.full(shape=detection_x.shape, fill_value=np.nan)
    # TODO : use high frequency heading and navigation
    for ping, (heading, nav_lon, nav_lat) in enumerate(zip(headings, nav_lons, nav_lats)):
        detection_longitude[ping], detection_latitude[ping] = compute_detection_position(
            along=detection_x[ping],
            across=detection_y[ping],
            heading=heading,
            nav_latitude=nav_lat,
            nav_longitude=nav_lon,
        )
    return detection_longitude, detection_latitude


def compute_soundings_position(
    xsf: nc.Dataset,
    beam_incidence_angle: np.ndarray,
    beam_azimuth_angle: np.ndarray,
    draft: np.ndarray,
    tx_off_a2o: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # compute detection X,Y,Z position relative to Tx array
    detection_x, detection_y, detection_z = compute_detection_xyz(xsf, beam_incidence_angle, beam_azimuth_angle, draft)
    # offset detection position relative to Tx to reference point origin
    detection_x, detection_y, detection_z = offset_detection_xyz_to_origin(
        detection_x,
        detection_y,
        detection_z,
        tx_off_a2o.reshape(detection_x.shape + (3,)),
    )
    # Get heading, platform_latitude and platform_longitude
    heading = xsf[sg.BeamGroup1Grp.PLATFORM_HEADING(ident=DFLT_BM_GRP)][:]
    nav_latitude = xsf[sg.BeamGroup1Grp.PLATFORM_LATITUDE(ident=DFLT_BM_GRP)][:]
    nav_longitude = xsf[sg.BeamGroup1Grp.PLATFORM_LONGITUDE(ident=DFLT_BM_GRP)][:]
    # compute detection latitude and longitude
    detection_lon, detection_lat = compute_detection_lonlat(
        detection_x, detection_y, heading, nav_longitude, nav_latitude
    )

    # find out which detection position were not computed due to missing attitude, and update flag status accordingly
    status = xsf[sg.BathymetryGrp.STATUS(ident=DFLT_BM_GRP)][:]
    status_detail = xsf[sg.BathymetryGrp.STATUS_DETAIL(ident=DFLT_BM_GRP)][:]
    not_computed = np.isnan(detection_x)
    to_be_flagged = np.logical_and(not_computed, np.logical_not(status))
    status[to_be_flagged] = 1  # rejected
    status_detail[to_be_flagged] = 1  # automatic
    xsf[sg.BathymetryGrp.STATUS(ident=DFLT_BM_GRP)][:] = status[:]
    xsf[sg.BathymetryGrp.STATUS_DETAIL(ident=DFLT_BM_GRP)][:] = status_detail[:]

    # update XSF detection position variables, preserving old ones for flagged values
    detection_x_orig = xsf[sg.BathymetryGrp.DETECTION_X(ident=DFLT_BM_GRP)][:]
    detection_y_orig = xsf[sg.BathymetryGrp.DETECTION_Y(ident=DFLT_BM_GRP)][:]
    detection_z_orig = xsf[sg.BathymetryGrp.DETECTION_Z(ident=DFLT_BM_GRP)][:]
    detection_lat_orig = xsf[sg.BathymetryGrp.DETECTION_LATITUDE(ident=DFLT_BM_GRP)][:]
    detection_lon_orig = xsf[sg.BathymetryGrp.DETECTION_LONGITUDE(ident=DFLT_BM_GRP)][:]

    detection_x[not_computed] = detection_x_orig[not_computed]
    detection_y[not_computed] = detection_y_orig[not_computed]
    detection_z[not_computed] = detection_z_orig[not_computed]
    detection_lat[not_computed] = detection_lat_orig[not_computed]
    detection_lon[not_computed] = detection_lon_orig[not_computed]

    xsf[sg.BathymetryGrp.DETECTION_X(ident=DFLT_BM_GRP)][:] = detection_x[:]
    xsf[sg.BathymetryGrp.DETECTION_Y(ident=DFLT_BM_GRP)][:] = detection_y[:]
    xsf[sg.BathymetryGrp.DETECTION_Z(ident=DFLT_BM_GRP)][:] = detection_z[:]
    xsf[sg.BathymetryGrp.DETECTION_LATITUDE(ident=DFLT_BM_GRP)][:] = detection_lat[:]
    xsf[sg.BathymetryGrp.DETECTION_LONGITUDE(ident=DFLT_BM_GRP)][:] = detection_lon[:]

    return detection_x, detection_y, detection_z, detection_lon, detection_lat
