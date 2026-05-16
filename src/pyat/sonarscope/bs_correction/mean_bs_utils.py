from typing import List, Tuple

import numpy as np
import xarray as xr
from scipy.interpolate import make_smoothing_spline

from ..common.configuration import default_config
from .mean_bs_model import BackscatterCurve, BackscatterCurveByIncidence, BackscatterCurveByTransmission, MeanBSModel
from . import gsab_model


def compute_means(means_per_file: List[np.ndarray], count_per_file: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """compute means statistics values
    @param means_per_file the list of 3darray (rx_antenna, tx_beam, angles) containing mean values for each file
    @param count_per_file the list of 3darray (rx_antenna, tx_beam, angles) containing value count for each file
    """
    # convert lists to array
    mean_values = np.stack(means_per_file, axis=0)
    counts = np.stack(count_per_file, axis=0)

    values_sums = mean_values * counts

    total_value_sum = np.nansum(values_sums, axis=0)
    total_count_sum = np.nansum(counts, axis=0)
    # compute mean values = sum of values / count
    total_value_sum[total_value_sum == 0] = np.nan
    total_means = total_value_sum / total_count_sum
    return total_means, total_count_sum


def apply_spline_filtering(x, y, count):
    """
    Apply smoothing spline filtering.
    """
    mask = np.isfinite(y)
    if np.any(mask):
        if np.sum(mask) <= 4:
            # not enough points to fit a spline
            filtered_data = y.copy()
        else:
            wcount = count / np.nanmean(count[mask])
            spl = make_smoothing_spline(x[mask], y[mask], w=wcount[mask], lam=1)
            filtered_data = spl(x)
        filtered_data[~mask] = np.nan
    else:
        filtered_data = np.full_like(y, fill_value=np.nan, dtype=np.float64)
    return filtered_data


def merge_incidence_curves(
    input_curves: List[BackscatterCurveByIncidence],
    use_raw_data: bool = True,
    apply_filtering: bool = True,
    filter_angle_lower_limit: float = 3.0,
) -> BackscatterCurveByIncidence:
    """
    Merge all BackscatterCurveByIncidence objects into a single BackscatterCurveByIncidence.
    This function computes the mean backscatter values by incidence angle across all input curves.
    @param input_curves: List of BackscatterCurveByIncidence objects to merge.
    @param use_raw_data: If True, use raw mean backscatter values and counts as weights; otherwise, use processed mean values.
    @param apply_filtering: If True, apply spline filtering to the merged data.
    @param filter_angle_lower_limit: incidence angle below which no filtering is applied.
    """
    means_by_incidence_per_file_linear = []
    count_by_incidence_per_file = []
    angle_by_incidence_per_file = []

    for curve_by_incidence in input_curves:
        ds_incidence = curve_by_incidence.ds

        # get mean values per file for this mode
        if use_raw_data:
            mean_values_incidence = ds_incidence[BackscatterCurve.RAW_MEAN_BS].data
            count_incidence = ds_incidence[BackscatterCurve.VALUE_COUNT].data
        else:
            mean_values_incidence = ds_incidence[BackscatterCurve.MEAN_BS].data
            count_incidence = np.ones_like(mean_values_incidence)
        # switch to linear
        mean_values_incidence = default_config.db_to_linear(mean_values_incidence)
        angle_incidence = ds_incidence[BackscatterCurve.ANGLE].data
        # mult by the value count, this contains the sum of values in linear scale
        means_by_incidence_per_file_linear.append(mean_values_incidence)
        count_by_incidence_per_file.append(count_incidence)
        angle_by_incidence_per_file.append(angle_incidence)
    # compute the sum of each values per beam, taking into account for nan values
    mean_by_incidence_linear, count_by_incidence_sum = compute_means(
        means_by_incidence_per_file_linear, count_by_incidence_per_file
    )
    mean_values_by_incidence = default_config.linear_to_db(mean_by_incidence_linear)

    # check that all angles are the same
    if not all(np.array_equal(angle_by_incidence_per_file[0], angles) for angles in angle_by_incidence_per_file):
        raise ValueError("All incidence angles must be the same across input curves.")

    # apply spline filtering
    if apply_filtering:
        filtered_data = apply_spline_filtering(
            x=angle_by_incidence_per_file[0],
            y=mean_values_by_incidence,
            count=count_by_incidence_sum,
        )
        # force angles lower than 3 deg not being filtered
        filtered_data[angle_by_incidence_per_file[0] < filter_angle_lower_limit] = mean_values_by_incidence[
            angle_by_incidence_per_file[0] < filter_angle_lower_limit
        ]
    else:
        # if no filtering, use the mean values directly
        filtered_data = mean_values_by_incidence

    # create curve by incidence
    return BackscatterCurveByIncidence.build(
        mean_values=filtered_data,
        count=count_by_incidence_sum.astype(np.int32),
        bin_centers=angle_by_incidence_per_file[0],
        raw_mean_values=mean_values_by_incidence,
        origin=None,
    )


def merge_transmission_curves(
    input_curves: List[BackscatterCurveByTransmission],
    apply_filtering: bool = True,
) -> BackscatterCurveByTransmission:
    """
    Merge all BackscatterCurveByTransmission objects into a single BackscatterCurveByTransmission.
    This function computes the mean backscatter values by transmission angle across all input curves.
    """
    means_by_transmission_per_file_linear = []
    means_residual_by_transmission_per_file_linear = []
    count_by_transmission_per_file = []
    angle_by_transmission_per_file = []

    for curve_by_transmission in input_curves:
        ds_transmission = curve_by_transmission.ds

        # get mean values per file for this mode
        mean_values_transmission = ds_transmission[BackscatterCurve.MEAN_BS].data
        mean_residual_values_transmission = ds_transmission[BackscatterCurve.RAW_MEAN_RESIDUAL_BS].data

        # switch to linear
        mean_values_transmission = default_config.db_to_linear(mean_values_transmission)
        mean_residual_values_transmission = default_config.db_to_linear(mean_residual_values_transmission)
        count_transmission = ds_transmission[BackscatterCurve.VALUE_COUNT].data
        angle_transmission = ds_transmission[BackscatterCurve.ANGLE].data

        # mult by the value count, this contains the sum of values in linear scale
        means_by_transmission_per_file_linear.append(mean_values_transmission)
        means_residual_by_transmission_per_file_linear.append(mean_residual_values_transmission)
        count_by_transmission_per_file.append(count_transmission)
        angle_by_transmission_per_file.append(angle_transmission)

    # compute the sum of each values per beam, taking into account for nan values
    mean_by_transmission_linear, count_by_transmission_sum = compute_means(
        means_by_transmission_per_file_linear, count_by_transmission_per_file
    )
    mean_values_by_transmission = default_config.linear_to_db(mean_by_transmission_linear)

    mean_residual_by_transmission_linear, count_by_transmission_sum = compute_means(
        means_residual_by_transmission_per_file_linear, count_by_transmission_per_file
    )
    mean_residual_values_by_transmission = default_config.linear_to_db(mean_residual_by_transmission_linear)

    # check that all angles are the same
    if not all(np.array_equal(angle_by_transmission_per_file[0], angles) for angles in angle_by_transmission_per_file):
        raise ValueError("All transmission angles must be the same across input curves.")

    # apply spline filtering on residuals
    if apply_filtering:
        filtered_residuals_data = np.full_like(
            mean_residual_values_by_transmission, fill_value=np.nan, dtype=np.float64
        )
        for rx_antenna in range(mean_residual_values_by_transmission.shape[0]):
            for tx_beam in range(mean_residual_values_by_transmission.shape[1]):
                filtered_residuals_data[rx_antenna][tx_beam] = apply_spline_filtering(
                    x=angle_by_transmission_per_file[0],
                    y=mean_residual_values_by_transmission[rx_antenna][tx_beam],
                    count=count_by_transmission_sum[rx_antenna][tx_beam],
                )
    else:
        # if no filtering, use the mean residual values directly
        filtered_residuals_data = mean_residual_values_by_transmission

    return BackscatterCurveByTransmission.build(
        rx_antenna_count=mean_residual_values_by_transmission.shape[0],
        tx_beam_count=mean_residual_values_by_transmission.shape[1],
        mean_values=mean_values_by_transmission,
        mean_residual_values=filtered_residuals_data,
        count=count_by_transmission_sum,
        raw_mean_residual_values=mean_residual_values_by_transmission,
        bin_centers=angle_by_transmission_per_file[0],
        origin=None,
    )


def merge_curves_by_incidence(mean_bs: MeanBSModel, use_raw_data: bool = True) -> BackscatterCurveByIncidence:
    """
    Merge all BackscatterCurveByIncidence objects from the mean_bs model into a single BackscatterCurveByIncidence.
    @param input_curves: List of BackscatterCurveByIncidence objects to merge.
    @param use_raw_data: If True, use raw mean backscatter values and counts as weights; otherwise, use smoothed mean values.
    """
    curves_by_incidence = [mean_bs.model[mode][0] for mode in mean_bs.model]
    return merge_incidence_curves(
        input_curves=curves_by_incidence,
        use_raw_data=use_raw_data,
        apply_filtering=True,
    )


def split_curves(mean_bs: MeanBSModel) -> BackscatterCurveByIncidence:
    """
    Merge all BackscatterCurveByIncidence objects from the mean_bs model into a single BackscatterCurveByIncidence.
    """
    curves_by_incidence = [mean_bs.model[mode][0] for mode in mean_bs.model]
    return merge_incidence_curves(
        input_curves=curves_by_incidence,
        apply_filtering=True,
    )


def fit_gsab_to_incidence_curve(
    incidence_curve: BackscatterCurveByIncidence, use_raw_data: bool = True
) -> BackscatterCurveByIncidence:
    """
    Fit GSAB model to the incidence curve and return a new incidence curve with fitted values.
    """
    # get mean values per file for this mode
    x = incidence_curve.ds[BackscatterCurve.ANGLE][:]
    count = incidence_curve.ds[BackscatterCurve.VALUE_COUNT][:]

    if use_raw_data:
        y = incidence_curve.ds[BackscatterCurve.RAW_MEAN_BS][:]
        count = incidence_curve.ds[BackscatterCurve.VALUE_COUNT][:]
    else:
        y = incidence_curve.ds[BackscatterCurve.MEAN_BS][:]
        count = xr.ones_like(count)

    gsab_data_model = gsab_model.GsabDataModel(x, y, count)
    gsab_data_model.fit_gsab()

    return BackscatterCurveByIncidence.build(
        raw_mean_values=y.data,
        mean_values=gsab_data_model.apply().data,
        count=count.data,
        bin_centers=gsab_data_model.x.data,
        origin=None,
    )
