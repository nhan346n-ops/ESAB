import os

import numpy as np
from scipy.interpolate import CubicSpline
from pyat.sonarscope.bs_correction.bscorr_model import BSCorrCurve, BSCorrModel
from pyat.sonarscope.bs_correction.mean_bs_model import BackscatterCurve, MeanBSModel
from pyat.xsf import xsf_driver

from ..common.configuration import default_config


def xsf_to_bscorr_process(
    i_path: str,
    o_path: str,
    overwrite: bool = False,
) -> None:
    """
    Export bscorr data from xsf file to bscorr file.
    @param i_paths : input file path
    @param o_path : output file path
    @param overwrite : True to overwrite output files if needed
    """

    with xsf_driver.open_xsf(file_path=i_path) as xsf_file:
        if not overwrite and os.path.exists(o_path):
            default_config.logger.error(f"Output file {o_path} already exist and overwrite is not allowed")
            raise IOError(f"Output file {o_path} already exist and overwrite is not allowed")

        bs_corr = xsf_file.get_bscorr()
        if bs_corr is None:
            default_config.logger.error(f"File {i_path} does not contain bscorr data")
        else:
            # write bscorr str to txt file with standard io
            default_config.logger.info(f"File {i_path} contains bscorr data")
            # use newline='' to avoid adding extra newlines on windows
            with open(o_path, "w", encoding="utf-8", newline="") as f:
                f.write(bs_corr)
            default_config.logger.info(f"Exported bscorr data to {o_path}")


def bsar_to_bscorr_process(
    i_path: str,
    i_bscorr_path: str,
    o_path: str,
    overwrite: bool = False,
) -> None:
    """Process backscatter data by combining BSAR and BSCorr models.
    This function reads a BSAR (Backscatter Angular Response) model and a BSCorr (Backscatter
    Correction) model, applies corrections based on mean residual backscatter values, and exports
    the modified BSCorr model to a text file.
    Args:
        i_path (str): Path to the input BSAR model file (NetCDF format)
        i_bscorr_path (str): Path to the input BSCorr model file (text format)
        overwrite (bool, optional): If True, allows overwriting existing output file.
            Defaults to False.
    Returns:
        None
    The function performs the following steps:
    1. Reads the BSAR model from NetCDF file
    2. Reads the BSCorr model from text file
    3. For each mode in BSCorr:
        - Finds corresponding mode in BSAR
        - Gets transmission curves for both models
        - Modifies BSCorr curve using BSAR mean residual values
        - Applies interpolation for smoothing
    4. Exports modified BSCorr model to text file
    The output filename is created by appending '_modified.txt' to the input BSCorr
    filename (without extension).
    Note:
        - Currently supports only single RX antenna configurations
        - If multiple RX antennas present, only first antenna is used
        - For KMALL format, calibration can be removed based on configuration
    """
    # export bscorr model to txt file
    # o_path = f"{os.path.splitext(i_bscorr_path)[0]}_modified.txt"
    if not overwrite and os.path.exists(o_path):
        default_config.logger.error(f"Output file {o_path} already exist and overwrite is not allowed")
        return
    meanbs = MeanBSModel.read_from_netcdf(i_path)
    if not meanbs.sounder_type:
        default_config.logger.error(f"File {i_path} does not contain sounder type")
        return
    bscorr = BSCorrModel.import_from_txt(i_bscorr_path, meanbs.sounder_type)
    if bscorr is None:
        default_config.logger.error(f"File {i_bscorr_path} does not contain bscorr data")
        return

    # for each mode in bscorr, get the corresponding meanbs
    for mode in bscorr.model.keys():
        # find the corresponding mode in meanbs
        meanbs_mode = meanbs.find_equivalent_mode(mode)
        if meanbs_mode is None:
            default_config.logger.warning(f"Mode {mode} not found in MeanBS model")
            continue

        # get the corresponding meanbs transmission curves
        _, meanbs_curve = meanbs.model[meanbs_mode]

        # get the corresponding bscorr
        bscorr_curve = bscorr.get_curve(mode)
        if bscorr_curve is None:
            default_config.logger.warning(f"Curve {mode} not found in bscorr")
            continue

        # modify the bscorr curve with the meanbs curve applying interpolation on meanbs curve
        bscorr_values = bscorr_curve.ds[BSCorrCurve.BS_CORR]
        # use only first rx antenna, dual head not yet supported
        if meanbs_curve.ds[BackscatterCurve.MEAN_RESIDUAL_BS].shape[0] > 1:
            default_config.logger.warning(
                f"MeanBS curve {meanbs_mode} has more than one RX antenna, using only first antenna"
            )
        # increment by sector
        for tx_index in range(meanbs_curve.ds[BackscatterCurve.MEAN_RESIDUAL_BS].shape[1]):
            meanbs_mask = np.isfinite(meanbs_curve.ds[BackscatterCurve.MEAN_RESIDUAL_BS][0][tx_index])
            meanbs_x = meanbs_curve.ds[BackscatterCurve.ANGLE]
            meanbs_y = meanbs_curve.ds[BackscatterCurve.MEAN_RESIDUAL_BS][0][tx_index]
            bscorr_mask = np.isfinite(bscorr_values[tx_index])
            bscorr_x = bscorr_curve.ds[BSCorrCurve.ANGLE][bscorr_mask]
            bscorr_y = bscorr_values[tx_index][bscorr_mask]
            if default_config.remove_calibration:  # kmall case
                bscorr_y = np.zeros_like(bscorr_y)
            # interpolate meanbs curve on bscorr curve
            if np.sum(meanbs_mask) < 2:
                # not enough points to fit a spline
                default_config.logger.warning(f"Not enough points to fit a spline for mode {mode}, tx index {tx_index}")
                continue
            meanbs_spl = CubicSpline(meanbs_x[meanbs_mask], meanbs_y[meanbs_mask])
            bscorr_values[tx_index][bscorr_mask] = bscorr_y - meanbs_spl(bscorr_x)
        # set the modified bscorr curve back to bscorr model
        bscorr_curve.ds[BSCorrCurve.BS_CORR] = bscorr_values
        bscorr.set_curve(mode, bscorr_curve)
        default_config.logger.info(f"Curve {mode} written in bscorr")

    default_config.logger.info(f"Write modified bscorr data to {o_path}")
    bscorr.export_to_txt(o_path)
