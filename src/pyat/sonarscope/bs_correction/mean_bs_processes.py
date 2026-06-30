import os
import pathlib
from typing import List, Optional

from ..common.configuration import default_config
from ..model.sonar_factories import ModeComputerFactory
from ..model.sounder_lib import SounderType
from . import mean_bs_utils
from .mean_bs_model import MeanBSModel


def csv_to_bsar_process(
    i_path: str,
    o_path: str,
    overwrite: bool = False,
    frequency: Optional[float] = None,
) -> None:
    """
    Convert mean bs model by incidence from csv to bsar file.
    The csv delimiter is automatically guessed and the columns are: angle, bs_value
    @param i_path : input file path
    @param o_path : output file path
    @param overwrite : True to overwrite output files if needed
    @param frequency : Frequency in Hz to determine sounder mode.
    """
    mean_bs_model = MeanBSModel.import_from_csv(input_files=[i_path], frequency=frequency)
    if mean_bs_model is None:
        default_config.logger.error(f"File {i_path} does not contain mean BS data")
    else:
        # write bsar file
        mean_bs_model.save_to_netcdf(o_path, overwrite=overwrite)
        default_config.logger.info(f"Exported mean BS data to {o_path}")


def export_summary_lines_process(
    i_paths: List[str],
    sounder_type: str = SounderType.AUTO,
    o_dir: Optional[str] = None,
) -> None:
    """Export summary lines from XSF files to a directory.
    @param sounder_type : type from sounder_lib.SounderType
    @param i_paths : input file paths
    @param o_path : output dir path
    """

    # Checks input files
    default_config.check_files_version(i_paths)
    sounder_type = default_config.check_files_soundertype(i_paths, sounder_type=sounder_type)

    # Find avalable modes
    default_config.logger.info("Modes :")
    mode_computer = ModeComputerFactory.create_mode_computer(sounder_type=sounder_type)
    keymodes_ids, file_modeids = mode_computer.compute(input_files=i_paths)
    for mode in keymodes_ids.keys():
        if mode.is_valid():
            default_config.logger.info(mode)

    # If o_dir is empty or None, use the first file path as output directory
    if not o_dir:
        o_dir = os.path.dirname(i_paths[0])
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    default_config.logger.info(f"Exporting summary lines to {o_dir}")
    # Write a txt file for each mode containing the files where the mode is present
    for mode, modeid in keymodes_ids.items():
        if mode.is_valid():
            outfile_path = pathlib.Path(o_dir, f"SummaryLines-{str(mode)}.txt")
            with outfile_path.open(mode="w", encoding="utf-8") as outfile:
                for file, filemodeids in file_modeids.items():
                    if modeid in filemodeids:
                        outfile.write(f"{file}\n")


def merge_bsar_process(
    i_paths: List[str],
    o_path: str,
    overwrite: bool = False,
) -> None:
    """
    Merge BSAR files.
    If a mode is present in several files, only first file is retained
    @param i_paths : input file paths
    @param o_path : output file path
    @param overwrite : True to overwrite output files if needed
    """

    merged_model = None
    for file in i_paths:
        meanbsmodel = MeanBSModel.read_from_netcdf(file, apply_conf=False)
        if merged_model is None:
            merged_model = meanbsmodel
        elif merged_model.sounder_type == meanbsmodel.sounder_type:
            for current_mode in meanbsmodel.model.keys():
                if current_mode not in merged_model.model.keys():
                    merged_model.model[current_mode] = meanbsmodel.model[current_mode]
                else:
                    # merge incidence and transmission curves
                    curve_by_incidence, curve_by_transmission = merged_model.model[current_mode]
                    new_curve_by_incidence, new_curve_by_transmission = meanbsmodel.model[current_mode]
                    if new_curve_by_incidence is not None:
                        curve_by_incidence = mean_bs_utils.merge_incidence_curves(
                            [curve_by_incidence, new_curve_by_incidence]
                        )
                    if new_curve_by_transmission is not None:
                        curve_by_transmission = mean_bs_utils.merge_transmission_curves(
                            [curve_by_transmission, new_curve_by_transmission]
                        )
                    merged_model.model[current_mode] = (curve_by_incidence, curve_by_transmission)
        else:
            default_config.logger.warning(
                f"different sounder types found: {merged_model.sounder_type} != {meanbsmodel.sounder_type}.\n Ignoring {file}"
            )
        # close model to release resources
        meanbsmodel = None

    if merged_model is None:
        default_config.logger.error("No valid BSAR files found to merge.")
        return
    merged_model.save_to_netcdf(output_file=o_path, overwrite=overwrite)
    merged_model = None


def split_bsar_process(
    i_path: str,
    o_dir: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """Split bsar files by acquisition mode to a directory.
    @param i_path : input file path
    @param o_dir : output dir path
    """

    meanbsmodel = MeanBSModel.read_from_netcdf(i_path, apply_conf=True)
    if meanbsmodel is None:
        default_config.logger.error(f"File {i_path} does not contain mean BS data")
        return

    # If o_dir is empty or None, use the file path as output directory
    if not o_dir:
        o_dir = os.path.dirname(i_path)
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    default_config.logger.info(f"Exporting bsar by mode to {o_dir}")
    # Write a txt file for each mode containing the files where the mode is present
    for mode in meanbsmodel.model.keys():
        # get output file path adding the mode name as suffix to i_path
        basename = os.path.basename(i_path)
        if basename.endswith(MeanBSModel.EXTENSION):
            basename = basename[: -len(MeanBSModel.EXTENSION)]
        outfile_path = pathlib.Path(o_dir, f"{basename}-{str(mode)}{MeanBSModel.EXTENSION}")
        # create a new MeanBSModel for the current mode
        bsmodel_by_mode = MeanBSModel(meanbsmodel.sounder_type, {mode: meanbsmodel.model[mode]})
        try:
            # save the model to the output file
            bsmodel_by_mode.save_to_netcdf(output_file=str(outfile_path), overwrite=overwrite)
        except Exception as e:
            default_config.logger.error(f"Error saving {mode} model to {outfile_path}: {e}")
        bsmodel_by_mode = None


def merge_bsar_incidence_process(
    i_path: str, o_path: str, overwrite: bool = False, use_raw_data: bool = True, apply_gsab_fitting: bool = False
) -> None:
    """
    Merges all curves by incidence angles and creates a new MeanBSModel with a single common mode.
    Value counts are summed up, and weghted mean values are computed.
    @param i_path : input file path
    @param o_path : output file path
    @param overwrite : True to overwrite output files if needed
    @param use_raw_data: If True, use raw mean backscatter values and counts as weights; otherwise, use smooted mean values
    @param apply_gsab_fitting: If True, apply auto 6 parameters gsab fitting
    """
    meanbs_model = MeanBSModel.read_from_netcdf(i_path, apply_conf=True)
    incidence_curve = mean_bs_utils.merge_curves_by_incidence(mean_bs=meanbs_model, use_raw_data=use_raw_data)
    if apply_gsab_fitting:
        default_config.logger.info("Starting GSAB fitting...")
        # Apply GSAB fitting if requested
        incidence_curve = mean_bs_utils.fit_gsab_to_incidence_curve(incidence_curve)
        default_config.logger.info("GSAB fitting completed.")

    merged_model = MeanBSModel.build_from_incidence(incidence_curve, sounder_type=meanbs_model.sounder_type)
    merged_model.save_to_netcdf(output_file=o_path, overwrite=overwrite)
    default_config.logger.info(f"Exported common BSAR data to {o_path}")
