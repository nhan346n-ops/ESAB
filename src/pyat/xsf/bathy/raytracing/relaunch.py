import os
import shutil
import tempfile
from typing import List

import pandas as pd
import sonar_netcdf.sonar_groups as sg
from pygws.service.progress_monitor import ProgressMonitor

import pyat.utils.pyat_logger as log
from pyat.xsf.bathy.raytracing.beam_launch_vector import compute_beam_pointing_vector_in_scs
from pyat.xsf.bathy.raytracing.raytracing import compute_soundings_position
from pyat.xsf.bathy.raytracing.update_attitude import add_attitude_subgroup, xsf_attitude_to_df
from pyat.sensor.phins_repeater_driver import read_phins_repeater_as_df
from pyat.sonarscope.model.constants import BATHY_GROUP_NAME, BEAM_GROUP_NAME
from pyat.xsf import xsf_driver


def relaunch(i_paths: List[str], o_paths: List[str], algo="ncca_parallel", overwrite=False) -> None:
    """
    recompute new detection position from the data contained in the input XSF file, and save the output to the output XSF file.
    """
    # progress monitor and logger init
    logger = log.logging.getLogger(__name__)
    monitor = ProgressMonitor()
    monitor.begin_task("Relaunch process", len(i_paths))

    for i_file, o_file in zip(i_paths, o_paths):
        logger.info(f"Relaunching {i_file} to {o_file}")
        tmp_o_file = tempfile.mktemp(suffix=os.path.basename(i_file))
        # Copy input file to output path"
        if not overwrite and os.path.exists(o_file):
            logger.warning(f"File {o_file} already exists and overwrite is not allowed, skipping it")
            continue

        # create temp file
        logger.info(f"Using {tmp_o_file} as a temporary outputfile")
        shutil.copy(i_file, tmp_o_file)

        with xsf_driver.open_xsf(file_path=tmp_o_file, mode="r+") as xsf:
            # Compute geographic launch vectors and One-Way Travel Time (OWTT) for each beam
            beam_incidence_angle, beam_azimuth_angle, owtt_tx, draft, tx_off_a2o = compute_beam_pointing_vector_in_scs(
                xsf, algo=algo
            )
            # write OWTT to XSF file, double it to get TWTT
            xsf[BATHY_GROUP_NAME].variables[sg.BathymetryGrp.DETECTION_TWO_WAY_TRAVEL_TIME_VNAME][:] = 2 * owtt_tx[:]
            # Perform raytracing using beam incidence angle
            compute_soundings_position(xsf, beam_incidence_angle, beam_azimuth_angle, draft, tx_off_a2o)

        # everything went well, copy the result
        shutil.move(tmp_o_file, o_file)
        monitor.worked(1)

    monitor.done()


def relaunch_with_attitude(
    i_paths: List[str],
    o_paths: List[str],
    i_phinsrepeater_path: str,
    algo="ncca_parallel",
    overwrite=False,
) -> None:
    """
    Compute new detection position from the input XSF file data, using attitude data from a phins repeater file.
    """
    # progress monitor and logger init
    logger = log.logging.getLogger(__name__)
    monitor = ProgressMonitor()
    monitor.begin_task("Relaunch process with attitude", len(i_paths))

    # phins attitude data cache
    phins_data = pd.DataFrame()
    attitude_cols = ["heading", "roll", "pitch", "heave"]

    for i_file, o_file in zip(i_paths, o_paths):
        logger.info(f"Relaunching {i_file} to {o_file} with attitude from {i_phinsrepeater_path}")
        tmp_o_file = tempfile.mktemp(suffix=os.path.basename(i_file))

        if not overwrite and os.path.exists(o_file):
            logger.warning(f"File {o_file} already exists and overwrite is not allowed, skipping it")
            continue

        logger.info(f"Using {tmp_o_file} as a temporary outputfile")
        shutil.copy(i_file, tmp_o_file)

        # Get attitude data from xsf as a dataframe
        xsf_attitude = xsf_attitude_to_df(i_file)
        # read attitude from phins data file,
        # only keep orientation angles and heave, and clip data to xsf time range
        if phins_data.empty:
            logger.info(f"Reading Phins repeater data from {i_phinsrepeater_path}")
            phins_data = read_phins_repeater_as_df(i_phinsrepeater_path)
        phins_attitude = phins_data[attitude_cols].loc[xsf_attitude.index[0] : xsf_attitude.index[-1]]
        # rename "heave" to "vertical_offset"
        phins_attitude.rename(columns={"heave": "vertical_offset"}, inplace=True)
        # negate pitch (opposite in phins)
        phins_attitude["pitch"] = -phins_attitude["pitch"]
        # create a new attitude subgroup with phins data
        mru_idx = add_attitude_subgroup(tmp_o_file, att_data=phins_attitude, ident="phins_repeater")

        with xsf_driver.open_xsf(file_path=tmp_o_file, mode="r+") as xsf:
            # Set preferred_MRU attribute to phins_repeater data index in the Platform MRU sensors variables.
            xsf[BEAM_GROUP_NAME].setncattr("preferred_MRU", mru_idx)

            # Compute geographic launch vectors and OWTT for each beam using attitude from phins repeater file
            beam_incidence_angle, beam_azimuth_angle, owtt_tx, draft, tx_off_a2o = compute_beam_pointing_vector_in_scs(
                xsf, algo=algo
            )
            # write OWTT to XSF file, double it to get TWTT
            xsf[BATHY_GROUP_NAME].variables[sg.BathymetryGrp.DETECTION_TWO_WAY_TRAVEL_TIME_VNAME][:] = 2 * owtt_tx[:]
            # Perform raytracing using geographic launch vectors and OWTT
            compute_soundings_position(xsf, beam_incidence_angle, beam_azimuth_angle, draft, tx_off_a2o)

        shutil.move(tmp_o_file, o_file)
        monitor.worked(1)

    monitor.done()
