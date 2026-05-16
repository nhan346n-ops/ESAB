import glob
import os
import shutil
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import sonar_netcdf.sonar_groups as sg

from pyat.xsf.bathy.raytracing.beam_launch_vector import compute_beam_pointing_vector_in_scs
from pyat.xsf.bathy.raytracing.raytracing import compute_soundings_position
from pyat.sensor.phins_repeater_driver import read_phins_repeater_as_df
from pyat.sonarscope.model.constants import (
    BATHY_GROUP_NAME,
    BEAM_GROUP_NAME,
    DEFAULT_BEAM_GROUP_IDENT,
)
from pyat.sonarscope.model.sounder_lib import SounderRawFileFormat
from pyat.utils.path_utils import splitext_of_fname
from pyat.xsf import xsf_driver


def find_missing_attitude_data(xsf_file):
    """
    returns missing attitude data time interval and ping interval in XSF file
    """
    with nc.Dataset(xsf_file, mode="r") as xsf:
        attitude_time = xsf[sg.AttitudeSubGroup.get_group_path("001")].variables["time"][:].astype("datetime64[ns]")
        # time sort attitude data
        attitude_time.sort()
        ping_time = xsf[sg.BeamGroup1Grp.PING_TIME(ident=DEFAULT_BEAM_GROUP_IDENT)][:]

    # identify missing attitude data, based on attitude frequency
    attitude_freq = np.ma.median(np.diff(attitude_time))
    gap_start = np.flatnonzero(np.diff(attitude_time) > 1.5 * attitude_freq)
    gap_end = gap_start + 1  # min(gap_start + 1, len(attitude_time))

    # TODO : suppress this, once converter will be corrected.
    # Remove duplicate start and stops (lonely values)
    common_elements = np.intersect1d(gap_start, gap_end)
    # Remove the common elements from both arrays
    gap_start = gap_start[~np.isin(gap_start, common_elements)]
    gap_end = gap_end[~np.isin(gap_end, common_elements)]

    # Combine the start and end of gaps into a DataFrame
    missing_ranges = pd.DataFrame({"start": attitude_time[gap_start], "end": attitude_time[gap_end]})

    # find corresponding ping number
    missing_pings = []
    for index, missing_range in missing_ranges.iterrows():
        affected_pings = np.flatnonzero(
            (ping_time >= missing_range["start"].value) & (ping_time <= missing_range["end"].value)
        )
        if affected_pings.size > 0:
            missing_pings.append({"start": affected_pings[0], "end": affected_pings[-1]})
    missing_pings = pd.DataFrame(missing_pings)

    return missing_ranges, missing_pings


def nc_group_to_df(nc_group: nc.Group) -> pd.DataFrame:
    """
    Reads a NetCDF group, and returns a DataFrame
    """
    data_dict = {}
    for var_name in nc_group.variables:
        if var_name == "time":
            var_data = nc_group.variables[var_name][:].astype("datetime64[ns]")
            data_dict[var_name] = var_data
        else:
            var_data = nc_group.variables[var_name][:]
            data_dict[var_name] = var_data.flatten()  # Flattening might be needed depending on variable dimensions

    return pd.DataFrame(data_dict)


def xsf_attitude_to_df(xsf_file):
    """
    Reads attitude sub group data as a dataframe
    """
    with nc.Dataset(xsf_file, mode="r") as xsf:
        # read attitude subgroup data
        attitude = xsf[sg.AttitudeSubGroup.get_group_path("001")]
        attitude_df = nc_group_to_df(attitude)
        # read vendor specific subgroup
        attitude_vendor = xsf[sg.AttitudeSubGroupVendorSpecificGrp.get_group_path("001")]
        attitude_df_vendor = nc_group_to_df(attitude_vendor)

        # merge them and return a time-indexed sorted DataFrame
        return attitude_df.join(attitude_df_vendor).set_index("time").sort_index()


def insert_attitude_from_phinsrepeater(xsf_attitude, phins_attitude, missing_ranges):
    """
    Inserts missing attitude from phins data file into original data
    """
    # rename "heave" to "vertical_offset"
    phins_attitude.rename(columns={"heave": "vertical_offset"}, inplace=True)
    # negate pitch (not the same reference in phins
    phins_attitude["pitch"] = -phins_attitude["pitch"]
    # iterate over missing time range...
    for _, missing_range in missing_ranges.iterrows():
        # ... and combine the original and missing DataFrames
        xsf_attitude = pd.concat(
            [xsf_attitude, phins_attitude.loc[missing_range["start"] : missing_range["end"]]]
        ).sort_index()

    return xsf_attitude


def check_mru_vs_attitude_subgrp(xsf: nc.Dataset):
    """
    Check if we can add a new attitude subgroup : compare declared MRU ids and attitude subgroups
    NB : there are often 2 declared MRUs, without corresponding Platform/Attitude subgroup
    """
    att_grp_path = sg.AttitudeGrp.get_group_path()
    mru_ids = xsf[sg.PlatformGrp.MRU_IDS()][:]
    for i, mru_id in enumerate(mru_ids):
        if mru_id not in xsf[att_grp_path].groups:
            return i, mru_id
    # TODO : add the possibility to resize MRU dimension. Raise a NotImplementedError for now
    raise NotImplementedError(
        "Unable to add a new attitude subgroup inside an already existing file with a fixed MRU dimension"
    )


def add_attitude_subgroup(
    xsf_file, att_data: pd.DataFrame, ident: str = None, description="imported afterward", origin="unknown"
) -> int:
    """
    Adds attitude data from a panda time-indexed dataframe into a new sonarNetCDF attitude subgroup
    """
    with nc.Dataset(xsf_file, mode="r+") as xsf:

        # Check if we can add a new attitude subgroup : compare declared MRU ids and attitude subgroups
        i, mru_id = check_mru_vs_attitude_subgrp(xsf)

        # incrementing ident if none provided
        if ident is None:
            ident = mru_id
        else:
            # rename mru id with given ident
            xsf[sg.PlatformGrp.MRU_IDS()][i] = ident

        # create a new attitude subGroup
        att_structure = sg.AttitudeSubGroup()
        att_grp = att_structure.create_group(
            parent_group=xsf[sg.AttitudeGrp.get_group_path()],
            ident=ident,
            description=description,
            origin=origin,
        )
        # create time dimension and add data
        att_structure.create_dimension(att_grp, {att_structure.TIME_DIM_NAME: len(att_data)})
        time_data = att_structure.create_time(att_grp)
        time_data[:] = att_data.index.astype("int64")

        # add all other attitude variables, based on the dataframe column name, and only if it matches group variable names
        for att_name in att_data.columns:
            try:
                # retrieve corresponding create function signature
                create_func = getattr(att_structure, f"create_{att_name}")
                # create variable
                variable = create_func(att_grp)
                # and add data
                variable[:] = att_data[att_name]
            except AttributeError:
                # no create function relative to dataframe column name
                continue

        # increment MRU_ids, MRU_offsets, MRU_rotation
        # TODO

        return i


def apply_attitude(xsf_file, mru_idx: int):  # , ping_range=None):
    """
    Apply given attitude to MBES data
    """
    with xsf_driver.open_xsf(file_path=xsf_file, mode="r") as xsf:
        sounder_format = SounderRawFileFormat.from_dataset(xsf)

    with nc.Dataset(xsf_file, mode="r+") as xsf:
        # Set preferred_MRU attribute,
        # Index matches the ones used in the Platform MRU sensors variables.
        xsf[BEAM_GROUP_NAME].setncattr("preferred_MRU", mru_idx)

        # # Compute geographic launch vectors for each beam
        # beam_incidence_angle, beam_azimuth_angle, draft, tx_offA2O = compute_geographic_launch_vector_v2(
        #     xsf, raw_file_format=sounder_format, algo="ncca_parallel"
        # )

        # # Perform raytracing using beam incidence angle
        # detection_x, detection_y, detection_z, detection_longitude, detection_latitude = compute_soundings_position(
        #     xsf, beam_incidence_angle, beam_azimuth_angle, draft, tx_offA2O
        # )

        # Compute geographic launch vectors for each beam
        beam_incidence_angle, beam_azimuth_angle, owtt_tx, draft, tx_offA2O = compute_beam_pointing_vector_in_scs(
            xsf, algo="ncca_parallel"
        )
        # write One-Way Travel Time to XSF file, double it to get Two-Way Travel Time
        xsf[BATHY_GROUP_NAME].variables[sg.BathymetryGrp.DETECTION_TWO_WAY_TRAVEL_TIME_VNAME][:] = 2 * owtt_tx[:]
        # Perform raytracing using beam incidence angle
        compute_soundings_position(xsf, beam_incidence_angle, beam_azimuth_angle, draft, tx_offA2O)


if __name__ == "__main__":
    # # Paths to the original and new files.nc"
    xsf_file_dir = r"D:\AFF-SANTORIN-485-02\rejeu\XSF_test"  # r"D:\ESSULYX24\PL06\XSF"
    path_out = r"D:\AFF-SANTORIN-485-02\rejeu\XSF_test"
    PHINS_repeater_reference_file = r"D:\AFF-SANTORIN-485-02\AFF-SANTORIN-485-02\DATA\PHINS\PHINS_REPEATER_07042025_042227.log"  # r"D:\ESSULYX24\PL06\PHINS_REPEATER_2024-04-26_05-26-18_001.txt"

    phins_data = pd.DataFrame()
    attitude_cols = ["heading", "roll", "pitch", "heave"]

    for file in glob.glob(os.path.join(xsf_file_dir, "*.xsf.nc")):
        name, ext = splitext_of_fname(file)
        if "phins_repeater" not in name:
            file_i = file
            file_o = os.path.join(path_out, Path(name).stem + "_" + "phins_repeater" + "." + ext)

            print(f"{Path(name).stem} : inserting phins repeater attitude")
            # Copy input xsf to o_xsf before integrating new attitude data
            shutil.copy(file_i, file_o)
            # read attitude data from xsf as a dataframe
            xsf_attitude = xsf_attitude_to_df(file_i)
            # read attitude from phins data file,
            # only keep orientation angles and heave, and clip data to xsf time range
            if phins_data.empty:
                phins_data = read_phins_repeater_as_df(PHINS_repeater_reference_file)
            phins_attitude = phins_data[attitude_cols].loc[xsf_attitude.index[0] : xsf_attitude.index[-1]]
            # rename "heave" to "vertical_offset"
            phins_attitude.rename(columns={"heave": "vertical_offset"}, inplace=True)
            # negate pitch (not the same reference in phins
            phins_attitude["pitch"] = -phins_attitude["pitch"]
            # create a new attitude subgroup with phins data
            mru_i = add_attitude_subgroup(file_o, att_data=phins_attitude, ident="phins_repeater")
            # apply attitude data for missing ping ranges
            apply_attitude(file_o, mru_idx=mru_i)  # , ping_range=missing_pings)

            # # Is there missing attitude data in this file ?
            # missing_ranges, missing_pings = find_missing_attitude_data(file_i)

            # # if so ...
            # if len(missing_ranges) > 0:
            #     print(f"{Path(name).stem} : inserting phins repeater attitude")
            #     # Copy input xsf to o_xsf before integrating new attitude data
            #     shutil.copy(file_i, file_o)
            #     # read attitude data from xsf as a dataframe
            #     xsf_attitude = xsf_attitude_to_df(file_i)

            #     # read attitude from phins data file,
            #     # only keep orientation angles and heave, and clip data to xsf time range
            #     if phins_data.empty:
            #         phins_data = read_phins_repeater_as_df(PHINS_repeater_reference_file)
            #     phins_attitude = phins_data[attitude_cols].loc[xsf_attitude.index[0] : xsf_attitude.index[-1]]
            #     # Insert missing attitude from phins data file into original data
            #     xsf_attitude = insert_attitude_from_phinsrepeater(xsf_attitude, phins_attitude, missing_ranges)

            #     # create a new attitude subgroup
            #     mru_i = add_attitude_subgroup(file_o, att_data=xsf_attitude, ident="phins_repeater")

            #     # apply attitude data for missing ping ranges
            #     apply_attitude(file_o, mru_idx=mru_i)  # , ping_range=missing_pings)
