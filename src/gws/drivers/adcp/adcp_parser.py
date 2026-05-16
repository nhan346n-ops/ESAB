import netCDF4 as nc
import numpy as np
import pandas as pd
from mhkit import dolfyn

import gws.drivers.adcp.adcp_model as model
from pyat.utils.logger import logger


def parse_adcp_file(adcp_file_path: str) -> model.AdcpData:
    """
    returns a model.AdcpData of current data contained in the specified file
    """
    if adcp_file_path.endswith(".nc"):
        logger.info(f"Opening file {adcp_file_path} with NetCDF4")
        return parse_nc_adcp_file(adcp_file_path)
    else:
        logger.info(f"Opening file {adcp_file_path} with Dolfyn")
        return parse_dolfyn_adcp_file(adcp_file_path)


def parse_nc_adcp_file(adcp_file_path: str) -> model.AdcpData:
    """
    Parse a NetCDF ADCP file and return an model.AdcpData object.
    """
    try:

        with nc.Dataset(adcp_file_path, mode="r", format="NETCDF4") as dataset:
            logger.info(f"File {adcp_file_path} opened with NetCDF4")

            # Extract data from the dataset
            adcp_group = dataset["/Sonar/Beam_group1/ADCP"]
            mean_group = adcp_group["Mean_current"]

            # Build velocity arrays. Dimension is (time, sample)
            nv = _expand(mean_group["current_velocity_geographical_north"][:])
            ev = _expand(mean_group["current_velocity_geographical_east"][:])
            dv = _expand(mean_group["current_velocity_geographical_down"][:])
            # Array with True values when eastward_velocity and northward_velocity != NaN
            not_nan_velocity = np.isfinite(ev) & np.isfinite(nv) & np.isfinite(dv)

            # Compute elevations of vectors
            elevation_origin = adcp_group["depth_first_sample_center"][:]
            elevation_interval = adcp_group["vertical_sample_interval"][:]
            time_count = nv.shape[0]
            sample_count = nv.shape[1]

            ranges = np.tile(np.arange(sample_count, dtype=np.float32), (time_count, 1))
            ranges = ranges * elevation_interval[:, np.newaxis]
            ranges = -ranges - elevation_origin[:, np.newaxis]
            range_index = np.rint(np.interp(ranges, xp=[ranges.min(), ranges.max()], fp=[0, 100])).astype(np.int32)

            # Build an array of time indexes
            time_index = np.tile(np.arange(time_count, dtype=np.int32), (sample_count, 1)).T
            time_index = time_index[not_nan_velocity]

            return model.AdcpData(
                file_path=adcp_file_path,
                time=np.datetime64("1601-01-01 00:00:00") + mean_group["mean_time"][:].astype("timedelta64[ns]"),
                latitude=mean_group["mean_platform_latitude"][:],
                longitude=mean_group["mean_platform_longitude"][:],
                current_data=pd.DataFrame(
                    {
                        model.TIME_INDEX: time_index,
                        model.RANGE: ranges[not_nan_velocity],
                        model.RANGE_INDEX: range_index[not_nan_velocity],
                        model.EASTWARD_VELOCITY: ev[not_nan_velocity],
                        model.NORTHWARD_VELOCITY: nv[not_nan_velocity],
                        model.DOWNWARD_VELOCITY: dv[not_nan_velocity],
                    }
                ),
            )

    except IndexError as e:
        raise IOError(f"Not a Sonar ADCP file ({adcp_file_path})") from e


def parse_dolfyn_adcp_file(adcp_file_path: str) -> model.AdcpData:
    """
    Parse a Dolfyn ADCP file and return an model.AdcpData object."""
    with dolfyn.read(adcp_file_path) as dataset:
        # northward velocity
        nv = dataset["vel"].sel(dir="N").astype(np.float32).to_numpy()
        # eastward velocity
        ev = dataset["vel"].sel(dir="E").astype(np.float32).to_numpy()
        ev_shape = ev.shape
        # Array with True values when eastward_velocity and northward_velocity != NaN
        not_nan_velocity = np.isfinite(ev) & np.isfinite(nv)

        # Build an array of range indexes
        ranges = dataset["range"].astype(np.float32).to_numpy() * -1
        ranges = np.tile(ranges, (ev_shape[1], 1)).T
        range_index = np.rint(np.interp(ranges, xp=[ranges.min(), ranges.max()], fp=[0, 100])).astype(np.int32)

        # Build an array of time indexes
        time_index = np.tile(np.arange(ev_shape[1]), (ev_shape[0], 1)).astype(np.int32)
        time_index = time_index[not_nan_velocity]
        return model.AdcpData(
            file_path=adcp_file_path,
            time=dataset["time"].astype("datetime64[ns]").to_numpy(),
            latitude=dataset["latitude_gps"].astype(float).to_numpy(),
            longitude=dataset["longitude_gps"].astype(float).to_numpy(),
            current_data=pd.DataFrame(
                {
                    model.TIME_INDEX: time_index,
                    model.RANGE: ranges[not_nan_velocity],
                    model.RANGE_INDEX: range_index[not_nan_velocity],
                    model.EASTWARD_VELOCITY: ev[not_nan_velocity],
                    model.NORTHWARD_VELOCITY: nv[not_nan_velocity],
                    model.DOWNWARD_VELOCITY: np.zeros((len(time_index)), dtype=np.float32),
                }
            ),
        )


def _expand(array: np.ndarray) -> np.ndarray:
    """
    Flatten a list of lists of float32 into a 1D array
    """
    max_value_count = 0
    # Iterate through the array and count the number of values
    for sub_array in array:
        max_value_count = max(max_value_count, len(sub_array))

    # Create a new array to hold the flattened values
    result = np.full((len(array), max_value_count), dtype=np.float32, fill_value=np.nan)

    # Iterate through the array again and copy the values to the new array
    for index, sub_array in enumerate(array):
        sub_array_size = len(sub_array)
        result[index, :sub_array_size] = sub_array
    return result


if __name__ == "__main__":
    # Example usage
    adcp_file_path = r"e:\ifremer\data\adcp_cp300\CDLM_CP300_ESSTECH25_decimation_10cs_BT_OFF-D20250312-T075051.nc"
    # adcp_file_path = r"e:\ifremer\data\ADCP_Drix_STA\ADCP_DriX__20220922T101852_014_000000.STA"
    adcp_data = parse_adcp_file(adcp_file_path)
    print(f"time ({len(adcp_data.time)}): {adcp_data.time}")
    print(f"latitude : {adcp_data.latitude}")
    print(f"longitude : {adcp_data.longitude}")
    print("--- Current data ---")
    current_data = adcp_data.current_data
    print(f"Number of vector: {current_data.index.size}")
    print(f"time index : {current_data[model.TIME_INDEX]}")

    print("--- Filtered data ---")
    filtered_data = adcp_data.apply_time_filter(41, 41).current_data
    print(f"Number of vector: {filtered_data.index.size}")
    print(f"time index : {np.unique(filtered_data[model.TIME_INDEX])}")
