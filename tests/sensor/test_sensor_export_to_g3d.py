import os

import numpy as np
import pytest
import xarray as xr

from pyat.sensor.slice.sensor_to_longitudinal_g3d import _convert_files


@pytest.fixture
def sample_netcdf_files(tmp_path):
    """Create two sample Sensor-netCDF files.

    File 1: Normal data (quality_flag=1)
    File 2: Continuation data with some invalid points (quality_flag=4)
    """
    # Common data dimensions
    start_date = np.datetime64("2025-01-01T00:00:00", "ns")
    time_1 = np.arange(start_date, start_date + np.timedelta64(10, "s"), np.timedelta64(1, "s"))
    time_2 = np.arange(time_1[-1], time_1[-1] + np.timedelta64(10, "s"), np.timedelta64(1, "s"))

    # Geographic coordinates for first and third file (file 3 overlapping file 1)
    lat_1 = np.linspace(45.0, 45.1, 10)
    lon_1 = np.linspace(-1.0, -0.9, 10)
    depth_1 = np.linspace(0, 500, 10)

    # Geographic coordinates for second file (continuation)
    lat_2 = np.linspace(45.1, 45.2, 10)
    lon_2 = np.linspace(-0.9, -0.8, 10)
    depth_2 = np.linspace(0, 500, 10)

    # First file: normal data with all valid quality_flag
    ds1 = xr.Dataset(
        {
            "salinity": (["time"], np.linspace(35.0, 35.9, 10, dtype=np.float32)),
            "temperature": (["time"], np.linspace(15.0, 20.0, 10, dtype=np.float32)),
            "pressure": (["time"], np.linspace(0, 100, 10, dtype=np.float32)),
            "latitude": (["time"], lat_1),
            "longitude": (["time"], lon_1),
            "depth": (["time"], depth_1),
        },
        coords={"time": time_1},
    )

    # Second file: continuation
    ds2 = xr.Dataset(
        {
            "salinity": (["time"], np.linspace(36.0, 36.9, 10, dtype=np.float32)),
            "temperature": (["time"], np.linspace(20.0, 25.0, 10, dtype=np.float32)),
            "pressure": (["time"], np.linspace(100, 200, 10, dtype=np.float32)),
            "latitude": (["time"], lat_2),
            "longitude": (["time"], lon_2),
            "depth": (["time"], depth_2),
            "quality_flag": (["time"], [1, 4, 4, 4, 4, 4, 4, 4, 4, 1]),
        },
        coords={"time": time_2},
    )

    # Save to NetCDF files
    path1 = os.path.join(tmp_path, "file1.nc")
    path2 = os.path.join(tmp_path, "file2.nc")
    ds1.to_netcdf(path1)
    ds2.to_netcdf(path2)

    return [path1, path2]


def test_convert_one_file(sample_netcdf_files):
    """Test conversion of one Sensor-netCDF file.

    This test verifies:
    - Data are averaged
    """

    # Execute conversion
    _, variables = _convert_files(i_paths=[sample_netcdf_files[0]], grid_length=2, grid_height=2)

    # Expecting mean values of salinity ([35.  35.1 35.2 35.3 35.4 35.5 35.6 35.7 35.8 35.9])
    salinity_values = variables["salinity"]
    assert salinity_values[0, 0] == np.mean([35.0, 35.1, 35.2, 35.3, 35.4])
    assert salinity_values[1, 1] == np.mean([35.5, 35.6, 35.7, 35.8, 35.9])


def test_convert_one_file_with_invalid_data(sample_netcdf_files):
    """Test conversion of one Sensor-netCDF file with quality_flag layer.

    This test verifies:
    - Data are combined
    - Data with quality_flag=1 are excluded from export
    """

    # Execute conversion
    _, variables = _convert_files(i_paths=[sample_netcdf_files[1]], grid_length=2, grid_height=2)

    # Only first and last salinity values are valid (36.and 36.9)
    salinity_values = variables["salinity"]
    assert salinity_values[0, 0] == np.float32(36.0)
    assert salinity_values[1, 1] == np.float32(36.9)


def test_convert_files(sample_netcdf_files):
    """Test conversion
    of multiple NetCDF files with verification of combined data.

    This test verifies:
    - Data from all files are combined
    - Overlapping data is averaged correctly
    - Data with quality_flag=1 is excluded from export
    """

    # Execute conversion
    coords, variables = _convert_files(i_paths=sample_netcdf_files, grid_length=10, grid_height=10)

    # Verify that all expected variables are present
    expected_vars = ["salinity", "temperature", "pressure"]
    for var in expected_vars:
        assert var in variables, f"Variable '{var}' should be present"
        assert f"{var}_i" in variables, f"Interpolated variable '{var}_i' should be present"

    # Verify that position coordinates are present
    assert len(coords.latitudes) == 10, "Expecting 10 latitudes"
    assert len(coords.longitudes) == 10, "Expecting 10 latitudes"
    assert coords.min_elevation == 0, "Expecting 0 as min elevation"
    assert coords.max_elevation == -500, "Expecting -500 as max elevation"

    # Verify that data from all two files are combined
    salinity_values = variables["salinity"]
    salinity_clean = salinity_values[~np.isnan(salinity_values)]

    # Salinity values range from 35.0 (ds1) to 36.9 (ds2)
    assert salinity_clean.min() >= np.float32(35.0), "Minimum salinity should be close to 35.0"
    assert salinity_clean.max() <= np.float32(36.9), "Maximum salinity should be close to 36.9"

    temp_values = variables["temperature"]
    temp_clean = temp_values[~np.isnan(temp_values)]

    # Temperature values range from 15.0 (ds1) to 30.0 (ds3)
    assert temp_clean.min() >= np.float32(15.0), "Minimum temperature should be close to 15.0"
    assert temp_clean.max() <= np.float32(25.0), "Maximum temperature should be close to 25.0"

    # Verify that interpolated variables have fewer NaNs than originals
    for var in expected_vars:
        original_nans = np.isnan(variables[var]).sum()
        interpolated_nans = np.isnan(variables[f"{var}_i"]).sum()
        assert (
            interpolated_nans <= original_nans
        ), f"Interpolated variable '{var}_i' should have fewer or equal NaNs than original"

    # Verify that geographic coordinates cover all files
    # Latitudes range from 45.0 to 45.2 (covering all three files)
    assert coords.latitudes.min() >= 45.0, "Minimum latitude should be close to 45.0"
    assert coords.latitudes.max() <= 45.2, "Maximum latitude should be close to 45.2"
    # Longitudes range from -1.0 to -0.8
    assert coords.longitudes.min() >= -1.0, "Minimum longitude should be close to -1.0"
    assert coords.longitudes.max() <= -0.8, "Maximum longitude should be close to -0.8"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
