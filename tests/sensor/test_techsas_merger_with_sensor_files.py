"""
Unit tests for Sensor-netCDF merger.
"""

import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytechsas.sensor.sensor_constant as sc
import pytechsas.sensor.sensor_csv_to_netcdf_converter as csv_conv
import pytest
import xarray as xr

from pyat.sensor.techsas_merger import TechsasMerger


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def basic_csv_content_1():
    """CSV with timestamp column."""
    return """timestamp;lat;lon;temp
1705315800;48.5;-3.2;15.3
1705315860;48.51;-3.21;15.4
1705315920;48.52;-3.22;15.5"""


@pytest.fixture
def basic_csv_content_2():
    return """timestamp;lat;lon;temp
1705315960;48.3;-3.23;15.6
1705316020;48.54;-3.24;15.7
1705316080;48.55;-3.25;15.8"""


@pytest.fixture
def basic_csv_description():
    """Basic CSV description configuration."""
    return {
        "delimiter": ";",
        "decimal_point": ".",
        "header_line": 1,
        "first_data_line": 2,
        "columns": {
            "0": {"content_type": "TIMESTAMP", "format": "s"},
            "1": {"content_type": "LATITUDE"},
            "2": {"content_type": "LONGITUDE"},
            "3": {"content_type": "VALUE", "format": "FLOAT"},
        },
    }


class TestTechsasMerger:
    """Integration tests for the TechsasMerger process."""

    def test_simple_merge(self, temp_dir, basic_csv_content_1, basic_csv_content_2, basic_csv_description):
        # Generating first Sensor-netCDF file
        csv_path_1 = temp_dir / "test_1.csv"
        netcdf_path_1 = temp_dir / "test_1.nc"
        csv_path_1.write_text(basic_csv_content_1)
        csv_conv.convert(str(csv_path_1), str(netcdf_path_1), basic_csv_description)
        assert netcdf_path_1.exists()

        # Generating second Sensor-netCDF file
        csv_path_2 = temp_dir / "test_2.csv"
        netcdf_path_2 = temp_dir / "test_2.nc"
        csv_path_2.write_text(basic_csv_content_2)
        csv_conv.convert(str(csv_path_2), str(netcdf_path_2), basic_csv_description)
        assert netcdf_path_2.exists()

        # Merging
        out_merged_path = temp_dir / "out.nc"
        merger = TechsasMerger(i_paths=[str(netcdf_path_1), str(netcdf_path_2)], o_paths=[str(out_merged_path)])
        merger()
        assert out_merged_path.exists()

        # Verify NetCDF content
        ds = xr.open_dataset(out_merged_path)

        # Verify attributes
        assert sc.DATE_CREATED in ds.attrs
        assert ds.attrs[sc.CONVENTIONS] == sc.CONVENTIONS_VALUE
        assert ds.attrs[sc.CONVENTION_AUTHORITY] == sc.CONVENTION_AUTHORITY_VALUE

        # Verify variables
        npt.assert_array_equal(
            ds[sc.SENSOR_VAR_TIME].values.astype("datetime64[s]").astype("int64"),
            [1705315800, 1705315860, 1705315920, 1705315960, 1705316020, 1705316080],
        )
        npt.assert_array_equal(
            ds[sc.SENSOR_VAR_LATITUDE].values, np.array([48.5, 48.51, 48.52, 48.3, 48.54, 48.55], dtype="float64")
        )
        npt.assert_array_equal(
            ds[sc.SENSOR_VAR_LONGITUDE].values, np.array([-3.2, -3.21, -3.22, -3.23, -3.24, -3.25], dtype="float64")
        )
        npt.assert_array_equal(ds["temp"].values, np.array([15.3, 15.4, 15.5, 15.6, 15.7, 15.8], dtype="float32"))

        ds.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
