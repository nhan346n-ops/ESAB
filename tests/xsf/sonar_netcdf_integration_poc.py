""""
Test sonar netcdf sub project integration
"""

import os
import tempfile as tmp

import netCDF4 as nc
import sonar_netcdf.sonar_groups as sonar_definition


def test_netcdf_basic():
    """sample test and use of sonar_groups.py file and methods"""
    # filename = tmp.mktemp(dir="d:/tmp" , suffix=".nc")
    filename = tmp.mktemp(suffix=".nc")
    print(f"creating fake xsf file {filename}")

    with nc.Dataset(filename, mode="w") as file:
        root_structure = sonar_definition.RootGrp()
        root = root_structure.create_group(file)
        sonar_structure = sonar_definition.SonarGrp()

    # if no exception we consider to test valid
    assert True

    os.remove(filename)
