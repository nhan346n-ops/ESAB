#! /usr/bin/env python3
# coding: utf-8

import tempfile as tmp

import netCDF4 as nc
from sonar_netcdf.sonar_groups import RootGrp, SonarGrp

from pyat.sonarscope.model.sonar_metadata import SonarFileMetaData
from pyat.xsf import xsf_driver


def create_fake_file(file_name: str):
    """create basic structure with a few fields for test"""
    # create root Node
    with nc.Dataset(file_name, mode="w") as file:
        root_structure = RootGrp()
        root = root_structure.create_group(file)

        sonar_structure = SonarGrp()
        sonar = sonar_structure.create_group(root)


def test_read():
    # create a fake file
    file = tmp.mktemp(suffix="_unittest.nc")
    create_fake_file(file_name=file)
    xsf = xsf_driver.XsfDriver(file_path=file)
    try:
        xsf.open()
    except KeyError:
        # we expect a key error due to almost empty xsf file
        pass
    except ValueError:
        # we expect a key error due to almost empty xsf file
        pass

    metadata = SonarFileMetaData(xsf_dataset=xsf)
    metadata.read()
    assert "sonar_model" in metadata.metadata
