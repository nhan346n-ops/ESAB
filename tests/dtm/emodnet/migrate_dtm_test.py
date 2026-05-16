#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile
import unittest

import netCDF4 as nc
from osgeo import osr

import pyat.common.geo_file as gf
import pyat.dtm.dtm_legacy_constants as srcConstants
import pyat.dtm.dtm_standard_constants as targetConstants
from pyat.dtm.convert.migrate_dtm_process import DTMMigrate
from tests.generator.netcdf3_dtm_generator import Netcdf3DtmGenerator
from tests.tools.netcdf import comparator


def validate_dimensions(src: nc.Dataset, target: nc.Dataset, spatial_reference):
    """compare all dimensions between an old dtm and a upgraded one"""
    if spatial_reference.IsGeographic():
        assert (
            src.dimensions[srcConstants.DIM_COLUMNS].size == target.dimensions[targetConstants.DIM_LON].size
        ), "dimensions differs between files"
        assert (
            src.dimensions[srcConstants.DIM_LINE].size == target.dimensions[targetConstants.DIM_LAT].size
        ), "dimensions differs between files"
    else:
        assert (
            src.dimensions[srcConstants.DIM_COLUMNS].size == target.dimensions[targetConstants.ABSCISSA_NAME].size
        ), "dimensions differs between files"
        assert (
            src.dimensions[srcConstants.DIM_LINE].size == target.dimensions[targetConstants.ORDINATE_NAME].size
        ), "dimensions differs between files"


def validate_attributes(target):
    """check only that the history is filled by the migration script"""
    old_history = "2015-11-22T12:30:00Z Netcdf3DtmGenerator by unit_test_generator Generated for unit test"
    history = target.getncattr(targetConstants.HISTORY_ATTRIB_NAME)
    assert f"Upgraded from {target.title}.dtm, " + old_history in history


def validate_variables(src, target, spatial_reference):
    if spatial_reference.IsGeographic():
        comparator.compare_variables_data(src, srcConstants.VARIABLE_LINE, target, targetConstants.LAT_NAME)
        comparator.compare_variables_data(src, srcConstants.VARIABLE_COLUMN, target, targetConstants.LON_NAME)
    else:
        comparator.compare_variables_data(src, srcConstants.VARIABLE_LINE, target, targetConstants.ORDINATE_NAME)
        comparator.compare_variables_data(src, srcConstants.VARIABLE_COLUMN, target, targetConstants.ABSCISSA_NAME)

    comparator.compare_cdi_variables(src, srcConstants.VARIABLE_CDI_INDEX, target, targetConstants.CDI)
    comparator.compare_variables_data(src, srcConstants.VARIABLE_DEPTH, target, targetConstants.ELEVATION_NAME)
    comparator.compare_variables_data(src, srcConstants.VARIABLE_MIN_SOUNDING, target, targetConstants.ELEVATION_MIN)
    comparator.compare_variables_data(src, srcConstants.VARIABLE_MAX_SOUNDING, target, targetConstants.ELEVATION_MAX)
    comparator.compare_variables_data(src, srcConstants.VARIABLE_STDEV, target, targetConstants.STDEV)
    comparator.compare_variables_data(src, srcConstants.VARIABLE_VSOUNDINGS, target, targetConstants.VALUE_COUNT)
    comparator.compare_variables_data(
        src, srcConstants.VARIABLE_INTERPOLATION_FLAG, target, targetConstants.INTERPOLATION_FLAG
    )
    comparator.compare_variables_data(src, srcConstants.VARIABLE_CDI, target, targetConstants.CDI_INDEX)
    comparator.compare_variables_data(
        src, srcConstants.VARIABLE_ACCROSS_ANGLE, target, targetConstants.MAX_ACCROSS_ANGLE
    )
    comparator.compare_variables_data(
        src, srcConstants.VARIABLE_MAX_ACROSS_DISTANCE, target, targetConstants.MAX_ACROSS_DISTANCE
    )
    comparator.compare_variables_data(
        src, srcConstants.VARIABLE_MIN_ACROSS_DISTANCE, target, targetConstants.MIN_ACROSS_DISTANCE
    )
    comparator.compare_variables_data(src, srcConstants.VARIABLE_REFLECTIVITY, target, targetConstants.BACKSCATTER)


class TestMigrateDtm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")

    def setUp(self):
        self.generator = Netcdf3DtmGenerator()
        # Output File Path
        self.o_path = tempfile.mktemp(suffix=".nc")

    def test_migrate_wsg84_dtm(self):
        """
        generates a old dtm in WGS_84 spatial reference and migrates it to .nc dtm format
        """
        self.__migrate_dtm(gf.SR_WGS_84)

    def test_migrate_pseudo_mercator_dtm(self):
        """
        generates a old dtm in mercator projection and migrates it to .nc dtm format
        """
        self.__migrate_dtm(gf.SR_PSEUDO_MERCATOR)

    def test_migrate_mercator_a_dtm(self):
        """
        generates a old dtm in mercator projection and migrates it to .nc dtm format
        """
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4("+proj=merc +lon_0=2.0")
        assert spatial_reference.GetAttrValue("PROJECTION") == osr.SRS_PT_MERCATOR_1SP
        self.__migrate_dtm(spatial_reference)

    def test_migrate_transverse_mercator_dtm(self):
        """
        generates a old dtm in transverse mercator projection and migrates it to .nc dtm format
        """
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4("+proj=tmerc +lon_0=2.0 +lat_0=2.0")
        assert spatial_reference.GetAttrValue("PROJECTION") == osr.SRS_PT_TRANSVERSE_MERCATOR
        self.__migrate_dtm(spatial_reference)

    def test_migrate_mercator_b_dtm(self):
        """
        generates a old dtm in mercator projection and migrates it to .nc dtm format
        """
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4("+proj=merc +lat_ts=1.0 +lon_0=2.0")
        assert spatial_reference.GetAttrValue("PROJECTION") == osr.SRS_PT_MERCATOR_2SP
        self.__migrate_dtm(spatial_reference)

    def test_migrate_aea_dtm(self):
        """
        generates a old dtm in  projection and migrates it to .nc dtm format
        """
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4("+proj=aea +lat_1=1.0 +lat_2=3.0")
        assert spatial_reference.GetAttrValue("PROJECTION") == osr.SRS_PT_ALBERS_CONIC_EQUAL_AREA
        self.__migrate_dtm(spatial_reference)

    def test_migrate_aeqd_dtm(self):
        """
        generates a old dtm in aeqd projection and migrates it to .nc dtm format
        """
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4("+proj=aeqd +lat_ts=2 +lat_0=2 +lon_0=2.0")
        assert spatial_reference.GetAttrValue("PROJECTION") == osr.SRS_PT_AZIMUTHAL_EQUIDISTANT
        self.__migrate_dtm(spatial_reference)

    def test_migrate_laea_dtm(self):
        """
        generates a old dtm in laea projection and migrates it to .nc dtm format
        """
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4("+proj=laea +lat_0=2 +lon_0=2.0")
        assert spatial_reference.GetAttrValue("PROJECTION") == osr.SRS_PT_LAMBERT_AZIMUTHAL_EQUAL_AREA
        self.__migrate_dtm(spatial_reference)

    def test_migrate_lcc_1_dtm(self):
        """
        generates a old dtm in lcc 1 projection and migrates it to .nc dtm format
        """
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4("+proj=lcc +lat_1=2 +lat_0=2 +lon_0=2")
        assert spatial_reference.GetAttrValue("PROJECTION") == osr.SRS_PT_LAMBERT_CONFORMAL_CONIC_1SP
        self.__migrate_dtm(spatial_reference)

    def test_migrate_lcc_2_dtm(self):
        """
        generates a old dtm in lcc 2 projection and migrates it to .nc dtm format
        """
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4("+proj=lcc +lon_0=2 +lat_1=1.5 +lat_2=2.5")
        assert spatial_reference.GetAttrValue("PROJECTION") == osr.SRS_PT_LAMBERT_CONFORMAL_CONIC_2SP
        self.__migrate_dtm(spatial_reference)

    def test_migrate_cea_dtm(self):
        """
        generates a old dtm in cea projection and migrates it to .nc dtm format
        """
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4("+proj=cea +lon_0=2 +lat_ts=2")
        assert spatial_reference.GetAttrValue("PROJECTION") == osr.SRS_PT_CYLINDRICAL_EQUAL_AREA
        self.__migrate_dtm(spatial_reference)

    def test_migrate_ortho_dtm(self):
        """
        generates a old dtm in ortho projection and migrates it to .nc dtm format
        """
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4("+proj=ortho +lon_0=2 +lat_0=2")
        assert spatial_reference.GetAttrValue("PROJECTION") == osr.SRS_PT_ORTHOGRAPHIC
        self.__migrate_dtm(spatial_reference)

    def test_migrate_stere_dtm(self):
        """
        generates a old dtm in stere projection and migrates it to .nc dtm format
        """
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4("+proj=stere +lon_0=2 +lat_ts=2")
        assert spatial_reference.GetAttrValue("PROJECTION") == osr.SRS_PT_STEREOGRAPHIC
        self.__migrate_dtm(spatial_reference)

    def test_migrate_polar_stere_dtm(self):
        """
        generates a old dtm in polar stere projection and migrates it to .nc dtm format
        """
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4("+proj=stere +lon_0=2 +lat_0=90 +lat_ts=2")
        assert spatial_reference.GetAttrValue("PROJECTION") == osr.SRS_PT_POLAR_STEREOGRAPHIC
        self.__migrate_dtm(spatial_reference)

    def __migrate_dtm(self, spatial_reference):
        """
        generate from a src dtm a migration and compare the result contents with the source content
        """

        # indicate if we make some test on real dtm or on the one in gitlab
        self.i_path = self.generator.initialize_file(spatial_reference)
        print("\nConverting ", self.i_path, " to ", self.o_path)

        process = DTMMigrate(i_paths=[self.i_path], o_paths=[self.o_path])
        process()

        # comparing file contents
        with nc.Dataset(self.i_path) as src, nc.Dataset(self.o_path) as target:
            validate_dimensions(src, target, spatial_reference)
            validate_attributes(target)
            validate_variables(src, target, spatial_reference)

    def tearDown(self):
        if self.o_path and os.path.exists(self.o_path):
            os.remove(self.o_path)
        if self.i_path and os.path.exists(self.i_path):
            os.remove(self.i_path)

    @classmethod
    def tearDownClass(cls):
        print(f"End of {cls.__name__}.")


if __name__ == "__main__":
    unittest.main()
