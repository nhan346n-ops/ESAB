#! /usr/bin/env python3
# coding: utf-8

import os
import tempfile as tmp
import unittest

import netCDF4 as nc
import numpy as np
from osgeo import osr

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import tests.directory_utils as dir_util
from pyat.dtm.transform.update_boundingbox import ReprojectProcess
from tests.generator.dtm_generator import DtmGenerator


class TestReproject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")
        cls.directory = dir_util.get_test_directory()
        generator = DtmGenerator(cls.directory)
        cls.path = generator.create_pattern_smoothing(value=20, value_2=30)
        cls.path2 = generator.create_pattern(value=20, pair_impair=1, line_col=1, number=2, allValue=False)

    def test_reproject_2(self):
        # Parameters
        i_path = dir_util.get_test_directory() + "/raw/reset_cell_multi_cdi.nc"
        epsg = 4326
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(epsg)
        proj4 = dst_srs.ExportToProj4()

        # Read Metadata
        i_driver = dtm_driver.DtmDriver(i_path)
        dtm_file = i_driver.dtm_file
        # create a reprojection with the same bounding box : {'east': 3.5, 'north': 3.5, 'south': 0.5, 'west': 0.5}
        coord = {"north": dtm_file.north, "south": dtm_file.south, "east": dtm_file.east, "west": dtm_file.west}

        output = tmp.mktemp(suffix="tmp.nc")
        params = {"i_paths": [i_path], "o_paths": [output], "coord": coord}

        # Process
        reproject = ReprojectProcess(**params)
        reproject()

        # Verify
        try:
            with nc.Dataset(output) as o_data, nc.Dataset(dtm_file.file_path) as i_data:
                # compare global attributes
                for attribute_name in i_data.__dict__:
                    self.assertTrue(
                        attribute_name in o_data.__dict__, msg=f"Missing attribute {attribute_name} in output file"
                    )
                    if attribute_name not in {"history", "source", "references", "Conventions"}:  # history is different
                        self.assertEqual(
                            i_data.getncattr(attribute_name),
                            o_data.getncattr(attribute_name),
                            msg=f"Attribute {attribute_name} content differs  {str(i_data.getncattr(attribute_name))} vs {str(o_data.getncattr(attribute_name))}",
                        )
                    # if input_att!="History":
                    # compare contents
                # compare dimensions, should be the same since we made a reprojection with the same bounding box
                for dimension_name in i_data.dimensions:
                    self.assertTrue(
                        dimension_name in o_data.dimensions, msg=f"Missing attribute {dimension_name} in output file"
                    )
                    self.assertEqual(
                        len(i_data.dimensions[dimension_name]),
                        len(i_data.dimensions[dimension_name]),
                        msg=f"len differ for dimension {dimension_name}",
                    )

                for variable_name in i_data.variables:
                    print(f"Checking variable {variable_name} ")
                    self.assertTrue(
                        variable_name in o_data.variables, msg=f"Missing variable {variable_name} in output file"
                    )
                    # check variables attributes
                    i_variable = i_data.variables[variable_name]
                    o_variable = o_data.variables[variable_name]
                    for attribute_name in i_variable.__dict__:
                        if attribute_name != "standard_name":  # Attribute removed
                            self.assertTrue(
                                attribute_name in o_variable.__dict__,
                                msg=f"Missing attribute {attribute_name} in output file",
                            )
                            i_att = i_variable.getncattr(attribute_name)
                            o_att = o_variable.getncattr(attribute_name)
                            if isinstance(i_att, (float, np.float32)):
                                np.testing.assert_array_equal(i_att, o_att)
                            elif isinstance(i_att, np.ndarray):
                                np.testing.assert_almost_equal(i_att, o_att, decimal=6)
                            else:
                                self.assertEqual(
                                    i_att,
                                    o_att,
                                    msg=f"Attribute {attribute_name} content differs  {str(i_att)} vs {str(o_att)}",
                                )
                    # check variable contents

                    i_content = i_variable[:]
                    o_content = o_variable[:]
                    if variable_name not in {"crs"}:
                        if variable_name == "cdi_reference":
                            # variable differs since CDI content can be compressed
                            i_content = cdi_util.trim_string_array(i_content)
                            o_content = cdi_util.trim_string_array(o_content)
                            np.testing.assert_array_equal(
                                i_content, o_content, err_msg=f"content differs for {variable_name}"
                            )
                        else:
                            v = np.abs(i_content - o_content)
                            nmax = np.nanmax(v)
                            assert nmax.all() < 10e-5

        finally:
            os.remove(output)

    @classmethod
    def tearDownClass(cls):
        print(f"End of {cls.__name__}.")
        os.remove(cls.path)
        os.remove(cls.path2)


if __name__ == "__main__":
    unittest.main()
