import tempfile as tmp
import unittest

import netCDF4
import numpy as np

import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
from pyat.dtm.dtm_driver import LAYER_TYPES, get_missing_value
from pyat.utils.netcdf_utils import DEFAULT_COMPRESSION_LIB


class TestCDILayerUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Start of {cls.__name__}.")

    def test_trim_cdi(self):
        trimed_list = cdi_util.trim_string_array(["A", "", "B", "", ""])
        assert (trimed_list == np.array(["A", "", "B"])).all()

    def _create_vlen_with_3_values(self, file):
        fileout = netCDF4.Dataset(file, "w")
        unlimited_dimension = fileout.createDimension(DtmConstants.DIM_CDI, size=None)
        dimension_x = fileout.createDimension("X", size=6)
        dimension_y = fileout.createDimension("Y", size=2)
        v = fileout.createVariable(DtmConstants.CDI, str, dimensions=DtmConstants.DIM_CDI)
        layerType = LAYER_TYPES[DtmConstants.CDI_INDEX]
        missing = get_missing_value(DtmConstants.CDI_INDEX)
        v_index = fileout.createVariable(DtmConstants.CDI_INDEX, layerType, dimensions=("Y", "X"), fill_value=missing,
                                         compression=DEFAULT_COMPRESSION_LIB)
        v[0] = "A"
        v[1] = ""
        v[2] = "C"
        v[3] = "B"
        v[4] = "C"
        v[5] = ""
        values = np.array([[0, 1, 2, 3, 4, missing], [0, 3, 4, 1, missing, missing]], dtype=layerType)
        v_index[:] = values
        fileout.close()

    def test_modify_vlen(self):
        file = tmp.mktemp(suffix=".nc")
        self._create_vlen_with_3_values(file)
        fileout = netCDF4.Dataset(file, "a")
        missing = get_missing_value(DtmConstants.CDI_INDEX)
        cdi_util.clean_cdi(
            fileout,
        )

        cdi = fileout.variables[DtmConstants.CDI][:]
        assert (np.array_equal(cdi, np.array(["A", "B", "C", "", "", ""])) or np.array_equal(cdi, np.array(["A", "B", "C", "", ""])))

        v_index = fileout.variables[DtmConstants.CDI_INDEX]
        v_index.set_auto_mask(False)
        v_index = v_index[:]
        compare = [a == b for a, b in zip([[0, 3, 2, 1, 2, missing], [0, 1, 2, 3, missing, missing]], v_index)]
        assert all(compare[0]) and all(compare[1])

        fileout.close()

    def test_cdi_with_empty_value(self):
        cdi_array = ["", "AA", "BB", "DD", "", "DD"]
        new_ids, index_map = cdi_util.clean_double(np.array(cdi_array))
        assert np.array_equal(new_ids, np.array(["AA", "BB", "DD", ""]))
        assert np.array_equal(index_map, np.array([3, 0, 1, 2, 3, 2]))

    def test_cdi_(self):
        cdi_array = ["CC", "AA", "BB", "DD", "CC", "DD"]
        new_ids, index_map = cdi_util.clean_double(np.array(cdi_array))
        # check that cdi ids are unique and sorted
        assert np.array_equal(new_ids, np.array(["AA", "BB", "CC", "DD"]))
        assert np.array_equal(index_map, np.array([2, 0, 1, 3, 2, 3]))

    @classmethod
    def tearDownClass(cls):
        print(f"End of {cls.__name__}.")
