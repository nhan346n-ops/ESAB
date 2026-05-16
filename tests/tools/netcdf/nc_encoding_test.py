# Test encoding option of nc.Dataset constructor

import os
import tempfile as tmp

from pyat.utils import nc_encoding


def test_open_nc_file():
    with tmp.TemporaryDirectory() as temp_dir:
        # Creates an empty file
        xsf = tmp.mktemp(suffix=".xsf.nc", dir=temp_dir)
        with nc_encoding.open_nc_file(xsf, mode="w"):
            assert os.path.exists(xsf)

        with nc_encoding.open_nc_file(xsf):
            print(f"{xsf} opened successfully")

        # Rename with accent
        xsf_with_utf8 = os.path.join(temp_dir, "test_bépo.xsf")
        os.rename(xsf, xsf_with_utf8)
        assert os.path.exists(xsf_with_utf8)

        with nc_encoding.open_nc_file(xsf_with_utf8):
            print(f"{xsf_with_utf8} opened successfully")
