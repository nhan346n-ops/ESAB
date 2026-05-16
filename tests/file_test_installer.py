"""Download and install test files"""

import os
import pathlib
import urllib.request as req
import zipfile
from pathlib import Path

# version release for test path
release = "0.0.4"
destination_dir = Path(__file__).parent.parent / "data" / "external"
pyat_test_root_name = "pyat_test_file"
pyat_test_file_url = (
    f"https://gitlab.ifremer.fr/api/v4/projects/343/packages/generic/pyat_test_file/{release}/{pyat_test_root_name}.zip"
)
# To update pyat test files package registry, zip pyat_test_file directory and upload this archive using :
# curl --user "PRIVATE-TOKEN:<gitlabtoken>" --upload-file "pyat_test_file.zip" "https://gitlab.ifremer.fr/api/v4/projects/343/packages/generic/pyat_test_file/0.0.X/pyat_test_file.zip?select=package_file"
# Don't forget to update release version in the URL


def get_test_path() -> Path:
    """
    Downloads (1st call) and retrieves full test file path given relative path.
    On first call, if it does not already exist, pyat_test_file will be downloaded from pyat package registry
    and unzipped under pyat/data/external. Data are then kept for further tests, until being manually deleted.
    """
    unzip_dst = Path(f"{destination_dir}") / f"{release}"
    if not os.path.isdir(unzip_dst):
        dst = f"{destination_dir}/{pyat_test_root_name}_{release}.zip"
        req.urlretrieve(pyat_test_file_url, dst)

        with zipfile.ZipFile(dst, "r") as zip_ref:
            zip_ref.extractall(path=unzip_dst)

        os.remove(dst)
    else:
        print(f"Directory {unzip_dst} already exists")
    return pathlib.Path(unzip_dst, pyat_test_root_name)
