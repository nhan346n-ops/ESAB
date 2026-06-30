import locale
from typing import Any

import netCDF4 as nc


def open_nc_file(file_path: Any, mode="r", nc_format="NETCDF4") -> nc.Dataset:
    """
    Opens a netCDF file after checking the path encoding
    """
    _, encoding = locale.getlocale()
    try:
        return nc.Dataset(file_path, mode=mode, format=nc_format, encoding=encoding)
    except Exception as exc:
        raise IOError(f"Could not open netCDF file: {file_path}. Try closing other instances using it.") from exc


def filepath(dataset: nc.Dataset) -> str:
    """
    Return the file system path which was used to open/create the Dataset.
    """
    _, encoding = locale.getlocale()
    return dataset.filepath(encoding=encoding)
