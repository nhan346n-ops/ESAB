# pylint:disable=no-member

import shutil
from typing import List, Optional

import xarray
import rioxarray  # Don't delete : mandatory for rioxarray to work
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_standard_constants as DtmConstants
from pyat.dtm import dtm_driver
from pyat.dtm.transform.interpolation.coronis.interpolation import interpolate_dtms


class InterpolateProcess:
    """
    Interpolate process
    """

    def __init__(
        self,
        i_paths: List[str],
        o_paths: List[str],
        overwrite: bool = False,
        masks: Optional[str] = None,
        cdi_interpolation_algo: str = "closest_neighbor",  # or most_common_neighbor
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor.
        """
        interpolate_dtms(
            i_paths=i_paths,
            o_paths=o_paths,
            interpolation_algo=self._interpolates_one_dtm,
            cdi_interpolation_algo=cdi_interpolation_algo,
            overwrite=overwrite,
            areas=masks,
            monitor=monitor,
        )

    def _interpolates_one_dtm(self, i_path: str, o_path: str, _) -> None:
        """
        use of xarray and rioxarray to compute the missing elevations
        """
        rioxarray.show_versions() # Called to keep the import of rioxarray

        shutil.copyfile(src=i_path, dst=o_path)
        with dtm_driver.open_dtm(o_path, mode="r+") as o_dtm_driver:
            crs_description = o_dtm_driver.dtm_file.spatial_reference.ExportToProj4()
            xdata_array = xarray.DataArray(o_dtm_driver[DtmConstants.ELEVATION_NAME][:], dims=["y", "x"])
            xdata_array.rio.write_crs(crs_description, inplace=True)
            xdata_array.rio.write_nodata(dtm_driver.get_missing_value(DtmConstants.ELEVATION_NAME), inplace=True)
            interpolated_elevations = xdata_array.rio.interpolate_na(method="linear").to_numpy()
            o_dtm_driver[DtmConstants.ELEVATION_NAME][:] = interpolated_elevations
