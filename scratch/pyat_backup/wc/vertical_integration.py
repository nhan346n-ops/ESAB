# pylint:disable=no-member
import os
import time
from typing import Dict, List, Optional

import numpy as np
import sonarnative
from osgeo import osr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.utils.pyat_logger as log
from pyat.tiff import tiff_gridder
from pyat.utils import argument_utils
from pyat.utils.signal import db_to_energy, energy_to_db
from pyat.wc.utils.filter import apply_filters
from pyat.wc.utils.statistics import get_biggest_echoes_count


class VerticalIntegration:
    """
    Vertical integration of backscatters
    """

    @property
    def geobox(self) -> argument_utils.Geobox:
        return self._geobox

    @geobox.setter
    def geobox(self, geobox: argument_utils.Geobox) -> None:
        self._geobox = geobox

    def __init__(
        self,
        i_paths: List[str],
        o_paths: List[str],
        monitor: ProgressMonitor = DefaultMonitor,
        target_resolution: float = 1.0 / 3600.0,
        coord: Optional[Dict] = None,
        filters: Optional[str] = None,
        enable_normalization: bool = False,
        normalization_offset: float = 0.0,
        overwrite: bool = False,
    ):
        """
        Constructor.
        """
        self.logger = log.logging.getLogger(VerticalIntegration.__name__)
        self.i_paths: List[str] = i_paths
        self.o_paths: List[str] = o_paths
        self.monitor = monitor
        self.spatial_resolution = argument_utils.parse_float("target_resolution", target_resolution)
        self.coord = coord
        self.json_filters = filters
        self.enable_normalization = enable_normalization
        self.normalization_offset: float = argument_utils.parse_float("normalization_offset", normalization_offset)
        self.overwrite = overwrite

        if coord is not None:
            self.geobox = argument_utils.parse_geobox("coord", coord)
            self.geobox.spatial_reference = osr.SpatialReference()
            self.geobox.spatial_reference.ImportFromProj4("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

    def __call__(self):
        if len(self.i_paths) > 1 and len(self.o_paths) == 1:
            # Case of a merge
            self._convert(self.i_paths, self.o_paths[0], self.monitor)
        else:
            # Case of a single conversion
            self.monitor.set_work_remaining(len(self.i_paths))
            for in_xsf, out_tiff in zip(self.i_paths, self.o_paths):
                self._convert([in_xsf], out_tiff, self.monitor.split(1))

    def _convert(self, in_xsfs: List[str], out_tiff: str, monitor: ProgressMonitor):
        """
        Initiate a gridder
        Invoke the spatialization process on each xsf and fill the gridder with the echoes
        Finalize the gridder
        Write the resulting TIFF
        """
        self.logger.info("Read input files")

        if os.path.exists(out_tiff) and not self.overwrite:
            self.logger.error(
                "File already exists and overwrite not allowed (allow overwrite with option : '-o --overwrite)"
            )
            return

        gridder = tiff_gridder.TiffGridder(
            out_tiff,
            geobox=self.geobox,
            spatial_resolution=self.spatial_resolution,
            monitor=monitor,
        )

        swaths_wanted = 1
        echoes_count = get_biggest_echoes_count(in_xsfs, swaths_wanted)

        # memory reservation
        mem_echos = sonarnative.MemEchos(echoes_count)

        # file initialization
        gridder.initialize_tiff_file(float)

        for input_file in in_xsfs:
            spatializer = sonarnative.open_spatializer(input_file, -1, True)
            # setup filters
            apply_filters(self.json_filters, spatializer)
            # setup image processing
            if self.enable_normalization:
                native_param = sonarnative.RangeNormalizationParameter(True, self.normalization_offset)
                sonarnative.apply_range_normalization_signal_processing(spatializer, native_param)

            swath_count = spatializer.get_swath_count()
            self.logger.info(f"read file {input_file}")

            swath_list = list(range(swath_count))
            try:
                for i in swath_list[::swaths_wanted]:
                    # use of memory
                    # arg: file / swath index / number of swath wanted
                    sonarnative.spatialize_in_memory(spatializer, i, swaths_wanted, mem_echos)

                    longitudes = mem_echos.longitude
                    latitudes = mem_echos.latitude
                    echos = mem_echos.echo

                    # if longitude is nan, delete it
                    echos = echos[np.logical_not(np.isnan(longitudes))]
                    latitudes = latitudes[np.logical_not(np.isnan(longitudes))]
                    longitudes = longitudes[np.logical_not(np.isnan(longitudes))]

                    # transform reflectivity to natural energy
                    # transform reflectivity in db to natural energy
                    echos = db_to_energy(value=echos)

                    # First, compute columns and rows index
                    columns, rows = gridder.project_coords(longitudes, latitudes)
                    # Then, process values
                    gridder.grid_average(columns, rows, echos)

            finally:
                # release memory
                sonarnative.close_spatializer(spatializer)
                monitor.worked(1)

        # return values in db
        gridder.map_file = energy_to_db(gridder.map_file)

        # tiff finalized
        gridder.finalize_tiff()
        self.logger.info("file created")
        monitor.done()


if __name__ == "__main__":
    date = int(time.time())
    xsf_to_tiff = VerticalIntegration(
        i_paths=["list", "of", "input_files.xsf.nc"],
        o_paths=[f"path\\to\\outfile\\xsf_to_geotiff_{date}.tiff"],
        monitor=DefaultMonitor,
        target_resolution=0.00002777777778,
        coord={
            "north": -12.805393949090519,
            "south": -12.823830586993873,
            "west": 45.35544657735835,
            "east": 45.367953410896924,
        },
    )
    xsf_to_tiff()
