#! /usr/bin/env python3
# coding: utf-8

import datetime
import os
from os import PathLike
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from osgeo import osr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from pyproj import Transformer, crs

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.analyse.dtm_expected_stdev as expected_stdev
import pyat.dtm.analyse.dtm_quality_indicator as quality_indicator
import pyat.dtm.dtm_standard_constants as DTM
import pyat.dtm.transform.interpolation.gap_filling as gap_filling_process
import pyat.dtm.utils.process_utils as process_util
import pyat.sounder.sounder_driver as sounder_driver
import pyat.sounder.sounder_driver_factory as sounder_driver_factory
import pyat.utils.argument_utils as arg_util
import pyat.utils.netcdf_utils as nc_util
import pyat.utils.pyat_logger as log
from pyat.common.geo_file import SR_WGS_84
from pyat.dtm.dtm_gridder import DtmGridder
from pyat.dtm.mask import compute_geo_mask_from_dtm
from pyat.utils import signal

CHUNK_SIZE_IN_DETECTION_COUNT = 7_000_000


class SounderToDtmExporter:
    """
    Utility class to export an Mbg or Xsf file as a dtm (netcdf4 format)
    """

    @property
    def geobox(self) -> arg_util.Geobox:
        return self._geobox

    @geobox.setter
    def geobox(self, geobox: arg_util.Geobox) -> None:
        self._geobox = geobox

    def __init__(
        self,
        i_paths: list,
        o_paths: list,
        target_spatial_reference: str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        target_resolution: float = 1.0 / 3600.0,
        coord: Optional[Dict] = None,
        overwrite: bool = False,
        layers: Optional[List[str]] = None,
        valid_sounds_only: bool = False,
        min_elevation: float = float("-inf"),
        max_elevation: float = float("inf"),
        min_sounds: int = 0,
        spatial_antialiasing: bool = True,
        gap_filling: bool = False,
        mask_size: int = 3,
        mask: Optional[List[str]] = None,
        quality_indicator: bool = False,
        stdev_csv_path: PathLike | None = None,
        cdi: Optional[Dict[str, str]] = None,
        title: str = None,
        institution: str = None,
        source: str = None,
        references: str = None,
        comment: str = None,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Constructor.

        Parameters
        ----------
            i_paths :
                paths of the input file to convert
            o_paths
                paths of resulting dtm file
            quality_indicator :
                when True, calculates the quality indicators tif file for the resulting DTM
            stdev_csv_path :
                when present, calculates the expected STDEV by beam angle in a tif file for the resulting DTM
        """
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.layers = arg_util.parse_list_of_str(layers)

        self.target_spatial_reference = target_spatial_reference
        self.spatial_resolution = float(target_resolution)
        self.coord = coord

        if not coord is None:
            self.geobox = arg_util.parse_geobox("coord", coord)
            self.geobox.spatial_reference = osr.SpatialReference()
            self.geobox.spatial_reference.ImportFromProj4(self.target_spatial_reference)

        self.overwrite = overwrite
        self.valid_sounds_only = valid_sounds_only
        self.min_elevation = arg_util.parse_float("min_elevation", min_elevation, float("-inf"))
        self.max_elevation = arg_util.parse_float("max_elevation", max_elevation, float("inf"))
        self.min_sounds = arg_util.parse_int("min_sounds", min_sounds, 0)

        self.spatial_antialiasing = str.upper(str(spatial_antialiasing)) == "TRUE"
        self.gap_filling = str.upper(str(gap_filling)) == "TRUE"
        self.mask_size = arg_util.parse_int("mask_size", mask_size, default=3, min_value=3, max_value=31)
        self.mask_files = arg_util.parse_list_of_files("mask", mask) if self.gap_filling else []

        self.cdi = cdi

        self.title = title
        self.institution = institution
        self.source = source
        self.references = references
        self.comment = comment

        self.monitor = monitor
        self.logger = log.logging.getLogger(self.__class__.__name__)

        self.quality_indicator = quality_indicator
        self.stdev_csv_path = None
        if stdev_csv_path:
            if os.path.exists(stdev_csv_path):
                self.stdev_csv_path = stdev_csv_path
            else:
                self.logger.warning(f"File {stdev_csv_path} not found. Argument 'stdev_csv_path' ignored")

    def __evaluate_geobox(
        self,
        i_sounder_driver: sounder_driver.SounderDriver,
    ) -> None:
        spatial_reference = crs.CRS.from_proj4(self.target_spatial_reference)
        transform: Optional[Transformer] = None
        if spatial_reference.is_projected:
            transform = Transformer.from_crs(
                crs.CRS.from_epsg(4326),
                spatial_reference,
                always_xy=True,
            )

        # Iterates all beam's positions to compute geobounds
        x_min = y_min = float("inf")
        x_max = y_max = float("-inf")
        # Process beams by chunk
        chunk_size_in_swath_count = int(CHUNK_SIZE_IN_DETECTION_COUNT / i_sounder_driver.sounder_file.beam_count)

        # pylint:disable = unpacking-non-sequence
        for longitudes, latitudes in i_sounder_driver.iter_beam_positions(chunk_size_in_swath_count):
            if transform is None:
                x_min = min(x_min, np.nanmin(longitudes))
                x_max = max(x_max, np.nanmax(longitudes))
                y_min = min(y_min, np.nanmin(latitudes))
                y_max = max(y_max, np.nanmax(latitudes))
            else:
                xs, ys = transform.transform(longitudes, latitudes, radians=False)
                x_min = min(x_min, np.nanmin(xs))
                x_max = max(x_max, np.nanmax(xs))
                y_min = min(y_min, np.nanmin(ys))
                y_max = max(y_max, np.nanmax(ys))

        self.logger.info(f"DTM bounds [{x_min}, {x_max}] x [{y_min}, {y_max}]")
        if self.geobox is None:
            self.geobox = arg_util.Geobox(y_max, y_min, x_min, x_max)
            self.geobox.spatial_reference = osr.SpatialReference()
            self.geobox.spatial_reference.ImportFromProj4(self.target_spatial_reference)
        else:
            self.geobox.extend(y_max, y_min, x_min, x_max)

    def __read_real_depth(
        self,
        i_sounder_driver: sounder_driver.SounderDriver,
        from_swath: int,
        to_swath: int,
        validities: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Read depth values, depths are in fcs coordinate system, ie if set relative to an absolute surface reference
        Mask unvalid soundings if valid_sounds_only is True
        """
        depths = i_sounder_driver.read_fcs_depths(from_swath, to_swath)
        if validities is not None:
            depths[~validities] = np.nan
        # now return elevations which are positive up
        return depths

    def __read_reflectivities(
        self,
        i_sounder_driver: sounder_driver.SounderDriver,
        from_swath: int,
        to_swath: int,
        validities: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Read reflectivities in sounder file.
        Mask unvalid soundings if valid_sounds_only is True
        """
        result = i_sounder_driver.read_reflectivities(from_swath, to_swath)
        if validities is not None:
            result[~validities] = np.nan
        return result

    def __read_across_distances(
        self,
        i_sounder_driver: sounder_driver.SounderDriver,
        from_swath: int,
        to_swath: int,
        validities: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Read across distances in sounder file.
        Mask unvalid soundings if valid_sounds_only is True
        """
        result = i_sounder_driver.read_across_distances(from_swath, to_swath)
        if validities is not None:
            result[~validities] = np.nan
        return result

    def __read_across_angles(
        self,
        i_sounder_driver: sounder_driver.SounderDriver,
        from_swath: int,
        to_swath: int,
        validities: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Read across angles in sounder file.
        Mask unvalid soundings if valid_sounds_only is True
        """
        result = i_sounder_driver.read_across_angles(from_swath, to_swath)
        if validities is not None:
            result[~validities] = np.nan
        return result

    def __read_validities(
        self, i_sounder_driver: sounder_driver.SounderDriver, from_swath: int, to_swath: int
    ) -> Optional[np.ndarray]:
        """
        Read validity flags if need be.
        """
        if self.valid_sounds_only:
            return i_sounder_driver.read_validity_flags(from_swath, to_swath)
        return None

    def __make_gridder(self, o_dtm_driver: dtm_driver.DtmDriver) -> DtmGridder:
        """Create and prepare the Dtm gridder"""
        dtm_gridder = DtmGridder(
            o_dtm_driver,
            geobox=self.geobox,
            spatial_resolution=self.spatial_resolution,
            depth_factor=-1.0,
            average_elevations=True,
            spatial_antialiasing=self.spatial_antialiasing,
        )
        # Optional layer
        for layer in [DTM.BACKSCATTER, DTM.MIN_ACROSS_DISTANCE, DTM.MAX_ACROSS_DISTANCE]:
            if layer in self.layers:
                dtm_gridder.add_layer(layer)
        if DTM.MAX_ACCROSS_ANGLE in self.layers or self.stdev_csv_path is not None:
            # Add across angles if layer or process dtm_expected_stdev is required
            dtm_gridder.add_layer(DTM.MAX_ACCROSS_ANGLE)

        # Layer computed automatically
        dtm_gridder.deal_with(DTM.VALUE_COUNT)
        for layer in [DTM.ELEVATION_MIN, DTM.ELEVATION_MAX, DTM.STDEV, DTM.FILTERED_COUNT]:
            if layer in self.layers:
                dtm_gridder.deal_with(layer)

        if self.min_elevation != float("-inf") or self.max_elevation != float("inf"):
            dtm_gridder.restrict_elevations(self.min_elevation, self.max_elevation)

        dtm_gridder.initialize_dtm_file(
            title=self.title,
            institution=self.institution,
            source=self.source,
            references=self.references,
            comment=self.comment,
        )
        nc_util.set_history_attr(o_dtm_driver.dataset, self.__class__.__name__, self.i_paths)

        return dtm_gridder

    def __project_coords(
        self, dtm_gridder: DtmGridder, xs: np.ndarray, ys: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return dtm_gridder.project_coords(
            xs,
            ys,
            # SR_WGS_84 will force the transformation or lonlat to x/y coordinates
            SR_WGS_84 if self.geobox.spatial_reference.IsProjected() else None,
        )

    def __finalize_gridder(self, dtm_gridder: DtmGridder) -> ArrayLike | None:
        """
        Ask gridder to finalize the DTM.

        Parameters
        ----------
            dtm_gridder :
                current gridder.

        Returns
        -------
            the layer of beam angles if it has been generated by the gridder. None otherwise
        """
        # Delete isolated values
        if self.min_sounds > 0:
            dtm_gridder.reset_cell(self.min_sounds)

        #
        beam_angles = None
        if DTM.MAX_ACCROSS_ANGLE in dtm_gridder.layer_data:
            if DTM.MAX_ACCROSS_ANGLE not in self.layers:
                # Beam angles is present but not required in the DTM.
                #  It is removed before the finalization
                beam_angles = dtm_gridder.remove_layer(DTM.MAX_ACCROSS_ANGLE)
            else:
                beam_angles = dtm_gridder.layer_data[DTM.MAX_ACCROSS_ANGLE]

        # Finally, Write the Dtm
        dtm_gridder.finalize_dtm()

        return beam_angles

    def __fill_gap(self, path: str) -> None:
        mask = compute_geo_mask_from_dtm(path, self.mask_files)
        with dtm_driver.open_dtm(path, "r+") as o_dtm_driver:
            gap_filling_process.process(o_dtm_driver, self.mask_size, mask, self.logger, 0, 2)

    def __export(
        self, i_sounder_driver: sounder_driver.SounderDriver, dtm_gridder: DtmGridder, cdi: Optional[str], monitor
    ) -> None:
        """
        Launch the export
        """
        self.logger.info(f"Starting to convert {i_sounder_driver.sounder_file.file_path}")
        monitor.set_work_remaining(2)
        self.logger.info(
            f"Input file has {i_sounder_driver.sounder_file.swath_count} swaths and {i_sounder_driver.sounder_file.beam_count} beams per swath"
        )

        # Process layers
        i_swath = 0

        # Process beams by chunk
        chunk_size_in_swath_count = int(CHUNK_SIZE_IN_DETECTION_COUNT / i_sounder_driver.sounder_file.beam_count)

        for xs, ys in i_sounder_driver.iter_beam_positions(chunk_size_in_swath_count):
            self.logger.info(f"Process swaths {i_swath} - {i_swath + xs.shape[0] - 1}")
            x, y = self.__project_coords(dtm_gridder, xs, ys)

            # Flags to mask unvalid values
            validities = self.__read_validities(i_sounder_driver, i_swath, i_swath + xs.shape[0])

            # Depths
            elevations = self.__read_real_depth(i_sounder_driver, i_swath, i_swath + xs.shape[0], validities)
            dtm_gridder.grid_elevations(x, y, elevations, cdi)

            columns = np.floor(x).astype(int)
            rows = np.floor(y).astype(int)

            # Reflectivity
            if DTM.BACKSCATTER in self.layers:
                backscatter = self.__read_reflectivities(i_sounder_driver, i_swath, i_swath + xs.shape[0], validities)
                dtm_gridder.grid_backscatter(
                    columns=columns,
                    rows=rows,
                    elevations=elevations,
                    backscatter=signal.db_to_amplitude(backscatter.astype(np.float64)),
                )

            # Across distances
            if DTM.MIN_ACROSS_DISTANCE in self.layers or DTM.MAX_ACROSS_DISTANCE in self.layers:
                dtm_gridder.grid_min_max(
                    min_layer_name=DTM.MIN_ACROSS_DISTANCE,
                    max_layer_name=DTM.MAX_ACROSS_DISTANCE,
                    values=self.__read_across_distances(i_sounder_driver, i_swath, i_swath + xs.shape[0], validities),
                    columns=columns,
                    rows=rows,
                )

            # Across angles
            if DTM.MAX_ACCROSS_ANGLE in dtm_gridder.layer_desc:
                dtm_gridder.grid_min_max(
                    min_layer_name=None,
                    max_layer_name=DTM.MAX_ACCROSS_ANGLE,
                    values=self.__read_across_angles(i_sounder_driver, i_swath, i_swath + xs.shape[0], validities),
                    columns=columns,
                    rows=rows,
                )

            # Standard deviation
            if DTM.STDEV in self.layers:
                dtm_gridder.grid_standard_deviation(columns, rows, elevations)

            i_swath += xs.shape[0]

        monitor.done()

    def __infer_cdi(self, sounder_file_path: str) -> Optional[str]:
        if self.cdi is None or len(self.cdi) == 0:
            return None
        sounder_file_name = os.path.basename(sounder_file_path)
        if sounder_file_name in self.cdi:
            self.logger.info(f"CDI of {sounder_file_name} is {self.cdi[sounder_file_name]}")
            return self.cdi[sounder_file_name]
        return None

    def __merge_sounder_to_dtm(self) -> None:
        """
        Export all sounder files in an unique Dtm
        """
        self.logger.info("Merging all sounder files in one Dtm file")
        begin = datetime.datetime.now()
        self.monitor.set_work_remaining(len(self.i_paths))
        file_in_error = []

        # Geo bounds
        if self.geobox is None:
            for sounder_file_path in self.i_paths:
                with sounder_driver_factory.open_sounder(sounder_file_path) as i_sounder_driver:
                    self.__evaluate_geobox(i_sounder_driver)

        beam_angles = None
        with dtm_driver.open_dtm(self.o_paths[0], "w") as o_dtm_driver:
            # Create and prepare the Dtm gridder
            dtm_gridder = self.__make_gridder(o_dtm_driver)

            for sounder_file_path in self.i_paths:
                sub_monitor = self.monitor.split(1)
                try:
                    # Export each mbg into the Dtm
                    with sounder_driver_factory.open_sounder(sounder_file_path) as i_sounder_driver:
                        self.__export(i_sounder_driver, dtm_gridder, self.__infer_cdi(sounder_file_path), sub_monitor)

                except ValueError as e:
                    file_in_error.append(sounder_file_path)
                    self.logger.error(str(e))
                except Exception as error:
                    file_in_error.append(sounder_file_path)
                    self.logger.error(f"An exception was thrown : {str(error)}", exc_info=True, stack_info=True)

            if len(file_in_error) == 0:
                beam_angles = self.__finalize_gridder(dtm_gridder)

        if self.gap_filling:
            self.logger.info("Starting interpolation process (Fill Gap)")
            self.__fill_gap(self.o_paths[0])

        if self.quality_indicator:
            self.logger.info("Starting computing the quality indicators")
            quality_indicator.computes(i_paths=[self.o_paths[0]], overwrite=self.overwrite)

        if self.stdev_csv_path and beam_angles is not None:
            self.logger.info("Starting expected STDEV by beam angle")
            expected_stdev.computes(
                i_paths=[self.o_paths[0]],
                beam_angles=beam_angles,
                stdev_csv_path=self.stdev_csv_path,
                overwrite=self.overwrite,
            )
        process_util.log_result(self.logger, begin, file_in_error)

    def __export_sounder_to_dtm(self) -> None:
        """
        Export each sounder file in one Dtm file
        """
        begin = datetime.datetime.now()
        self.monitor.set_work_remaining(len(self.i_paths))
        file_in_error = []
        for sounder_file_path, o_path in zip(self.i_paths, self.o_paths):
            sub_monitor = self.monitor.split(1)
            try:
                if os.path.exists(o_path):
                    if not self.overwrite:
                        self.logger.warning(f"{o_path} skipped (already exists)")
                        continue
                    os.remove(o_path)

                beam_angles = None
                with (
                    sounder_driver_factory.open_sounder(sounder_file_path) as i_sounder_driver,
                    dtm_driver.open_dtm(o_path, "w") as o_dtm_driver,
                ):
                    if self.coord is None:
                        # Export over the whole mbg
                        self.geobox = None
                        self.__evaluate_geobox(i_sounder_driver)

                    # Create and prepare the Dtm gridder
                    dtm_gridder = self.__make_gridder(o_dtm_driver)
                    self.__export(i_sounder_driver, dtm_gridder, self.__infer_cdi(sounder_file_path), sub_monitor)
                    beam_angles = self.__finalize_gridder(dtm_gridder)

                if self.gap_filling:
                    self.logger.info("Starting interpolation process (Fill Gap)")
                    self.__fill_gap(o_path)

                if self.quality_indicator:
                    self.logger.info("Starting computing the quality indicators")
                    quality_indicator.computes(i_paths=[o_path], overwrite=self.overwrite)

                if self.stdev_csv_path and beam_angles is not None:
                    self.logger.info("Starting expected STDEV by beam angle")
                    expected_stdev.computes(
                        i_paths=[o_path],
                        beam_angles=beam_angles,
                        stdev_csv_path=self.stdev_csv_path,
                        overwrite=self.overwrite,
                    )

            except ValueError as e:
                file_in_error.append(sounder_file_path)
                self.logger.error(str(e))
            except Exception as error:
                file_in_error.append(sounder_file_path)
                self.logger.error(f"An exception was thrown : {str(error)}", exc_info=True, stack_info=True)

        process_util.log_result(self.logger, begin, file_in_error)

    def __call__(self) -> None:
        """Run method."""
        if len(self.i_paths) > 1 and len(self.o_paths) == 1:
            self.__merge_sounder_to_dtm()
        else:
            self.__export_sounder_to_dtm()
