import datetime
import os
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import osgeo.gdal as gdal
import scipy.ndimage as ndimage
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.transform.interpolation.gap_filling as gap_filling_process
import pyat.dtm.numba.merge_functions as nb
import pyat.dtm.utils.spatial_resolution_utils as ut
import pyat.dtm.utils.dtm_utils as dtmut
import pyat.utils.argument_utils as arg_util
import pyat.utils.netcdf_utils as nc_util
import pyat.utils.pyat_logger as log
from pyat.dtm.cdi.cdi_layer_util import check_undefined_cdi
from pyat.dtm.mask import compute_geo_mask


class AbstractMergeProcess(ABC):
    def _check_input_parameters(self):
        if not self.i_paths:
            raise SystemExit(
                "Useless process without input.\nStop the program.\nPlease enter 2 or more inputs with \n"
                "the option -i I_PATHS [I_PATHS ...], --i_paths I_PATHS [I_PATHS ...].\n"
            )

    def __init__(
        self,
        process_name,
        i_paths: list,
        coord: dict,
        o_path: str = None,
        overwrite=False,
        merged_layers: dict = None,
        mask: str = None,
        smoothing_border: int = 0,
        interpolate_border: bool = True,
        allow_undefined_cdi: bool = False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        self.process_name = process_name
        self.i_paths = i_paths
        self.overwrite = overwrite
        self.mask_files = arg_util.parse_list_of_files("mask", mask)
        self.smoothing_border = arg_util.parse_int("smoothing_border", smoothing_border)
        self.interpolate_border = interpolate_border
        self.allow_undefined_cdi = allow_undefined_cdi

        self.monitor = monitor

        self.logger = log.logging.getLogger(process_name)

        self.i_drivers: List[dtm_driver.DtmDriver] = []

        if not coord or not all(key in coord for key in ["north", "south", "west", "east"]):
            raise ValueError(
                f"Invalid parameter 'coord'. Expecting a dictionnary with keys 'north', 'south', 'west' and 'east'"
            )
        self.geobox = arg_util.parse_geobox("coord", coord)
        self.logger.debug(f"Process merge on: {self.geobox}")

        self._check_input_parameters()
        if o_path is None:
            self.o_path = os.path.join(os.path.dirname(self.i_paths[0]), self.process_name + DtmConstants.EXTENSION_NC)
        else:
            self.o_path = o_path

        self.merged_layers = [layer for (layer, value) in arg_util.parse_layers(merged_layers).items() if value]
        # Swap INTERPOLATION_FLAG and ELEVATION_NAME to process interpolation layer first
        # Useful for gap filling process
        if DtmConstants.INTERPOLATION_FLAG in self.merged_layers:
            index_inter_flag = self.merged_layers.index(DtmConstants.INTERPOLATION_FLAG)
            index_elevation = self.merged_layers.index(DtmConstants.ELEVATION_NAME)
            if index_inter_flag > index_elevation:
                self.merged_layers[index_inter_flag], self.merged_layers[index_elevation] = (
                    self.merged_layers[index_elevation],
                    self.merged_layers[index_inter_flag],
                )

        # Elevation array of erased values in case of smoothing
        self.elevation_erased_values = None

    def _compute_dims(self) -> Tuple[int, int]:
        """Compute col and row count."""
        row_count = dtmut.estimate_row(self.geobox.lower, self.geobox.upper, self.spatial_resolution_y)
        self.logger.info(f"Row count : {row_count}")
        col_count = dtmut.estimate_col(
            left_or_west=self.geobox.left, right_or_east=self.geobox.right, spatial_resolution=self.spatial_resolution_x
        )
        self.logger.info(f"Col count : {col_count}")
        return (col_count, row_count)

    def _get_mask(self) -> np.array:
        geo_transform = (
            self.geobox.left,
            self.spatial_resolution_x,
            0.0,
            self.geobox.upper,
            0.0,
            -self.spatial_resolution_y,
        )
        input_dataset = gdal.Open(f"NETCDF:{self.i_paths[0]}:{DtmConstants.ELEVATION_NAME}")

        return compute_geo_mask(
            mask_files=self.mask_files,
            x_size=self.col_count,
            y_size=self.row_count,
            geo_transform=geo_transform,
            projection=input_dataset.GetProjection(),
        )

    @abstractmethod
    def _process_layer(self, layer_name: str, geo_mask: np.ndarray, smoothing_mask: np.ndarray = None) -> None:
        pass

    @abstractmethod
    def _process_cdis(self, mask: np.array) -> None:
        pass

    @abstractmethod
    def process_global_data(self, mask):
        pass

    def _compute_smoothing_mask(self, geo_mask: np.ndarray) -> np.ndarray:
        """
        Return the mask array, where to smooth the elevations
        The result contains True for all cells to be processed by the smoothing operation
        """
        o_x = self.o_driver.get_x_axis()[:].data
        o_y = self.o_driver.get_y_axis()[:].data

        # Init smoothing mask
        smoothing_mask = np.full_like(geo_mask, False, dtype=bool)

        for i, i_driver in enumerate(reversed(self.i_drivers)):
            # elevations_mask, cell is True when elevation is present
            elevation_layer = i_driver[DtmConstants.ELEVATION_NAME][:]
            elevations_mask = np.logical_not(np.ma.getmaskarray(elevation_layer))
            i_x = i_driver.get_x_axis()[:].data
            i_y = i_driver.get_y_axis()[:].data
            o_elevations_mask = nb.merge_project(i_y, o_y, i_x, o_x, elevations_mask, False, geo_mask)

            if i == 0:  # lowest priority file
                all_elevations_mask = o_elevations_mask
            else:
                # Dilation : expand the shape of the elevation mask in all directions.
                dilation_mask = ndimage.binary_dilation(
                    o_elevations_mask, iterations=self.smoothing_border, brute_force=True
                )
                # Border : keep only the cells present in secondary files
                dilation_mask = np.logical_and(all_elevations_mask, dilation_mask)
                # Add new cells
                smoothing_mask = np.logical_or(smoothing_mask, dilation_mask)
                # Border : keep only the cells not present in current file
                smoothing_mask[o_elevations_mask] = False
                # Update secondary files elevations
                all_elevations_mask = np.logical_or(all_elevations_mask, o_elevations_mask)

        return smoothing_mask

    def _process_data(self) -> None:
        """Create variable and process it with the good method."""
        # Used for the log
        count = 0
        n = len(self.i_drivers[0].get_layers())
        self.monitor.set_work_remaining(n)

        # Get the mask
        mask = self._get_mask()
        self.process_global_data(mask)

        # Prepare the elevation interpolation by masking elevations in smoothing area
        smoothing_mask = None
        if self.smoothing_border > 0:
            smoothing_mask = self._compute_smoothing_mask(mask)

        # The first input file is the reference file
        for layer_name in self.merged_layers:
            # Merge CDI, CDI_INDEX together but after all layers
            if layer_name != DtmConstants.CDI_INDEX:
                # Create variable in the o_files[0].
                count += 1
                log.info_progress_layer(self.logger, "layer", layer_name, count, n)
                if layer_name == DtmConstants.ELEVATION_NAME and smoothing_mask is not None:
                    self._process_layer(layer_name, mask, smoothing_mask)
                else:
                    self._process_layer(layer_name, mask)
                if smoothing_mask is not None and layer_name == DtmConstants.ELEVATION_NAME:
                    # Invoke gap filling to smooth elevation at borders
                    if self.interpolate_border:
                        filling_mask = smoothing_mask.astype(np.uint8)
                        gap_filling_process.process(
                            self.o_driver, self.smoothing_border + 2, filling_mask, self.logger, 0, 2
                        )
                        # retrieve empty value (nan), contained
                        # They should have been filled, but the gap filling process have them fully inside the border area and interpolation failed
                        elevation = self.o_driver.dataset[layer_name][:].data

                        # values that should have been interpolated but are empty
                        mask_elevation = np.logical_and(smoothing_mask, np.isnan(elevation))
                        # by default we replace these values by the ones that were retained before erasure by the mask
                        elevation[mask_elevation] = self.elevation_erased_values[mask_elevation]
                        self.o_driver.dataset[layer_name][:] = elevation
                    else:
                        # Clear border, no interpolation required
                        elevation = self.o_driver.dataset[layer_name][:].data
                        elevation[smoothing_mask] = dtm_driver.get_missing_value(DtmConstants.ELEVATION_NAME)
                        self.o_driver.dataset[layer_name][:] = elevation

            self.monitor.worked(1)

        if DtmConstants.CDI_INDEX in self.merged_layers:
            # Merge CDI, CDI_INDEX together but after all layers
            log.info_progress_layer(self.logger, "layers", "cdi index & ref", count, n)
            self._process_cdis(mask)

    def __call__(self) -> None:
        """Main method of the class. Open files, then create dimensions, copy global attributes.
        After merge layer and copy variable attributes. Finally, close the files.
        """
        try:
            self.logger.info(f'Starting to {self.process_name} with {", ".join(self.i_paths)}.')
            begin = datetime.datetime.now()
            # Open input Files
            for i_path in self.i_paths:
                newDriver = dtm_driver.DtmDriver(i_path)
                newDriver.open()
                self.i_drivers.append(newDriver)
                if DtmConstants.ELEVATION_NAME in newDriver:
                    self.logger.info(f"Size: {newDriver[DtmConstants.ELEVATION_NAME].shape}")

            # Check if no file has undefined CDI
            for i_driver in self.i_drivers:
                check_undefined_cdi(i_driver.dataset, self.allow_undefined_cdi)

            # Check if all files have the same spatial resolution and projection.
            (
                self.spatial_reference,
                self.spatial_resolution_x,
                self.spatial_resolution_y,
            ) = ut.check_spatial_reso_and_projection(self.i_drivers)
            self.col_count, self.row_count = self._compute_dims()

            # Create and open output file
            self.o_driver = dtm_driver.DtmDriver(self.o_path)

            with self.o_driver.create_file(
                self.col_count,
                self.geobox.left,
                self.spatial_resolution_x,
                self.row_count,
                self.geobox.lower,
                self.spatial_resolution_y,
                self.spatial_reference,
                self.overwrite,
            ) as dataset:
                # History
                nc_util.set_history_attr(dataset, self.process_name, self.i_paths, append=False)
                self._process_data()

            end = datetime.datetime.now()
            self.logger.info(f"End of {self.process_name}, {end - begin} time elapsed.\n")

        except ValueError as e:
            self.logger.error(str(e))
        except FileExistsError as e:
            self.logger.error(
                f"{e.filename} already exists and overwrite not allowed (allow overwrite with option: '-ow --overwrite)"
            )
        except Exception:
            self.logger.error("An exception was thrown!", exc_info=True)

        finally:
            # close files
            for i_driver in self.i_drivers:
                i_driver.close()
            self._close()

    def _close(self):
        pass
