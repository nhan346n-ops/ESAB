#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.numba.merge_functions as nb
from pyat.dtm.merge.abstract_merge import AbstractMergeProcess


class MergeFillProcess(AbstractMergeProcess):
    """Class Merge which is used for the merge between (2 or more) dtm files."""

    def __init__(
        self,
        i_paths: list,
        coord: dict,
        o_path: str = None,
        overwrite=False,
        layers: dict = None,
        mask: str = None,
        smoothing_border: int = 0,
        interpolate_border: bool = True,
        allow_undefined_cdi: bool = False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """By default, process the merge simple, on all layers, with the same spatial resolution,
        the coordinates than the reference file (first file).

        Arguments:
            i_paths {list} -- List of dtm file input path.
            o_path {str} -- List of dtm file output path.

        Keyword Arguments:
            params {str} -- merge_type , layers, spatial_reso, coord

        Raises:
            ValueError: Raise an exception if the type of merge is different of simple, or simple.
            ValueError: Raise an exception if the spatial_reso is not a int, float or None.
        """
        super().__init__(
            process_name="merged_fill",
            i_paths=i_paths,
            o_path=o_path,
            overwrite=overwrite,
            merged_layers=layers,
            coord=coord,
            mask=mask,
            smoothing_border=smoothing_border,
            interpolate_border=interpolate_border,
            allow_undefined_cdi=allow_undefined_cdi,
            monitor=monitor,
        )

    def process_global_data(self, mask):
        # Nothing to do in merge fill process
        pass

    def _process_layer(self, layer_name: str, geo_mask: np.ndarray, smoothing_mask: np.ndarray = None) -> None:
        """For each file, project the layer in first. Then process it.

        Arguments:
            name {str} -- Name of the layer.
        """
        # copy variable attributes all at once via dictionary

        self.o_driver.add_layer(layer_name=layer_name)

        # Initialisation
        temp_buffer = self.o_driver[layer_name][:].data
        if smoothing_mask is not None:
            temp_buffer_erased_values = np.full_like(temp_buffer, fill_value=np.nan)
            # we assume that we are processing elevation layer if a masked is passed as argument
            # if not raise an exception
            if layer_name != DtmConstants.ELEVATION_NAME:
                raise NotImplementedError("Assumption failed: smoothing only applies to elevation layer")

        o_y = self.o_driver.get_y_axis()[:].data
        o_x = self.o_driver.get_x_axis()[:].data

        reference_dtm = True
        for i_dtm_driver in self.i_drivers:
            if layer_name not in i_dtm_driver:
                # we need to create a temporary buffer with default values for this variables
                # At least we expect to have one elevation layer
                i_elevation = i_dtm_driver[DtmConstants.ELEVATION_NAME]
                i_elevation_data = i_elevation[:].data
                i_data = np.empty(shape=i_elevation_data.shape, dtype=dtm_driver.LAYER_TYPES[layer_name])
                missing_value = dtm_driver.get_missing_value(layer_name)
                i_data.fill(missing_value)
                self.o_driver.fill_default_layer_buffer(layer_name, i_data, i_elevation_data, i_elevation._FillValue)
            else:
                i_data = i_dtm_driver[layer_name][:].data
                missing_value = i_dtm_driver[layer_name]._FillValue

            # Project points.
            i_y = i_dtm_driver.get_y_axis()[:].data
            i_x = i_dtm_driver.get_x_axis()[:].data

            if reference_dtm:
                # keeps the entire reference file outside the geographical area
                full_geo_mask = np.full_like(geo_mask, fill_value=1)
                i_data = nb.merge_project(i_y, o_y, i_x, o_x, i_data, missing_value, full_geo_mask)
            else:
                i_data = nb.merge_project(i_y, o_y, i_x, o_x, i_data, missing_value, geo_mask)

            # Mask elevations in smoothing area
            if smoothing_mask is not None:
                rejected = np.full_like(i_data, fill_value=np.nan)
                rejected[smoothing_mask] = i_data[smoothing_mask]

                i_data[smoothing_mask] = missing_value
                # retain and merge erased elevation data
                temp_buffer_erased_values = nb.merge_fill(temp_buffer_erased_values, rejected, missing_value)

            temp_buffer = nb.merge_fill(temp_buffer, i_data, missing_value)
            reference_dtm = False

        self.o_driver[layer_name][:] = temp_buffer
        if smoothing_mask is not None:
            self.elevation_erased_values = temp_buffer_erased_values

    def _process_cdis(self, mask: np.array) -> None:
        """Merge cdi. Project layer then process it."""
        cdi = DtmConstants.CDI
        cdi_index = DtmConstants.CDI_INDEX

        # Copy attributes
        self.o_driver.add_layer(DtmConstants.CDI_INDEX)
        self.o_driver.add_layer(DtmConstants.CDI)

        # Initialisation
        temp_cdi = self.o_driver[cdi][:]
        temp_cdi = cdi_util.trim_string_array(temp_cdi)
        temp_index = self.o_driver[cdi_index][:].data

        o_y = self.o_driver.get_y_axis()[:].data
        o_x = self.o_driver.get_x_axis()[:].data

        reference_dtm = True
        for i_driver in self.i_drivers:
            if DtmConstants.CDI in i_driver:
                i_data = i_driver[cdi_index][:].data
                m_val = i_driver[cdi_index]._FillValue

                # Project points.
                i_y = i_driver.get_y_axis()[:].data
                i_x = i_driver.get_x_axis()[:].data

                if reference_dtm:
                    # keeps the entire reference file outside the geographical area
                    full_geo_mask = np.full_like(mask, fill_value=1)
                    i_data = nb.merge_project(i_y, o_y, i_x, o_x, i_data, m_val, full_geo_mask)
                else:
                    i_data = nb.merge_project(i_y, o_y, i_x, o_x, i_data, m_val, mask)

                temp_cdi, temp_index = self.__process_cdi(
                    temp_cdi, i_driver[cdi][:][i_driver[cdi][:] != ""], temp_index, i_data, m_val
                )

            reference_dtm = False

        cdi_util.reset_all_cdi_id(self.o_driver.dataset)
        for i, name in enumerate(temp_cdi):
            # VLEN can be only accessed one at a time
            self.o_driver[cdi][i] = name

        self.o_driver[cdi_index][:] = temp_index
        cdi_util.clean_cdi(self.o_driver.dataset)

    def __process_cdi(self, o_cdi: np.array, i_cdi: np.array, o_index: np.array, i_index: np.array, m_val):
        """Copy cdi value into output file if the cdi value doesn't exist in the output file.
        In function of the type of merge, process it with the good function numba. The method
        count each point of cdi.

        Arguments:
            o_cdi {[type]} -- Layer CDI of the merge file.
            i_cdi {[type]} -- Layer CDI of the input file.
            o_index {[type]} -- Layer CDI_INDEX of the merge file.
            i_index {[type]} -- Layer CDI_INDEX of the input file.
            m_val {[type]} -- invalid value.
        """
        for i_count, cdi in enumerate(i_cdi):
            # On regarde si le cdi n'est pas présent dans le fichier en sortie.
            if not cdi in o_cdi:
                o_cdi = np.append(o_cdi, cdi)
            index_used = int(np.where(o_cdi == cdi)[0])
            o_index = nb.merge_fill_cdi_index(i_index, o_index, i_count, index_used, m_val)

        return o_cdi, o_index
