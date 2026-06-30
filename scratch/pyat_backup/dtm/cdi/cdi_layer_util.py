from typing import List

import netCDF4 as nc
import numpy as np
import scipy.interpolate
from scipy import ndimage

import pyat.dtm.dtm_driver as driver
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.numba.set_cdi_functions as nb
import pyat.utils.pyat_logger as log
from pyat.utils import nc_encoding
from pyat.utils.string_utils import trim_string_array

logger = log.logging.getLogger("cdi_layer_util")


def clean_double(old_cdi_names: List[str]):
    """
    remove doubles in a list of cdi, order the list and put empty strings at the end

    return uniques, inverse where
        uniques is a list of unique names
        inverse an array allowing to map old index to new ones
    """
    if len(old_cdi_names) == 0:
        return old_cdi_names, []
    uniques, inverse = np.unique(old_cdi_names, return_inverse=True)
    # result are sorted, which means that if the list of string contains an empty string
    # it will be located at the beginning. We want to move it at the end
    if uniques[0] == "":
        uniques = np.roll(uniques, -1)  # move empty string to the end
        inverse = inverse - 1  # shift all index by one
        inverse[inverse == -1] = len(uniques) - 1  # reset index to empty string to the end

    return uniques, inverse


def reset_all_cdi_id(dtm_file: nc.Dataset):
    _reset_all_cdi_id(dtm_file)


def _reset_all_cdi_id(dtm_file: nc.Dataset):
    """reset all cdi id in variable DtmConstants.CDI, ie set all values to an empty string"""
    old_cdi_names = dtm_file[DtmConstants.CDI][:]
    # trim to avoid to reset last values to "" where they already have those values
    old_cdi_names = trim_string_array(old_cdi_names)
    for index, name in enumerate(old_cdi_names):
        dtm_file[DtmConstants.CDI][index] = ""


def _clean_multiple_cdi_entries(dtm_file: nc.Dataset) -> None:
    """Parse a netcdf dtm dataset, remove duplicate entries in DtmConstants.CDI and update accordingly cdi_index layer"""
    old_cdi_names = dtm_file[DtmConstants.CDI][:]
    new_ids, index_map = clean_double(old_cdi_names)

    # reset all values of cdi
    _reset_all_cdi_id(dtm_file)

    # copy new values
    for index, name in enumerate(new_ids):
        # VLEN can be only accessed one at a time
        dtm_file[DtmConstants.CDI][index] = name
    missing = driver.get_missing_value(DtmConstants.CDI_INDEX)
    updated_values = nb.remap_cdi_index(dtm_file[DtmConstants.CDI_INDEX][:].data, index_map, missing)
    dtm_file[DtmConstants.CDI_INDEX][:] = updated_values


def clean_cdi(dataset: nc.Dataset) -> None:
    """
    Remove unused CDI id; shift index values and remove entry in DtmConstants.CDI
    Warning does not remove doubles CDI entry
    """
    if not DtmConstants.CDI_INDEX in dataset.variables:
        return

    # check if we have double entries:
    index_names = dataset[DtmConstants.CDI][:]

    # no cdi , exit
    if len(index_names) == 0:
        return

    # remove trailing empty strings
    index_names = trim_string_array(index_names)
    # do we have doubles
    uniques = np.unique(index_names)
    if any(index_names == "") or len(uniques) != len(index_names):
        # we at least got an empty CDI or a double, clean all and update indexes
        _clean_multiple_cdi_entries(dataset)

    index_values = dataset[DtmConstants.CDI_INDEX]
    index_values_used = np.unique(dataset[DtmConstants.CDI_INDEX])
    if np.ma.is_masked(index_values_used):
        index_values_used = index_values_used[~index_values_used.mask].data

    # remove negative index values
    index_values_used = index_values_used[index_values_used >= 0].data

    # now we remove values that are not referenced, each new id will

    # clean up CDI names
    old_cdi_names = dataset[DtmConstants.CDI][:]
    _reset_all_cdi_id(dataset)

    for o_index, i_index in enumerate(index_values_used):
        # VLEN can be only accessed one at a time
        dataset[DtmConstants.CDI][o_index] = old_cdi_names[i_index]

    # clean up CDI index, ie apply new indexes
    # create a map from old index to new index
    missing = driver.get_missing_value(DtmConstants.CDI_INDEX)
    if len(index_values_used) > 0:
        max_index = np.max(index_values_used)
        index_map = np.empty(max_index + 1, dtype=np.int32)
        for index, value in enumerate(index_values_used):
            index_map[value] = index
        updated_values = nb.remap_cdi_index(index_values[:].data, index_map, missing)
        dataset[DtmConstants.CDI_INDEX][:] = updated_values
    else:
        dataset[DtmConstants.CDI_INDEX][:] = missing


def update_cdi(o_dtm_driver, mask: np.ndarray):
    """
    update cdi_index of the dtm file where cells of mask are True
    """
    if DtmConstants.CDI_INDEX in o_dtm_driver:
        cdi_index = o_dtm_driver[DtmConstants.CDI_INDEX][:]
        labeled_mask, num_features = ndimage.label(mask)
        # browse each label but 0 (0 represents elevations in original file)
        for num_feature in range(1, num_features + 1):
            # keep only the hole of label num_feature
            feature_mask = np.where(labeled_mask == num_feature, True, False)
            # Dilation of the hole to reach some cells with CDI
            dilation_mask = np.logical_not(ndimage.binary_dilation(feature_mask))

            # Keep only CDI indexes over the hole
            cdi_on_feature = np.ma.masked_array(cdi_index, dilation_mask)
            # Statistics of the CDI index
            cdi, nb = np.unique(cdi_on_feature, return_counts=True)
            # Ignore cells without CDI
            nb[cdi.mask] = -1
            # Set the most encountered CDI index
            cdi_index[feature_mask] = cdi[np.argmax(nb)]

        o_dtm_driver[DtmConstants.CDI_INDEX][:] = cdi_index


def update_with_closest_cdi(o_dtm_driver, mask):
    """
    update cdi_index of the dtm file
    mask of cells to recompute
    """
    if DtmConstants.CDI_INDEX in o_dtm_driver:
        missing = driver.get_missing_value(DtmConstants.CDI_INDEX)
        i_cdi_index = o_dtm_driver[DtmConstants.CDI_INDEX][:]
        o_cdi_index = o_dtm_driver[DtmConstants.CDI_INDEX][:]

        if DtmConstants.INTERPOLATION_FLAG in o_dtm_driver:
            interpolated_mask = o_dtm_driver[DtmConstants.INTERPOLATION_FLAG][:] == 1
        else:
            interpolated_mask = np.zeros_like(i_cdi_index)

        # exclude interpolated cells
        i_cdi_index[interpolated_mask] = missing

        # interpolate on mask
        source_coords = np.nonzero(i_cdi_index != missing)
        source_values = i_cdi_index[source_coords]
        dest_coords = np.nonzero(mask)
        dest_values = scipy.interpolate.griddata(source_coords, source_values, dest_coords, method="nearest")
        # apply new interpolated values on mask coords
        o_cdi_index[dest_coords] = dest_values

        # write result
        o_dtm_driver[DtmConstants.CDI_INDEX][:] = o_cdi_index


# pylint: disable=singleton-comparison
def check_undefined_cdi(dataset: nc.Dataset, allow_undefined_cdi: bool = True) -> bool:
    """
    Check if all elevation cells have a CDI, and if all cells with CDI have elevation.
    If this is not the case, log the differences between the two layers.

    When parameter allow_undefined_cdi is set to False, differences are log and an exception is raised.

    Returns True if all elevation cells have a CDI.
    """
    if DtmConstants.ELEVATION_NAME in dataset.variables and DtmConstants.CDI_INDEX in dataset.variables:
        cdi_index = dataset[DtmConstants.CDI_INDEX][:]
        elev = dataset[DtmConstants.ELEVATION_NAME][:]

        # check cells with elevation without CDI
        if (cdi_index.mask != elev.mask).any():
            message = f"CDI issue detected in {nc_encoding.filepath(dataset)} : "

            # Informs about differences
            elev_without_cdi = ((elev.mask == False) & (cdi_index.mask == True)).sum()
            if elev_without_cdi > 0:
                message += f"{elev_without_cdi} cells with elevation where CDI is missing."

            cdi_without_elev = ((elev.mask == True) & (cdi_index.mask == False)).sum()
            if cdi_without_elev > 0:
                message += f"{cdi_without_elev} cells with CDI where elevation is missing."

            # If undefined cdi are not allowed : raise an error.
            if allow_undefined_cdi:
                logger.warning(message)
            else:
                message += " (set option 'allow undefined CDI' to ignore this error)"
                raise ValueError(message)
            return False

    # If no CDI Layer : return True.
    return True
