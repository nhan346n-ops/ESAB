import math
from typing import List, Tuple

from osgeo import osr

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.utils.dtm_utils as dtmut
import pyat.utils.pyat_logger as log
from pyat.dtm.utils.mercator_utils import is_same_mercator


def check_spatial_reso_and_projection(
    dtm_drivers: List[dtm_driver.DtmDriver],
) -> Tuple[osr.SpatialReference, float, float]:
    """Check if all files have the same spatial resolution and the same projection.
    Else, raise an error.
    """
    logger = log.logging.getLogger("emodnet.utils")

    ref_dtm_file = dtm_drivers[0].dtm_file
    # Save spatial resolution and compare to the others.
    spatial_reference_ref = ref_dtm_file.spatial_reference

    # Check projections
    for driver in dtm_drivers[1:]:
        if (
            not spatial_reference_ref.IsSame(driver.dtm_file.spatial_reference)
            # Accepts all geographic projections (make sure "+proj=longlat +ellps=WGS84" is same as "+proj=longlat +datum=WGS84")
            and not (spatial_reference_ref.IsGeographic() and driver.dtm_file.spatial_reference.IsGeographic())
            # Accepts equivalent mercator projections (to ensure compatibility between old and new DTM files)
            and not is_same_mercator(spatial_reference_ref, driver.dtm_file.spatial_reference)
        ):
            raise ValueError(
                f"Input files have different projection {ref_dtm_file.file_path} differs from {driver.dtm_file.file_path} ({spatial_reference_ref.ExportToProj4()}) vs {driver.dtm_file.spatial_reference.ExportToProj4()} "
            )

    spatial_resolution_x_ref = ref_dtm_file.spatial_resolution_x
    spatial_resolution_y_ref = ref_dtm_file.spatial_resolution_y
    # Check spatial resolutions
    for driver in dtm_drivers[1:]:
        # some spatial resolution are considered as not the same even if egal at 1e-8 precision (mm resolution)
        spatial_resolution_x = round(driver.dtm_file.spatial_resolution_x, dtmut.DTM_PRECISION_DECIMAL_COUNT)
        if abs(spatial_resolution_x - spatial_resolution_x_ref) > dtmut.DTM_PRECISION:
            raise ValueError(
                f"Not same spatial resolution for file {driver.dtm_file.file_path} on x axis : {spatial_resolution_x} vs {spatial_resolution_x_ref}"
            )
        spatial_resolution_y = round(driver.dtm_file.spatial_resolution_y, dtmut.DTM_PRECISION_DECIMAL_COUNT)
        if abs(spatial_resolution_y - spatial_resolution_y_ref) > dtmut.DTM_PRECISION:
            raise ValueError(
                f"Not same spatial resolution for file {driver.dtm_file.file_path} on y axis : {spatial_resolution_y} vs {spatial_resolution_y_ref}"
            )

    # try to recompute ideal resolution as a fraction of arcmin
    spatial_resolution_x_ref = estimate_resolution_rel_arcmin_frac(spatial_resolution_x_ref)
    spatial_resolution_y_ref = estimate_resolution_rel_arcmin_frac(spatial_resolution_y_ref)

    if spatial_reference_ref.IsProjected():
        logger.info(f"Spatial resolution for x axis set to {spatial_resolution_x_ref}")
        logger.info(f"Spatial resolution for y axis set to {spatial_resolution_y_ref}")
    else:
        logger.info(
            f"Spatial resolution for x axis set to {spatial_resolution_x_ref} (1/{round(60 / spatial_resolution_x_ref)} of an arcmin)"
        )
        logger.info(
            f"Spatial resolution for y axis set to {spatial_resolution_y_ref} (1/{round(60 / spatial_resolution_y_ref)} of an arcmin)"
        )

    return (spatial_reference_ref, spatial_resolution_x_ref, spatial_resolution_y_ref)


def estimate_resolution_rel_arcmin_frac(estimated_resolution: float):
    """estimate a spatial resolution, if it is close to a fraction of arcmin the resolution is recomputed"""
    # try to recompute resolution without rounding issues as a fraction of an arcmin
    # compare with a arcmin fraction
    estimated_fraction_of_arcmin = 1 / (60 * estimated_resolution)
    if estimated_fraction_of_arcmin > 1:  # if less than an arcmin we leave it like this
        # now we check if we are close enough to a fraction of arcmin, given the estimation precision
        how_close_to_an_integer = 1 - round(estimated_fraction_of_arcmin) / estimated_fraction_of_arcmin
        if (
            math.fabs(how_close_to_an_integer) < 10e-3
        ):  # if less than 10e-3 of an integer we consider that is is willed to be fraction of arcmin
            return 1 / (60 * round(estimated_fraction_of_arcmin))
    return estimated_resolution
