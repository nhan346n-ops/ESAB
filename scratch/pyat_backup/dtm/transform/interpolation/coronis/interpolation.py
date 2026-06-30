#! /usr/bin/env python3
# coding: utf-8

import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from numpy.typing import ArrayLike
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.mask as mask_util
import pyat.utils.pyat_logger as log
from pyat.utils.kml_splitter import extract_polygons_from_kml

logger = log.logging.getLogger("Interpolation process")

# Progress tracking constants
PROGRESS_STEPS_PER_DTM = 5
PROGRESS_INTERPOLATION = 3
PROGRESS_CDI_UPDATE = 1
PROGRESS_FINAL = 1


@dataclass(frozen=True)
class GeoMask:
    """Geographic mask defining a region of interest."""

    mask_flags: ArrayLike
    region_slice: tuple[slice, slice]

    @property
    def row_start(self) -> int:
        return self.region_slice[0].start or 0

    @property
    def row_end(self) -> int:
        return self.region_slice[0].stop - 1 if self.region_slice[0].stop else -1

    @property
    def col_start(self) -> int:
        return self.region_slice[1].start or 0

    @property
    def col_end(self) -> int:
        return self.region_slice[1].stop - 1 if self.region_slice[1].stop else -1


@dataclass(frozen=True)
class InterpolatedRegion:
    """Interpolated region with its metadata."""

    kml_path: str
    geo_mask: Optional[GeoMask]
    interpolate_dtm_path: str


def interpolate_dtms(
    i_paths: List,
    o_paths: List,
    interpolation_algo: Callable[[str, str, str | None], None],
    overwrite: bool = False,
    areas: Optional[str] = None,
    cdi_interpolation_algo: str = "closest_neighbor",  # or most_common_neighbor
    monitor: ProgressMonitor = DefaultMonitor,
):
    """
    Interpolate DTM files according to defined geographic zones.

    For each input DTM file:
    1. Crop regions of interest defined by KML polygons
    2. Apply interpolation algorithm to each region
    3. Merge results into output file
    4. Update CDI layer

    Args:
        i_paths: Paths to input DTM files
        o_paths: Paths to output DTM files
        interpolation_algo: Interpolation function (input_path, output_path, kml_path)
        overwrite: If True, overwrite existing files
        areas: Path to KML file defining regions of interest
        cdi_interpolation_algo: Algorithm for CDI layer
            ("closest_neighbor" or "most_common_neighbor")
        monitor: Progress monitor

    Raises:
        ValueError: If no cells are found in a region of interest
    """
    monitor.set_work_remaining(len(i_paths) * PROGRESS_STEPS_PER_DTM)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kml_files = _prepare_area_files(areas, tmp_dir)

        for input_path, output_path in zip(i_paths, o_paths):
            if not _should_process_file(input_path, output_path, overwrite):
                continue

            try:
                _process_single_dtm(
                    input_path=input_path,
                    output_path=output_path,
                    kml_files=kml_files,
                    tmp_dir=tmp_dir,
                    interpolation_algo=interpolation_algo,
                    cdi_interpolation_algo=cdi_interpolation_algo,
                    monitor=monitor,
                )
            except Exception:
                logger.error(f"Error while processing file {input_path}", exc_info=True)

    monitor.done()


def _prepare_area_files(areas: Optional[str], tmp_dir: str) -> List[str]:
    """Extract polygons from KML file into separate files."""
    if areas is None:
        return []
    return extract_polygons_from_kml(areas, tmp_dir)


def _should_process_file(input_path: str, output_path: str, overwrite: bool) -> bool:
    """Check if the file should be processed."""
    logger.info(f"Processing file {input_path}")

    if os.path.exists(output_path) and not overwrite:
        logger.error(
            f"File already exists and overwrite not allowed " f"(allow overwrite with option: '-o --overwrite')"
        )
        return False

    return True


def _process_single_dtm(
    input_path: str,
    output_path: str,
    kml_files: List[str],
    tmp_dir: str,
    interpolation_algo: Callable[[str, str, str | None], None],
    cdi_interpolation_algo: str,
    monitor: ProgressMonitor,
) -> None:
    """Process a single DTM file with interpolation."""
    interpolated_regions = _interpolate_regions(input_path, kml_files, tmp_dir, interpolation_algo, monitor)

    _merge_results_to_output(input_path, output_path, interpolated_regions, cdi_interpolation_algo, monitor)


def _interpolate_regions(
    input_path: str,
    kml_files: List[str],
    tmp_dir: str,
    interpolation_algo: Callable[[str, str, str | None], None],
    monitor: ProgressMonitor,
) -> List[InterpolatedRegion]:
    """Interpolate all regions defined by KML files, or entire file if no KML provided."""
    interpolated_regions: List[InterpolatedRegion] = []

    # If no KML files, interpolate entire input file
    if not kml_files:
        try:
            region = _interpolate_entire_file(input_path, tmp_dir, interpolation_algo)
            interpolated_regions.append(region)
        except Exception as e:
            logger.error(f"Error interpolating entire file {input_path}: {e}")
    else:
        # Interpolate specific regions defined by KML files
        for kml_path in kml_files:
            try:
                region = _interpolate_single_region(input_path, kml_path, tmp_dir, interpolation_algo)
                if region is not None:
                    interpolated_regions.append(region)
            except ValueError as e:
                logger.error(f"Error processing polygon in {kml_path}: {e}")
                continue

    monitor.worked(PROGRESS_INTERPOLATION)
    return interpolated_regions


def _interpolate_single_region(
    input_path: str, kml_path: str, tmp_dir: str, interpolation_algo: Callable[[str, str, str | None], None]
) -> InterpolatedRegion | None:
    """Interpolate a single region defined by a KML file."""
    logger.info(f"Processing area of interest defined in {kml_path}")

    geo_mask = _create_geo_mask(input_path, kml_path)
    if geo_mask is None:
        return None

    with dtm_driver.open_dtm(input_path) as input_dtm:
        kml_file_name = os.path.splitext(os.path.basename(kml_path))[0]
        cropped_dtm_path = os.path.join(tmp_dir, f"{kml_file_name}_cropped.dtm.nc")

        if _crop_dtm_to_area(input_dtm, cropped_dtm_path, geo_mask):
            interpolated_path = _apply_interpolation(
                cropped_dtm_path, kml_file_name, tmp_dir, kml_path, interpolation_algo
            )
            return InterpolatedRegion(
                kml_path=kml_path,
                geo_mask=geo_mask,
                interpolate_dtm_path=interpolated_path,
            )
    return None


def _crop_dtm_to_area(input_dtm: dtm_driver.DtmDriver, output_path: str, geo_mask: GeoMask) -> bool:
    """Crop DTM to region of interest."""
    logger.info("Cropping input DTM to area of interest...")

    output_dtm = dtm_driver.DtmDriver(output_path)
    try:
        return _crop_dtm_to_area_of_interest(input_dtm, output_dtm, geo_mask)
    finally:
        output_dtm.close()
    return False


def _apply_interpolation(
    cropped_dtm_path: str,
    kml_file_name: str,
    tmp_dir: str,
    kml_path: str,
    interpolation_algo: Callable[[str, str, str | None], None],
) -> str:
    """Apply interpolation algorithm on cropped DTM."""
    logger.info("Processing elevation interpolation...")

    interpolated_path = os.path.join(tmp_dir, f"{kml_file_name}_interpolated.dtm.nc")
    interpolation_algo(cropped_dtm_path, interpolated_path, kml_path)

    logger.info("Interpolation done")
    return interpolated_path


def _merge_results_to_output(
    input_path: str,
    output_path: str,
    interpolated_regions: List[InterpolatedRegion],
    cdi_interpolation_algo: str,
    monitor: ProgressMonitor,
) -> None:
    """Merge interpolation results into output file."""
    logger.info("Copying input DTM to output...")
    shutil.copy(input_path, output_path)

    logger.info("Opening DTM files to update...")
    with dtm_driver.open_dtm(output_path, mode="r+") as output_dtm:
        mask_of_new_elevations = _transfer_all_interpolated_regions(output_dtm, interpolated_regions)

        monitor.worked(PROGRESS_CDI_UPDATE)
        if mask_of_new_elevations is not None:
            _update_cdi_layer(output_dtm, mask_of_new_elevations, cdi_interpolation_algo)

        logger.info(f"Interpolation of {input_path} done successfully.")

    monitor.worked(PROGRESS_FINAL)


def _transfer_all_interpolated_regions(
    output_dtm: dtm_driver.DtmDriver, interpolated_regions: List[InterpolatedRegion]
) -> ArrayLike:
    """Transfer all interpolated regions to output DTM."""
    mask_of_new_elevations = None

    for region in interpolated_regions:
        with dtm_driver.open_dtm(region.interpolate_dtm_path) as interpolated_dtm:
            result = _transfer_interpolated_cells(output_dtm, interpolated_dtm, region.geo_mask)
            mask_of_new_elevations = result if mask_of_new_elevations is None else (mask_of_new_elevations | result)

    return mask_of_new_elevations


def _update_cdi_layer(
    output_dtm: dtm_driver.DtmDriver, mask_of_new_elevations: ArrayLike, cdi_interpolation_algo: str
) -> None:
    """Update CDI layer according to chosen algorithm."""
    if "closest" in cdi_interpolation_algo.lower():
        logger.info("Update CDI layer with closest neighbor...")
        cdi_util.update_with_closest_cdi(output_dtm, mask_of_new_elevations)
    else:
        logger.info("Update CDI layer with most common neighbor...")
        cdi_util.update_cdi(output_dtm, mask_of_new_elevations)


def _create_geo_mask(dtm_path: str, kml_path: str) -> GeoMask | None:
    """
    Create a geographic mask from a KML file.

    Args:
        dtm_path: Path to DTM file
        kml_path: Path to KML file defining the area

    Returns:
        GeoMask containing the mask and region indices

    Raises:
        ValueError: If no cells are found in the area
    """
    mask_flags = mask_util.compute_geo_mask_from_dtm(dtm_path, [kml_path] if kml_path is not None else [])

    rows, cols = np.where(mask_flags == 1)
    if len(rows) == 0 or len(cols) == 0:
        logger.warning(f"No cell found in area of interest. Area will be ignored.")
        return None

    row_start = rows.min()
    row_end = rows.max()
    col_start = cols.min()
    col_end = cols.max()

    region_slice = (slice(row_start, row_end + 1), slice(col_start, col_end + 1))
    result = GeoMask(mask_flags[region_slice], region_slice)

    logger.info(
        f"Area of interest: rows [{result.row_start}:{result.row_end}] " f"cols [{result.col_start}:{result.col_end}]"
    )

    return result


def _crop_dtm_to_area_of_interest(
    input_dtm: dtm_driver.DtmDriver, output_dtm: dtm_driver.DtmDriver, geo_mask: GeoMask
) -> bool:
    """
    Crop a DTM to a region of interest defined by a geographic mask.

    Args:
        input_dtm: Source DTM
        output_dtm: Destination DTM
        geo_mask: Mask defining the area to extract
    """
    input_file = input_dtm.dtm_file

    west = input_file.west + geo_mask.col_start * input_file.spatial_resolution_x
    south = input_file.south + geo_mask.row_start * input_file.spatial_resolution_y
    logger.debug(f"West: {west}, South: {south}")

    output_dtm.create_file(
        col_count=geo_mask.col_end - geo_mask.col_start + 1,
        origin_x=west,
        spatial_resolution_x=input_file.spatial_resolution_x,
        row_count=geo_mask.row_end - geo_mask.row_start + 1,
        origin_y=south,
        spatial_resolution_y=input_file.spatial_resolution_y,
        spatial_reference=input_file.spatial_reference,
    )

    # Copy elevation layer
    elevations = input_dtm[DtmConstants.ELEVATION_NAME][geo_mask.region_slice]
    output_dtm.add_layer(layer_name=DtmConstants.ELEVATION_NAME, data=elevations)

    # Interpolation flag layer is no longer copied to avoid a confusing behavior where only interpolated cells are re-interpolated.
    # # Copy interpolation flag layer
    # if DtmConstants.INTERPOLATION_FLAG in input_dtm:
    #     interp_flags = input_dtm[DtmConstants.INTERPOLATION_FLAG][geo_mask.region_slice]
    #     output_dtm.add_layer(layer_name=DtmConstants.INTERPOLATION_FLAG, data=interp_flags)

    cell_count = np.count_nonzero(elevations.mask)
    logger.info(f"Number of cells to interpolate: {cell_count}")
    # Ensure there is something to interpolate
    if cell_count == 0:
        logger.warning("No cell found in cropped DTM. Area will be ignored.")

    return cell_count > 0


def _transfer_interpolated_cells(
    output_dtm: dtm_driver.DtmDriver, interpolated_dtm: dtm_driver.DtmDriver, geo_mask: Optional[GeoMask]
) -> ArrayLike:
    """
    Transfer interpolated cells to output DTM.

    Args:
        output_dtm: Destination DTM
        interpolated_dtm: DTM containing interpolated values
        geo_mask: Mask defining the area to transfer (None for entire file)

    Returns:
        Mask of updated elevations
    """
    logger.info("Creating interpolation layer...")
    output_dtm.create_interpolation_layer()

    logger.info("Update elevation layer with interpolation result...")
    interpolated_elevations = interpolated_dtm[DtmConstants.ELEVATION_NAME][:]

    if geo_mask is not None:
        # Mask values outside the region of interest
        reset_cell_mask = geo_mask.mask_flags != 1
        interpolated_elevations[reset_cell_mask] = dtm_driver.get_missing_value(DtmConstants.ELEVATION_NAME)
        return output_dtm.update_elevation(
            interpolated_elevations, row_start=geo_mask.row_start, col_start=geo_mask.col_start
        )
    else:
        # Update entire file
        return output_dtm.update_elevation(interpolated_elevations)


def _interpolate_entire_file(
    input_path: str, tmp_dir: str, interpolation_algo: Callable[[str, str, str | None], None]
) -> InterpolatedRegion:
    """
    Interpolate entire DTM file when no specific regions are defined.

    Args:
        input_path: Path to input DTM file
        tmp_dir: Temporary directory for intermediate files
        interpolation_algo: Interpolation function to apply

    Returns:
        InterpolatedRegion representing the entire file
    """
    logger.info(f"Processing entire file (no specific area defined)")

    file_name = os.path.splitext(os.path.basename(input_path))[0]

    # Apply interpolation directly on the entire input file
    logger.info("Processing elevation interpolation...")
    interpolated_path = os.path.join(tmp_dir, f"{file_name}_interpolated.dtm.nc")
    interpolation_algo(input_path, interpolated_path, None)
    logger.info("Interpolation done")

    # Return InterpolatedRegion without geo_mask (entire file)
    return InterpolatedRegion(
        kml_path="",  # No KML file for entire file interpolation
        geo_mask=None,
        interpolate_dtm_path=interpolated_path,
    )
