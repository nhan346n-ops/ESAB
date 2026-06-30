#! /usr/bin/env python3
# coding: utf-8

import csv
from functools import partial
from os import PathLike
from typing import List, NamedTuple

import numpy as np
import osgeo.gdal as gdal
import pandas
from numpy.typing import ArrayLike
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.pyat_logger as log
from pyat.utils import gdal_utils

PROCESS_NAME = "expected_stdev"
__logger = log.logging.getLogger(PROCESS_NAME)


class CsvContent(NamedTuple):
    """
    Tuple holding values contained in the csv file (stdev_csv_path)
    """

    angle: ArrayLike
    stdev: ArrayLike


def computes(
    i_paths: List[PathLike],
    stdev_csv_path: PathLike,
    beam_angles: ArrayLike | None = None,
    o_paths: List[PathLike] | None = None,
    overwrite: bool = False,
    monitor: ProgressMonitor = DefaultMonitor,
) -> None:
    """
    Main function

    Browse the list of input files (DTM) and calculate the expected STDEV for each of them..

    Parameters
    ----------
        i_paths : list of PathLike
            input files (DTM)
        stdev_csv_path : PathLike
            CSV file containing stdev corresponding to angles
        beam_angles :
            Array of across angles to use for interpolating stdev.
            If absent, the across angles are deduced from the layers elevations and across distance of the DTM
        i_paths : list of PathLike
            Optional list of input TIF files.
        overwrite :
            True to overwrite existing tif files
        monitor :
            Progress monitor
    returns the resulting statistics are grouped in a AllMetrics instance
    """
    # Loading expected stdev
    try:
        expected_stdev = load_stdev_in_csv(stdev_csv_path)
    except ValueError as e:
        __logger.error("Unable to parse the CSV file. Process aborted")
        return
    except FileNotFoundError as e:
        __logger.error("CSV file not found. Process aborted")
        return

    # pylint:disable=unused-argument
    def __process_one_dtm(i_path: PathLike, o_path: PathLike, sub_monitor, beam_angles: ArrayLike | None):
        with dtm_driver.open_dtm(i_path) as i_driver:
            try:
                if beam_angles is None:
                    beam_angles = _computes_beam_angle(i_driver)
                interpolated_stdev = _interpolate_stdev(beam_angles, expected_stdev)
                write_tif(i_driver.dtm_file, interpolated_stdev, o_path)
            except IndexError as e:
                raise ValueError(f"Unable to computes the beam angles for {i_path}") from e

    # Launch
    process_util.process_each_input_file_to_output_file(
        PROCESS_NAME,
        i_paths=i_paths,
        process_data_func=partial(__process_one_dtm, beam_angles=beam_angles),
        logger=__logger,
        o_paths=o_paths,
        suffix="_" + PROCESS_NAME,
        extension=".tif",
        overwrite=overwrite,
        monitor=monitor,
    )


def load_stdev_in_csv(stdev_csv_path: PathLike) -> CsvContent:
    """
    Parsing of the csv file containing the expected stdev for angles

    Parameters
    ----------
        stdev_csv_path : PathLike
            Path of the csv file.

    Raises
    ------
        ValueError
            if file is not parsable
        FileNotFoundError
            if file can not be find

    Returns
    -------
        Instance of CsvContent with array of angle and stdev
    """
    with open(stdev_csv_path, "r", encoding="utf-8") as stdev_csv_file:
        dialect = csv.Sniffer().sniff(stdev_csv_file.read(1024))
        csv_content = None
        try:
            # Try to read with a point as decimal separator
            csv_content = _read_csv(stdev_csv_file, dialect, ".")
        except ValueError:
            # Try to read with a comma as decimal separator
            csv_content = _read_csv(stdev_csv_file, dialect, ",")
        return CsvContent(csv_content[CsvContent._fields[0]].to_numpy(), csv_content[CsvContent._fields[1]].to_numpy())


def _read_csv(stdev_csv_file, dialect: csv.Dialect, decimal: str):
    """
    Read the content of the csv file

    Parameters
    ----------
        stdev_csv_file :
            Opened handler of the CSV file.
        dialect :
            csv.Dialect obtained with csv.Sniffer
        decimal :
            Expected decimal separator

    Raises
    ------
        ValueError
            if file is not parsable

    Returns
    -------
        The DataFrame of angles and stdevs
    """
    stdev_csv_file.seek(0)
    return pandas.read_csv(
        stdev_csv_file,
        dialect=dialect,
        names=CsvContent._fields,
        header=0,
        dtype={"angle": float, "stdev": float},
        decimal=decimal,
    )


def _computes_beam_angle(dtm_driver: dtm_driver.DtmDriver) -> ArrayLike:
    """
    Computes the beam angles from layers elevation and max across distance of the DTM

    Parameters
    ----------
        dtm_driver : DtmDriver
            Input DTM.

    Raises
    ------
        IndexError
            if DTM does not contained the expected layer

    Returns
    -------
        An array of beam angles
    """
    max_across_distance = dtm_driver[DtmConstants.MAX_ACROSS_DISTANCE][:]
    elevations = dtm_driver[DtmConstants.ELEVATION_NAME][:]
    return np.arctan(max_across_distance / elevations) * 180.0 / np.pi


def _interpolate_stdev(beam_angle: ArrayLike, expected_stdev: CsvContent) -> ArrayLike:
    """
    Computes the beam angles from layers elevation and max across distance of the DTM

    Parameters
    ----------
        dtm_driver : DtmDriver
            Input DTM.

    Raises
    ------
        IndexError
            if DTM does not contained the expected layer

    Returns
    -------
        An array of beam angles
    """
    return np.interp(beam_angle, expected_stdev.angle, expected_stdev.stdev)


def write_tif(dtm_file: dtm_driver.DtmFile, interpolated_stdev: ArrayLike, o_path: PathLike) -> None:
    """
    Writing the result in a tiff file

    Parameters
    ----------
        dtm_file :
            original dtm file.
        interpolated_stdev :
            Array of float of the resulting stdev
        o_path :
            Path of the TIF
    """
    dataset = gdal.GetDriverByName("GTiff").Create(
        o_path,
        xsize=dtm_file.col_count,
        ysize=dtm_file.row_count,
        bands=1,
        eType=gdal.GDT_Float32,
    )
    dataset.SetGeoTransform(
        (dtm_file.west, dtm_file.spatial_resolution_x, 0.0, dtm_file.north, 0.0, -dtm_file.spatial_resolution_y)
    )

    dataset.SetProjection(dtm_file.spatial_reference.ExportToProj4())
    band = dataset.GetRasterBand(1)
    band.SetDescription(f"Expected STDEV for file {dtm_file._file_path}")
    band.SetNoDataValue(np.nan)
    band.SetRasterColorInterpretation(gdal.GCI_GrayIndex)

    band.WriteArray(gdal_utils.netcdf_to_gdal(interpolated_stdev))
    dataset.FlushCache()
    dataset = None
