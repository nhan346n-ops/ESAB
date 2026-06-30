#! /usr/bin/env python3
# coding: utf-8

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import netCDF4 as nc
import numpy as np
from osgeo import ogr, osr
from pygws.service.progress_monitor import DefaultMonitor

import pyat.common.geo_file as gf
import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.cdi.cdi_layer_util as cdi_util
import pyat.dtm.dtm_legacy_constants as OldIfr
import pyat.dtm.dtm_standard_constants as NewFormatConst
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.netcdf_utils as nc_util
import pyat.utils.pyat_logger as log
from pyat.utils import nc_encoding
from pyat.utils.nc_encoding import open_nc_file
from pyat.utils.proj_utils import validate_proj4_string


class DTMMigrate:
    """
    Utility classe used to migrate netcdf3 dtm file to netcdf4 dtm format
    The nc format is inspired from the GEBCO nc release format and
    the NOAA grid format template https://www.nodc.noaa.gov/data/formats/nc/v2.0/grid.cdl
    """

    def __init__(
        self,
        i_paths: list,
        o_paths: list,
        overwrite: bool = False,
        monitor=DefaultMonitor,
        logger=log.logging.getLogger("DTMMigrate"),
    ):
        """Constructor.

        Arguments:
            i_paths {list} -- Input file list (.dtm).
            o_paths {list} -- Optional output file list (.nc). (default: {None})
            overwrite {bool} -- true to overwrite output file if exists. (default: {False})
            monitor {list} -- Progress monitor. (default is a silent monitor: {DefaultMonitor})
        """
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.overwrite = overwrite
        self.monitor = monitor

        self.logger = logger
        # self.logger = log.logging.getLogger(self.__class__.__name__)

    def _migrateToDataset(
        self,
        inDataset: nc.Dataset,
        o_dtm_driver: dtm_driver.DtmDriver,
        monitor,
    ) -> None:

        # create a mapping between variable name in old DTM and new DTM
        dic = {
            OldIfr.VARIABLE_DEPTH: NewFormatConst.ELEVATION_NAME,
            OldIfr.VARIABLE_DEPTH_SMOOTH: NewFormatConst.ELEVATION_SMOOTHED_NAME,
            OldIfr.VARIABLE_MAX_SOUNDING: NewFormatConst.ELEVATION_MAX,
            OldIfr.VARIABLE_MIN_SOUNDING: NewFormatConst.ELEVATION_MIN,
            OldIfr.VARIABLE_VSOUNDINGS: NewFormatConst.VALUE_COUNT,
            OldIfr.VARIABLE_STDEV: NewFormatConst.STDEV,
            OldIfr.VARIABLE_INTERPOLATION_FLAG: NewFormatConst.INTERPOLATION_FLAG,
            OldIfr.VARIABLE_CDI: NewFormatConst.CDI_INDEX,
            OldIfr.VARIABLE_REFLECTIVITY: NewFormatConst.BACKSCATTER,
            OldIfr.VARIABLE_MIN_ACROSS_DISTANCE: NewFormatConst.MIN_ACROSS_DISTANCE,
            OldIfr.VARIABLE_MAX_ACROSS_DISTANCE: NewFormatConst.MAX_ACROSS_DISTANCE,
            OldIfr.VARIABLE_ACCROSS_ANGLE: NewFormatConst.MAX_ACCROSS_ANGLE,
            "Z": NewFormatConst.ELEVATION_NAME,
            "DEPTH_cor_RAC_fromv".upper(): NewFormatConst.ELEVATION_NAME,
            "DEPTH_cor_RAC125".upper(): NewFormatConst.ELEVATION_NAME,
            "Z_krige_cote_500m_b".upper(): NewFormatConst.ELEVATION_NAME,
        }

        # check if any of the above variables is present in the dataset
        # pylint:disable=use-a-generator
        if not any([v.upper() in dic for v in inDataset.variables]):
            raise KeyError("No variable corresponding to DEPTH or REFLECTIVITY")

        # o_dtm_driver.dataset.history = "Converted with Python MigrateDtm script from " + nc_encoding.filepath(inDataset)

        # Migrate variables
        n = len(inDataset.variables) + 1
        monitor.set_work_remaining(n)
        for v in inDataset.variables:
            if v.upper() in dic:
                # Get data from old DTM
                if is_very_old(inDataset):
                    data = nc3_read_var(inDataset, v)
                else:
                    data = inDataset.variables[v][:]
                # Set data in new DTM
                if v.upper() == OldIfr.VARIABLE_REFLECTIVITY:
                    o_dtm_driver.add_layer(dic[v.upper()], np.flipud(data))
                else:
                    o_dtm_driver.add_layer(dic[v.upper()], data)

            monitor.worked(1)

        # Write layers CDI. need to create a mapping
        if OldIfr.VARIABLE_CDI_INDEX in inDataset.variables:
            v = inDataset.variables[OldIfr.VARIABLE_CDI_INDEX]
            monitor.worked(1)
            ids = cdi_util.trim_string_array(nc.chartostring(v[:]))
            # we do not remove empty entries, just in case of bugs and empty values remaining
            # ids=ids[np.logical_not(ids == "")]
            o_dtm_driver.create_cdi_reference_variable(cdis=ids)

        cdi_util.clean_cdi(inDataset)

    def _upgradeWkt(self, esri_pe_string: str) -> str:
        """
        Utility function to reinterpret projection WKT description from old MNT to a GDAL compatible description
        Args: esri_pe_string: projection description in WKT from old MNT format
        Returns: reformatted esri_pe_string
        """
        if "Lambert" in esri_pe_string:
            if "Scale_Factor" in esri_pe_string:
                esri_pe_string = esri_pe_string.replace("Lambert_Conformal_Conic", "Lambert_Conformal_Conic_1SP")
            else:
                esri_pe_string = esri_pe_string.replace("Lambert_Conformal_Conic", "Lambert_Conformal_Conic_2SP")
        elif '"Mercator"' in esri_pe_string:
            if "Scale_Factor" in esri_pe_string:
                esri_pe_string = esri_pe_string.replace("Mercator", "Mercator_1SP")
            else:
                esri_pe_string = esri_pe_string.replace("Mercator", "Mercator_2SP")
        elif "Equidistant_Cylindrical" in esri_pe_string:
            esri_pe_string = esri_pe_string.replace("Equidistant_Cylindrical", "Equirectangular")
        elif "UTM" in esri_pe_string:
            # to suppress a wrong closing bracket in WKT strings for UTM projection from OLD Caraibes DTM
            esri_pe_string = esri_pe_string.replace(
                ']],PARAMETER["Latitude_Of_Origin"', '],PARAMETER["Latitude_Of_Origin"'
            )

        return esri_pe_string

    def _get_projection(self, inDataset: nc.Dataset) -> tuple[osr.SpatialReference, str]:
        """
        Return projection definition from old DTM file as an osr.SpatialReference() object.
        Raise a ValueError exception if the projection is missing or not supported.
        """
        srs_orig = ""
        srs_new = osr.SpatialReference()
        try:  # 1 : rely on mbProj4String attribute
            if inDataset.mbProj4String:
                srs_orig = validate_proj4_string(inDataset.mbProj4String)
                if srs_new.ImportFromProj4(srs_orig) != ogr.OGRERR_NONE:
                    error_msg = self._log_error(
                        f"Migration not available for this projected files. Wrong proj4 input file projection : {srs_orig}"
                    )
                    raise ValueError(error_msg)

        except AttributeError as exc:  # 2 :  or DEPTH or REFLECTIVITY variables attribute esri_pe_string
            var_esri_pe_gen = (v for _, v in inDataset.variables.items() if "esri_pe_string" in v.ncattrs())
            var_esri_pe = next(var_esri_pe_gen, None)
            if var_esri_pe:
                srs_orig = self._upgradeWkt(var_esri_pe.esri_pe_string)
                if srs_new.ImportFromWkt(srs_orig) != ogr.OGRERR_NONE:
                    error_msg = self._log_error(
                        f"Migration not available for this projected files. Wrong WKT input file projection : {srs_orig})"
                    )
                    raise ValueError(error_msg) from exc

            else:  # 3 : and finaly, old CIB projection variables
                try:
                    if inDataset.variables[OldIfr.VARIABLE_PROJECTION]:
                        srs_orig = nc3_get_proj4(inDataset)
                        if srs_new.ImportFromProj4(srs_orig) != ogr.OGRERR_NONE:
                            error_msg = self._log_error(
                                f"Migration not available for this projected files. Wrong CIB input file projection : {srs_orig})"
                            )
                            raise ValueError(error_msg) from exc

                except KeyError:
                    error_msg = self._log_error(f"Migration not available for this file : no spatial reference")
                    raise ValueError(error_msg) from exc

        return srs_new, srs_orig

    def _check_projection(self, spatial_ref: osr.SpatialReference) -> bool:
        if spatial_ref.IsProjected():
            if not nc_util.is_spatial_reference_supported(spatial_ref):
                error_msg = self._log_error(f'Projection not supported : {spatial_ref.GetAttrValue("PROJECTION")} ')
                raise ValueError(error_msg)
            self.logger.info(f'Projection : {spatial_ref.GetAttrValue("PROJECTION")}')

    def _check_grid_orientation(self, inDataset: nc.Dataset):
        """
        Checks that grid orientation is supported : only straight (aka "STR") orientation supported.
        """
        # retrieve DTM grid orientation : "STR" or "OBL"
        if OldIfr.VARIABLE_FORMAT in inDataset.variables.keys():
            dtm_format = nc3_read_att(inDataset, OldIfr.VARIABLE_FORMAT)
        elif OldIfr.VARIABLE_FORMAT in inDataset.ncattrs():
            dtm_format = inDataset.getncattr(OldIfr.VARIABLE_FORMAT)
        else:
            dtm_format = ""

        # it's impossible to convert oblique oriented files
        if dtm_format == "OBL":
            error_msg = self._log_error(f"Oblique grid orientation not supported")
            raise ValueError(error_msg)

    def _log_error(self, msg, *args, **kwargs) -> str:
        self.logger.error(msg, *args, **kwargs)
        return msg

    def __call__(self, whatif=False) -> List[str] | None:
        """
        main entry point for this class, will migrate an old netCDF DTM format to a new netCDF4 dtm.nc format
        Whatif : if True, don't create output file, only test eligibility to migration
        """
        start = datetime.now()
        self.monitor.set_work_remaining(len(self.i_paths))
        files_in_error = []
        srs_orig = ""
        error_msg = ""
        srs_new = osr.SpatialReference()

        for ind, path in enumerate(self.i_paths):
            self.logger.info(f"Start migration of file {path} to file {self.o_paths[ind]}.")
            start_tmp = datetime.now()
            sub_monitor = self.monitor.split(1)

            try:
                with open_nc_file(path) as inDataset:
                    # check if we really need to upgrade this file
                    if NewFormatConst.VERSION_ATTRIB_NAME in inDataset.ncattrs():
                        if "crs_wkt" in inDataset.variables["crs"].ncattrs():
                            srs_new.ImportFromWkt(inDataset.variables["crs"].crs_wkt)
                        else:
                            grid_mapping_name = inDataset.variables["crs"].grid_mapping_name
                            if grid_mapping_name == "latitude_longitude":
                                srs_new = gf.SR_WGS_84

                        if not whatif:
                            self.logger.info(
                                f"Format up to date : no need for migration, trying compression instead (Input file : {path})"
                            )
                            nc_util.copy_and_compress_dataset(inDataset, self.o_paths[ind])
                        else:
                            end_tmp = datetime.now()
                            self.logger.info(
                                f"File {self.o_paths[ind]} format is up to date (might need compression) (time elapsed {end_tmp - start_tmp} )."
                            )

                    else:
                        # Retrieve and check projection of the input DTM
                        try:
                            srs_new, srs_orig = self._get_projection(inDataset)
                        except ValueError as e:
                            error_msg = str(e)
                            files_in_error.append(path)
                            continue

                        # Is retrieved projection supported ?
                        try:
                            self._check_projection(srs_new)
                        except ValueError as e:
                            error_msg = str(e)
                            files_in_error.append(path)
                            continue

                        # retrieve DTM grid orientation : "STR" or "OBL"
                        try:
                            self._check_grid_orientation(inDataset)
                        except ValueError as e:
                            error_msg = str(e)
                            files_in_error.append(path)
                            continue

                        # Get grid origin coordinates, cell size, row/col count
                        origin_x, origin_y, spatial_resolution_x, spatial_resolution_y, col_count, row_count = (
                            get_grid_spec(inDataset)
                        )
                        # retrieve original metadata
                        metadata = get_metadata(inDataset)

                        if not whatif:
                            # Migrate DTM
                            o_dtm_driver = dtm_driver.DtmDriver(self.o_paths[ind])
                            with o_dtm_driver.create_file(
                                col_count=col_count,
                                origin_x=origin_x,
                                spatial_resolution_x=spatial_resolution_x,
                                row_count=row_count,
                                origin_y=origin_y,
                                spatial_resolution_y=spatial_resolution_y,
                                spatial_reference=srs_new,
                                overwrite=self.overwrite,
                                metadata=metadata,
                            ) as outDataset:
                                self._migrateToDataset(inDataset, o_dtm_driver, sub_monitor)

                            end_tmp = datetime.now()
                            self.logger.info(
                                f"File {self.o_paths[ind]} migrated with success (time elapsed {end_tmp - start_tmp} )."
                            )
                        else:
                            end_tmp = datetime.now()
                            self.logger.info(
                                f"File {self.o_paths[ind]} can be upgraded (time elapsed {end_tmp - start_tmp} )."
                            )

            except ValueError as e:
                error_msg = self._log_error(str(e))
                files_in_error.append(path)

            except KeyError as e:
                error_msg = self._log_error(str(e))
                files_in_error.append(path)

            except Exception as e:
                error_msg = self._log_error(str(e), exc_info=True)
                files_in_error.append(path)

            finally:
                # delete output file if in error
                if path in files_in_error and os.path.exists(self.o_paths[ind]):
                    os.remove(self.o_paths[ind])
                sub_monitor.done()

        process_util.log_result(self.logger, start, files_in_error)

        # Used to migrate GeoOcean DTM database (see migrate_database.py in MNTBathy)
        return files_in_error, srs_orig, srs_new, error_msg


def is_very_old(inDataset):
    """
    return True if it's a very old DTM without COLUMN and LINES variables (before caraibes v3.4).
    NB : we can't rely on mbVersion attribute as it has been proved to be wrong in some DTM files.
    """
    if OldIfr.VARIABLE_COLUMN in inDataset.variables.keys():
        return False
    else:
        return True


def get_grid_spec(inDataset: nc.Dataset):
    """
    Returns grid specification : origin coordinates, cell size, row/col count
    """
    if not is_very_old(inDataset):
        # General case: with COLUMN and LINES variables (> caraibes v3.4)
        # Must recompute from COLUMN/LINES as we can't rely on element_x_size and element_y_size which can be wrong in some DTM
        spatial_resolution_x = abs(
            inDataset.variables[OldIfr.VARIABLE_COLUMN][1] - inDataset.variables[OldIfr.VARIABLE_COLUMN][0]
        )
        spatial_resolution_y = abs(
            inDataset.variables[OldIfr.VARIABLE_LINE][1] - inDataset.variables[OldIfr.VARIABLE_LINE][0]
        )
        if spatial_resolution_y == 0 or spatial_resolution_x == 0:
            spatial_resolution_x = np.round(
                (inDataset.getncattr(OldIfr.VARIABLE_XMAX_METRIC) - inDataset.getncattr(OldIfr.VARIABLE_XMIN_METRIC))
                / (inDataset.dimensions[OldIfr.DIM_COLUMNS].size - 1)
            )
            spatial_resolution_y = np.round(
                (inDataset.getncattr(OldIfr.VARIABLE_YMAX_METRIC) - inDataset.getncattr(OldIfr.VARIABLE_YMIN_METRIC))
                / (inDataset.dimensions[OldIfr.DIM_LINE].size - 1)
            )
        origin_x = inDataset.variables[OldIfr.VARIABLE_COLUMN][0] - 0.5 * spatial_resolution_x
        origin_y = np.min(inDataset.variables[OldIfr.VARIABLE_LINE]) - 0.5 * spatial_resolution_y
        col_count = inDataset.dimensions[OldIfr.DIM_COLUMNS].size
        row_count = inDataset.dimensions[OldIfr.DIM_LINE].size

    else:
        # very old DTM without COLUMN and LINES variables (< caraibes v3.4)
        # rely on (element_x_size, xmin_metric) attributes
        spatial_resolution_x = nc3_read_att(inDataset, OldIfr.VARIABLE_ELEMENT_X_SIZE) * 10 ** (-3)
        spatial_resolution_y = nc3_read_att(inDataset, OldIfr.VARIABLE_ELEMENT_Y_SIZE) * 10 ** (-3)
        origin_x = nc3_read_att(inDataset, OldIfr.VARIABLE_XMIN_METRIC)
        origin_y = nc3_read_att(inDataset, OldIfr.VARIABLE_YMIN_METRIC)
        col_count = nc3_read_att(inDataset, OldIfr.VARIABLE_NUMBER_COLUMNS)
        row_count = nc3_read_att(inDataset, OldIfr.VARIABLE_NUMBER_LINES)

    return origin_x, origin_y, spatial_resolution_x, spatial_resolution_y, col_count, row_count


def check_grid_spec(inDataset: nc.Dataset, origin_x, origin_y, spatial_resolution_x, spatial_resolution_y, spatial_ref):
    """
    check coherence with projected geographic bounding box
    """
    XY_origin = get_projected_bbox(inDataset, spatial_ref)
    delta_xmin_metric = abs(XY_origin[0][0] - origin_x) / spatial_resolution_x * 100
    delta_ymin_metric = abs(XY_origin[0][1] - origin_y) / spatial_resolution_y * 100
    if max(delta_ymin_metric, delta_xmin_metric) > 50:  # percent
        # more than half a cell-size offset between the two, rely on projected geographic bounding box
        origin_x = XY_origin[0][0]
        origin_y = XY_origin[0][1]


def get_projected_bbox(src, spatial_reference):
    """
    Compute south-west and north-east projected boundingbox coordinates from LatLon bounding box extent and projection definition.
    The bounding box should take into account that the row/col coordinates are referred to the center.
    of the cells, so the real bounding box should be increased by half a cell size on each direction.
    Return [west, south, east north] projected coordinates of the outermost bottom-left and top-right pixels corners.
    """
    if nc3_read_att(src, OldIfr.VARIABLE_NORTH_LATITUDE) is not None:
        north = nc3_read_att(src, OldIfr.VARIABLE_NORTH_LATITUDE) * 10**-6
        south = nc3_read_att(src, OldIfr.VARIABLE_SOUTH_LATITUDE) * 10**-6
        east = nc3_read_att(src, OldIfr.VARIABLE_EAST_LONGITUDE) * 10**-6
        west = nc3_read_att(src, OldIfr.VARIABLE_WEST_LONGITUDE) * 10**-6
    elif OldIfr.VARIABLE_NORTH_LATITUDE in src.ncattrs():
        north = src.getncattr(OldIfr.VARIABLE_NORTH_LATITUDE)
        south = src.getncattr(OldIfr.VARIABLE_SOUTH_LATITUDE)
        east = src.getncattr(OldIfr.VARIABLE_EAST_LONGITUDE)
        west = src.getncattr(OldIfr.VARIABLE_WEST_LONGITUDE)
    else:
        north = src.getncattr(OldIfr.ATT_MB_NORTH_LATITUDE)
        south = src.getncattr(OldIfr.ATT_MB_SOUTH_LATITUDE)
        east = src.getncattr(OldIfr.ATT_MB_EAST_LONGITUDE)
        west = src.getncattr(OldIfr.ATT_MB_WEST_LONGITUDE)

    # unproject the bounding box
    WGS84 = osr.SpatialReference()
    WGS84.ImportFromEPSG(4326)
    WGS84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)  # to avoid Lon/Lat inversion since gdal3.0
    fromWGS84 = osr.CoordinateTransformation(WGS84, spatial_reference)

    return (fromWGS84.TransformPoint(west, south)[:2], fromWGS84.TransformPoint(east, north)[:2])


def get_lambert_proj4(inDataset: nc.Dataset) -> str:
    """
    Returns a proj4 string corresponding to Lambert projection defined in old DTM
    :param inDataset:
    :return:proj4 formated string
    """
    # Lambert projection : parameters as a function of Lambert_type
    lambert_type_to_EPSG = {
        "Lambert 93": 2154,
        "France I    (North)": 27561,
        "France II (Center)": 27562,
        "France III  (South)": 27563,
        "France IV (Corsica)": 27564,
        "France I extended": 27572,
        "France II extended": 27572,
        "France III extended": 27573,
        "France IV extended": 27574,
        "Europe": None,
        "World": None,
    }

    lambert_type = nc3_read_att(inDataset, OldIfr.VARIABLE_LAMBERT_TYPE)

    if lambert_type:
        lambert_proj4 = proj4_from_EPSG(lambert_type_to_EPSG[lambert_type])
    else:
        lambert_proj4 = (
            f"+proj=lcc "
            f"+lat_0={nc3_read_att(inDataset, OldIfr.VARIABLE_LAMBERT_REFERENCE_LATITUDE_1) * 10 ** (-6)} "
            f"+lat_1={nc3_read_att(inDataset, OldIfr.VARIABLE_LAMBERT_REFERENCE_LATITUDE_2) * 10 ** (-6)} "
            f"+lon_0={nc3_read_att(inDataset, OldIfr.VARIABLE_LAMBERT_CENTRAL_MERIDIAN) * 10 ** (-6)} "
            f"+k_0={nc3_read_att(inDataset, OldIfr.VARIABLE_LAMBERT_FACTOR) * 10 ** (-6)} "
            f"+x_0={nc3_read_att(inDataset, OldIfr.VARIABLE_LAMBERT_X_ORIGIN)} "
            f"+y_0={nc3_read_att(inDataset, OldIfr.VARIABLE_LAMBERT_Y_ORIGIN)}"
        )

    return lambert_proj4


def get_utm_proj4(inDataset: nc.Dataset) -> str:
    """
    Returns a proj4 string corresponding to UTM projection defined in old DTM
    :param inDataset:
    :return: proj4 formated string
    """

    # UTM projection : parameters as a function UTM_type
    UTM_Hemisphere = {1: None, 2: "+south", 3: None}

    utm_type = nc3_read_att(inDataset, OldIfr.VARIABLE_UTM_TYPE)
    utm_zone = nc3_read_att(inDataset, OldIfr.VARIABLE_UTM_ZONE)

    if utm_type and utm_zone:
        utm_proj4 = (
            f"+proj=utm "
            f"+zone={utm_zone}"
            f"{UTM_Hemisphere[nc3_read_att(inDataset, OldIfr.VARIABLE_UTM_HEMISPHERE)]}"
        )
    else:
        utm_proj4 = (
            f"+proj=tmerc "
            f"+lon_0={nc3_read_att(inDataset, OldIfr.VARIABLE_UTM_CENTRAL_MERIDIAN) * 10 ** (-6)} "
            f"+x_0={nc3_read_att(inDataset, OldIfr.VARIABLE_UTM_X_ORIGIN)} "
            f"+y_0={nc3_read_att(inDataset, OldIfr.VARIABLE_UTM_Y_ORIGIN)} "
        )

    return utm_proj4


def get_mercator_proj4(inDataset: nc.Dataset) -> str:
    """
    Returns a proj4 string corresponding to Mercator projection defined in old DTM
    :param inDataset:
    :return: proj4 formated string
    """
    mercator_x_origin = nc3_read_att(inDataset, OldIfr.VARIABLE_MERCATOR_X_ORIGIN)
    mercator_y_origin = nc3_read_att(inDataset, OldIfr.VARIABLE_MERCATOR_Y_ORIGIN)
    return (
        f"+proj=merc "
        f"+lat_ts={nc3_read_att(inDataset, OldIfr.VARIABLE_MERCATOR_REFERENCE_LATITUDE) * 10 ** (-6)} "
        f"+lon_0={nc3_read_att(inDataset, OldIfr.VARIABLE_MERCATOR_CENTRAL_MERIDIAN) * 10 ** (-6)} "
        f"+x_0={mercator_x_origin if mercator_x_origin is not None else 0} "
        f"+y_0={mercator_y_origin if mercator_y_origin is not None else 0}"
    )


def get_stereo_proj4(inDataset: nc.Dataset) -> str:
    """
    Returns a proj4 string corresponding to Stereo polar projection defined in old DTM
    :param inDataset:
    :return:proj4 formated string
    """
    return (
        f"+proj=stere "
        f"+lat_ts={nc3_read_att(inDataset, OldIfr.VARIABLE_POLAR_REFERENCE_LATITUDE) * 10 ** (-6)} "
        f"+lat_0={nc3_read_att(inDataset, OldIfr.VARIABLE_POLAR_HEMISPHERE)}"
        f"+lon_0={nc3_read_att(inDataset, OldIfr.VARIABLE_POLAR_CENTRAL_MERIDIAN) * 10 ** (-6)}"
    )


def get_eqc_proj4(inDataset: nc.Dataset) -> str:
    """
    Returns a proj4 string corresponding to Equidistant cylindrical projection defined in old DTM
    :param inDataset:
    :return:proj4 formated string
    """
    return (
        f"+proj=eqc "
        f"+lat_ts={nc3_read_att(inDataset, OldIfr.VARIABLE_CYLINDRICED_REFERENCE_LATITUDE) * 10 ** (-6)} "
        f"+lon_0={nc3_read_att(inDataset, OldIfr.VARIABLE_CYLINDRICED_CENTRAL_MERIDIAN) * 10 ** (-6)}"
    )


def nc3_get_proj4(inDataset: nc.Dataset) -> str:
    """
    Returns a proj4 string equivalent of the spatial reference of the old DTM file
    :param inDataset:
    :return:
    """

    proj = {1: get_mercator_proj4, 2: get_lambert_proj4, 3: get_utm_proj4, 4: get_stereo_proj4, 5: get_eqc_proj4}

    # Ellipsoid
    ellips_cartolib_to_proj4 = {2: "intl", 3: "WGS72", 4: "bessel", 6: "clrk66", 9: "WGS66", 14: "WGS84", 16: "GRS80"}

    proj_code = nc3_read_att(inDataset, OldIfr.VARIABLE_PROJECTION)
    ellps_code = nc3_read_att(inDataset, OldIfr.VARIABLE_ELLIPSOID)

    if ellps_code in ellips_cartolib_to_proj4.keys():
        ellps_prj4_str = f"+ellps={ellips_cartolib_to_proj4[ellps_code]}"
    else:
        ellps_prj4_str = (
            f"+a={inDataset.variables[OldIfr.VARIABLE_HALF_GREAT_AXIS][0] * 10 ** -2} "
            f"+e={inDataset.variables[OldIfr.VARIABLE_SQUARE_ECCENTRICITY][0] * 10 ** -10}"
        )

    return f"{proj[proj_code](inDataset)} {ellps_prj4_str}"


def nc3_read_att(inDataset: nc.Dataset, varname):
    """
    Decode Caraibes DTM v0.0 attributes stocked as variables
    TODO : Should be in a DTM v0.0 driver elsewhere
    """
    try:
        var_value = inDataset.variables[varname][:]
        if var_value.mask.all():
            var_value = None
        elif len(var_value.shape) > 0:
            if isinstance(var_value[0], np.bytes_) and f"{var_value[0]}"[0:3] == "b'\\":
                if var_value.size > 1:
                    var_value = [ord(s) for s in var_value]
                else:
                    var_value = ord(var_value[0])
            elif isinstance(var_value[0], np.bytes_) and f"{var_value[0]}"[0:2] == "b'":
                var_value = "".join([s.decode("UTF-8") for s in var_value if isinstance(s, np.bytes_)])
        else:
            if isinstance(var_value, np.bytes_) and f"{var_value[0]}"[0:3] == "b'\\":
                var_value = ord(var_value)
            elif isinstance(var_value, np.bytes_) and f"{var_value[0]}"[0:2] == "b'":
                var_value = var_value.decode("UTF-8")
        return var_value

    except KeyError:
        return None


def nc3_read_var(inDataset: nc.Dataset, varname):
    """
    Read Caraibes DTM v0.0 X variables and scale it accordingly
    Y = A_resol x X + B_reso
    """
    missing_value = nc3_read_att(inDataset, OldIfr.VARIABLE_SIGN_VALUE)[0]
    A = inDataset.variables[OldIfr.VARIABLE_A_RESOL][:]
    a = A[A != 0]
    b = inDataset.variables[OldIfr.VARIABLE_B_RESOL][A != 0]
    data = inDataset.variables[varname][:].astype(np.float32)
    data[data == missing_value] = np.nan
    varvalue = a * data + b

    return varvalue


def proj4_from_EPSG(epsg_code) -> str:
    """
    returns a proj4 string equivalent to given EPSG code
    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    return srs.ExportToProj4()


def read_history_new(inDataset: nc.Dataset) -> str:
    """
    Read and format history attribute from old DTM dataset
    :param inDataset:
    :return: formatted history string
    """
    hist_julian_date = inDataset.variables[OldIfr.ATT_MB_HISTORY_DATE]
    hist_julian_date.set_auto_mask(False)
    hist_time_in_ms = inDataset.variables[OldIfr.ATT_MB_HISTORY_TIME]
    hist_time_in_ms.set_auto_mask(False)
    hist_autor = inDataset.variables[OldIfr.ATT_MB_HISTORY_AUTOR]
    hist_autor.set_auto_mask(False)
    hist_module = inDataset.variables[OldIfr.ATT_MB_HISTORY_MODULE]
    hist_module.set_auto_mask(False)
    hist_comment = inDataset.variables[OldIfr.ATT_MB_HISTORY_COMMENT]
    hist_comment.set_auto_mask(False)

    history = []
    for hist_index in range(inDataset.mbNbrHistoryRec):
        time_stamp = (hist_julian_date[hist_index] - 2440588) * 24 * 3600 + (hist_time_in_ms[hist_index] / 1000)
        if time_stamp < 0:
            # skip unused modification history entries
            continue
        hist_datetime = datetime.fromtimestamp(time_stamp, timezone.utc)
        history.append(
            f"{hist_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')} {str(nc.chartostring(hist_module[hist_index]))} by {str(nc.chartostring(hist_autor[hist_index]))} {str(nc.chartostring(hist_comment[hist_index]))}"
        )

    return history


def read_history_old(inDataset: nc.Dataset) -> str:
    """
    Returns formatted history string from old DTM dataset

    :param inDataset: a DTM netcdf dataset in old format with creation/modification history variables
    :type inDataset: nc.Dataset
    :return: Description
    :rtype: str
    """

    history = []
    # Creation
    creation_date = inDataset.variables[OldIfr.VARIABLE_CREATION_DAY]
    creation_hour = inDataset.variables[OldIfr.VARIABLE_CREATION_HOUR]
    creation_name = nc3_read_att(inDataset, OldIfr.VARIABLE_CREATION_NAME)
    creation_module = nc3_read_att(inDataset, OldIfr.VARIABLE_CREATION_MODULE)
    time_stamp = (creation_date[0] - 2440588) * 24 * 3600 + (creation_hour[0] / 1000)
    if time_stamp >= 0:
        creation_datetime = datetime.fromtimestamp(time_stamp, timezone.utc)
        history.append(
            f"{creation_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')} {'creation' if creation_module is None else creation_module} by {'unknown' if creation_name is None else creation_name}"
        )
    # History of modifications {"unknown" if creation_name is None else creation_name}
    modif_date = inDataset.variables[OldIfr.VARIABLE_MODIF_DAY]
    modif_hour = inDataset.variables[OldIfr.VARIABLE_MODIF_HOUR]
    modif_name = nc3_read_att(inDataset, OldIfr.VARIABLE_MODIF_NAME)
    modif_type = nc3_read_att(inDataset, OldIfr.VARIABLE_MODIF_TYPE)
    if np.any(modif_date[:]):
        for hist_index in range(inDataset.dimensions["MODIFICATIONS"].size):
            time_stamp = (modif_date[hist_index] - 2440588) * 24 * 3600 + (modif_hour[hist_index] / 1000)
            if time_stamp < 0:
                # skip unused modification history entries
                continue
            modif_datetime = datetime.fromtimestamp(time_stamp, timezone.utc)
            history.append(
                f"{modif_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')} {str(nc.chartostring(modif_type[hist_index]))} by {str(nc.chartostring(modif_name[hist_index]))}"
            )
    return history


def read_history(inDataset: nc.Dataset) -> str:
    """
    Read and format history attribute from old DTM dataset
    """
    if OldIfr.VARIABLE_CREATION_DAY in inDataset.variables:
        return read_history_old(inDataset)
    else:
        return read_history_new(inDataset)


def get_metadata(inDataset: nc.Dataset) -> Dict[str, str]:
    """
    Extract metadata from old DTM dataset
    :param inDataset:
    :return: dictionary of metadata
    """
    filepath = Path(nc_encoding.filepath(inDataset))
    history = read_history(inDataset)
    # append migration info into history
    history.append(f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} Upgraded from {filepath.name}")
    metadata = {
        "title": filepath.stem,
        "institution": "ifremer",
        "source": str(filepath.relative_to(filepath.anchor)),
        "history": ", ".join(history[::-1]),  # reverse to have newer first
        "references": f"{str(filepath.relative_to(filepath.anchor))}",
        "comment": f"Upgraded with PyAt DTMMigrate script from file cited in reference above.",
    }

    return metadata


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------------------
    i_path = r"C:\Users\agaillot\Downloads\Levante_margin.dtm"
    o_path = r"C:\Users\agaillot\Downloads\Levante_margin.dtm.nc"

    process = DTMMigrate(i_paths=[i_path], o_paths=[o_path], overwrite=True)
    process()
