import errno
import logging
import os
from typing import Optional

import netCDF4 as nc
import numpy as np
from osgeo import osr

DEFAULT_COMPRESSION_LIB = "zlib"
DEFAULT_COMPRESSION_LEVEL = 4


def copy_and_compress_dataset(src: nc.Dataset, out_file) -> None:
    """
    Copy a NetCDF dataset and compress variables, preserving global and variables attributes
    :param src: a netcdf dataset
    :param out_file: output file path
    :return:
    """
    with nc.Dataset(out_file, "w") as dst:
        # copy global attributes all at once
        dst.setncatts(src.__dict__)
        # copy dimensions
        duplicate_dimensions(src, dst)
        # copy and compres all variables preserving variables attributes
        for name, i_var in src.variables.items():

            ncatts = {v: i_var.getncattr(v) for v in i_var.ncattrs() if not (i_var._isvlen and v == "_FillValue")}
            compression = DEFAULT_COMPRESSION_LIB if not i_var._isvlen else None

            o_var = dst.createVariable(
                name,
                i_var.datatype,
                i_var.dimensions,
                compression=compression,
                complevel=DEFAULT_COMPRESSION_LEVEL,
            )

            o_var.setncatts(ncatts)
            o_var[:] = i_var[:]


def duplicate_dataset(input_ds: nc.Dataset, output_ds: nc.Dataset) -> None:
    """
    copy one dataset from input to another dataset
    """
    # copy group attributes
    for att in input_ds.ncattrs():
        output_ds.setncattr(att, input_ds.getncattr(att))
    # copy cmptypes declaration
    for name, cmp in input_ds.cmptypes.items():
        output_ds.createCompoundType(datatype=cmp.dtype, datatype_name=name)
    # copy enum types declaration
    for name, enum in input_ds.enumtypes.items():
        output_ds.createEnumType(datatype=enum.dtype, datatype_name=name, enum_dict=enum.enum_dict)
    # copy vlen types declaration
    for name, vl in input_ds.vltypes.items():
        output_ds.createVLType(datatype=vl.dtype, datatype_name=name)
    # Copy dimensions
    for dname, the_dim in input_ds.dimensions.items():
        output_ds.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)
    # Copy variables
    for name, variable in input_ds.variables.items():
        duplicate_variable(variable, output_ds)

    # copy subgroups recursively
    for name in input_ds.groups:
        out_subgroup = output_ds.createGroup(name)
        duplicate_dataset(input_ds.groups[name], out_subgroup)


def duplicate_variable(i_var: nc.Variable, o_dataset: nc.Dataset) -> None:
    """Create the same variable (same attributes and  data) in the output Dataset than the specified variable

    Arguments:
        i_var -- input variable.
        o_dataset -- output dataset.
    """
    # retrieve compression parameters
    complib = None
    complevel = DEFAULT_COMPRESSION_LEVEL
    chunksizes = None
    if i_var.filters() is not None and not i_var._isvlen:
        complib = DEFAULT_COMPRESSION_LIB if i_var.filters().get(DEFAULT_COMPRESSION_LIB, False) else None
        complevel = i_var.filters().get("complevel", DEFAULT_COMPRESSION_LEVEL)
    if i_var.chunking() is not None:
        chunksizes = tuple(i_var.chunking()) if i_var.chunking() != "contiguous" else None

    # don’t pass fill_value when duplicating enum variables
    if is_enum_var(i_var):
        fill_value = None
    else:
        fill_value = getattr(i_var, "_FillValue", None)

    # disable automatic masking and scaling to avoid issues when copying data to avoid datatype mismatch
    i_var.set_auto_maskandscale(False)

    o_var = o_dataset.createVariable(
        i_var.name,
        i_var.datatype,
        i_var.dimensions,
        fill_value=fill_value,
        compression=complib,
        complevel=complevel,
        chunksizes=chunksizes,
    )
    # o_var.setncatts(i_var.__dict__)
    # Copy all attributes except _FillValue to avoid issues with enum variables
    o_var.setncatts({k: v for k, v in i_var.__dict__.items() if k != "_FillValue"})
    o_var[:] = i_var[:]


def is_enum_var(i_var):
    """Return True if i_var appears to use a netCDF4 enum type."""
    dt = getattr(i_var, "datatype", None)
    if dt is None:
        return False

    # Common: netCDF4 Datatype has 'enum_dict' attribute
    if getattr(dt, "enum_dict", None):
        return True

    # Some versions expose an isenum flag
    if getattr(dt, "isenum", False):
        return True

    # Fallback: check parent's enumtypes mapping for exact datatype match
    try:
        ds = getattr(i_var, "dataset", None) or getattr(i_var, "group", None)
        if ds is not None and hasattr(ds, "enumtypes"):
            for name, enumtype in ds.enumtypes.items():
                if enumtype == dt:
                    return True
    except Exception:
        pass

    return False


def duplicate_dimensions(i_dataset: nc.Dataset, o_dataset: nc.Dataset) -> None:
    """Copy all dimensions based on the input file.

    Arguments:
        i_dataset -- input dataset.
        o_dataset -- output dataset.
    """
    for name, dimension in i_dataset.dimensions.items():
        o_dataset.createDimension(name, len(dimension) if not dimension.isunlimited() else None)


def open_read(path: str, nc_format="NETCDF4") -> nc.Dataset:
    """Open the nc file with path in a read mode.

    Arguments:
        path {str} -- Path of the file.
        nc_format -- underlying file format .

    Raises:
        SystemError: File is not in the expected format
        FileNotFoundError: Doesn't find the file.
    """
    logging.getLogger(__name__).info(f"Open file {path} in read mode.")
    return open_ncfile(path, "r", nc_format)


def create_file(path: str, nc_format="NETCDF4", overwrite=True) -> nc.Dataset:
    """Create the nc file with path.

    Arguments:
        path {str} -- Path of the file.
        nc_format -- underlying file format.
        overwrite -- True to overwrite the file if exists

    Raises:
        FileExistsError: file exists and overwrite not allowed
        SystemError: File is not in the expected format
    """
    logging.getLogger(__name__).info(f"Create file {path}.")
    if not os.path.exists(path) or overwrite:
        return open_ncfile(path, "w", nc_format)
    raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), path)


def open_modify(path: str, nc_format="NETCDF4") -> nc.Dataset:
    """Open the nc file with path in a modify mode.

    Arguments:
        path {str} -- Path of the file.
        nc_format -- underlying file format .

    Raises:
        SystemError: File is not in the expected format
        FileNotFoundError: Doesn't find the file.
    """
    logging.getLogger(__name__).info(f"Open file {path} in modify mode.")
    return open_ncfile(path, "r+", nc_format)


def open_ncfile(path: str, nc_mode="r", nc_format="NETCDF4") -> nc.Dataset:
    """Open the nc file with path in a specified mode.

    Arguments:
        path {str} -- Path of the file.
        mode -- access mode ("r" means read-only; "w" new file is created; "r+" mean append
        nc_format -- underlying file format .

    Raises:
        SystemError: File is not in the expected format
        FileNotFoundError: Doesn't find the file.
    """
    try:
        result = nc.Dataset(path, mode=nc_mode)
        if result.file_format != nc_format:
            result.close()
            raise SystemError(f"The format of the file {path} must be {nc_format} (not {result.file_format}).")
        return result
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dtm file {path} not found.") from e


def set_history_attr(o_dataset: nc.Dataset, process_name: str, i_paths=None, append=True) -> None:
    """Appened the history global attribute. If no history attribute, create and fill it with
        - name of the process
        - file(s)
    If history attribute exists, just add the process after it.

    Arguments:
        o_dataset -- output dataset.
        process_name -- name of the python process.
        i_paths -- input files if any.
        append -- if false, history is replaced
    """

    if o_dataset.history and append:
        o_dataset.history = o_dataset.history + f", process with Python {process_name}"
    else:
        o_dataset.history = f"Process with Python {process_name}"

    if i_paths:
        if isinstance(i_paths, list):
            o_dataset.history = o_dataset.history + f' from {", ".join(i_paths)}.'
        else:
            o_dataset.history = o_dataset.history + f" from {i_paths}."


def get_variable(i_dataset: nc.Dataset, variable_path: str) -> Optional[nc.Variable]:
    """return the nc variable designated by the path layer_path"""
    path = variable_path.split("/")
    variable_name = path.pop()
    parent_group = i_dataset
    for sub_group in path:
        if sub_group:
            if sub_group not in parent_group.groups:
                return None
            parent_group = parent_group.groups[sub_group]
    return parent_group.variables[variable_name] if variable_name in parent_group.variables else None


def get_group(i_dataset: nc.Dataset, group_path: str) -> Optional[nc.Group]:
    """return the nc group designated by the path group_path"""
    path = [p for p in group_path.split("/") if p]  # Remove empty strings from path (can happen if path ends with "/")
    group_name = path.pop()
    parent_group = i_dataset
    for sub_group in path:
        if sub_group:
            if sub_group not in parent_group.groups:
                return None
            parent_group = parent_group.groups[sub_group]
    return parent_group.groups[group_name] if group_name in parent_group.groups else None


# __        ___  _______  ______  ____   ___      _ _  _     _                       _       _   _               _           ____ _____   _   __
# \ \      / / |/ /_   _|/ /  _ \|  _ \ / _ \    | | || |   | |_ _ __ __ _ _ __  ___| | __ _| |_(_) ___  _ __   | |_ ___    / ___|  ___| / | / /_
#  \ \ /\ / /| ' /  | | / /| |_) | |_) | | | |_  | | || |_  | __| '__/ _` | '_ \/ __| |/ _` | __| |/ _ \| '_ \  | __/ _ \  | |   | |_    | || '_ \
#   \ V  V / | . \  | |/ / |  __/|  _ <| |_| | |_| |__   _| | |_| | | (_| | | | \__ \ | (_| | |_| | (_) | | | | | || (_) | | |___|  _|   | || (_) |
#    \_/\_/  |_|\_\ |_/_/  |_|   |_| \_\\___/ \___(_) |_|    \__|_|  \__,_|_| |_|___/_|\__,_|\__|_|\___/|_| |_|  \__\___/   \____|_|     |_(_)___/
#
# See https://cfconventions.org/wkt-proj-4.html


def __translate_spatial_reference_epsg_9804(spatial_ref: osr.SpatialReference) -> dict:
    """
    Convert a Mercator EPSG 9804 SpatialReference to netcdf compliant attributes.
    Mercator Variant A is defined with the equator as the single standard parallel, with scale factor on the equator also defined.
    GDAL id = osr.SRS_PT_MERCATOR_1SP
    Proj4 = +proj=merc
    """
    return {
        "grid_mapping_name": "mercator",
        "longitude_of_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN),
        "scale_factor_at_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_SCALE_FACTOR),
        "false_easting": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_EASTING),
        "false_northing": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_NORTHING),
    }


def __translate_spatial_reference_epsg_9805(spatial_ref: osr.SpatialReference) -> dict:
    """
    Convert a Mercator EPSG 9805 SpatialReference to netcdf compliant attributes
    Mercator Variant B is defined through the latitude of two parallels equidistant either side of the equator upon which the grid scale is true
    GDAL id = osr.SRS_PT_LAMBERT_CONFORMAL_CONIC_2SP
    Proj4 = +proj=merc
    """
    return {
        "grid_mapping_name": "mercator",
        "longitude_of_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN),
        "standard_parallel": spatial_ref.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1),
        "false_easting": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_EASTING),
        "false_northing": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_NORTHING),
    }


def __translate_spatial_reference_aea(spatial_ref: osr.SpatialReference) -> dict:
    """
    Convert a Albers Conical Equal Area SpatialReference to netcdf compliant attributes
    GDAL id = osr.SRS_PT_ALBERS_CONIC_EQUAL_AREA
    Proj4 = +proj=aea
    """
    return {
        "grid_mapping_name": "albers_conical_equal_area",
        "standard_parallel": [
            spatial_ref.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1),
            spatial_ref.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_2),
        ],
        "longitude_of_central_meridian": spatial_ref.GetProjParm(osr.SRS_PP_LONGITUDE_OF_CENTER),
        "latitude_of_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_LATITUDE_OF_CENTER),
        "false_easting": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_EASTING),
        "false_northing": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_NORTHING),
    }


def __translate_spatial_reference_aeqd(spatial_ref: osr.SpatialReference) -> dict:
    """
    Convert a Azimuthal Equidistant SpatialReference to netcdf compliant attributes
    GDAL id = osr.SRS_PT_AZIMUTHAL_EQUIDISTANT
    Proj4 = +proj=aeqd
    """
    return {
        "grid_mapping_name": "azimuthal_equidistant",
        "longitude_of_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_LONGITUDE_OF_CENTER),
        "latitude_of_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_LATITUDE_OF_CENTER),
        "false_easting": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_EASTING),
        "false_northing": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_NORTHING),
    }


def __translate_spatial_reference_laea(spatial_ref: osr.SpatialReference) -> dict:
    """
    Convert a Lambert Azimuthal Equal Area SpatialReference to netcdf compliant attributes
    GDAL id = osr.SRS_PT_LAMBERT_AZIMUTHAL_EQUAL_AREA
    Proj4 = +proj=laea
    """
    return {
        "grid_mapping_name": "lambert_azimuthal_equal_area",
        "longitude_of_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_LONGITUDE_OF_CENTER),
        "latitude_of_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_LATITUDE_OF_CENTER),
        "false_easting": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_EASTING),
        "false_northing": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_NORTHING),
    }


def __translate_spatial_reference_lcc_1sp(spatial_ref: osr.SpatialReference) -> dict:
    """
    Convert a Lambert conformal SpatialReference to netcdf compliant attributes
    GDAL id = osr.SRS_PT_LAMBERT_CONFORMAL_CONIC_1SP
    Proj4 = +proj=lcc
    """
    return {
        "grid_mapping_name": "lambert_conformal_conic",
        "standard_parallel": spatial_ref.GetProjParm(osr.SRS_PP_LATITUDE_OF_ORIGIN),
        "longitude_of_central_meridian": spatial_ref.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN),
        "latitude_of_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_LATITUDE_OF_ORIGIN),
        # Scale factor at natural origin not in CF (always 1)
        "false_easting": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_EASTING),
        "false_northing": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_NORTHING),
    }


def __translate_spatial_reference_lcc_2sp(spatial_ref: osr.SpatialReference) -> dict:
    """
    Convert a Lambert conformal SpatialReference to netcdf compliant attributes
    GDAL id = osr.SRS_PT_LAMBERT_CONFORMAL_CONIC_2SP
    Proj4 = +proj=lcc
    """
    return {
        "grid_mapping_name": "lambert_conformal_conic",
        "standard_parallel": [
            spatial_ref.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1),
            spatial_ref.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_2),
        ],
        "longitude_of_central_meridian": spatial_ref.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN),
        "latitude_of_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_LATITUDE_OF_ORIGIN),
        "false_easting": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_EASTING),
        "false_northing": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_NORTHING),
    }


def __translate_spatial_reference_cea(spatial_ref: osr.SpatialReference) -> dict:
    """
    Convert a Lambert cylindrical equal area to netcdf compliant attributes
    GDAL id = osr.SRS_PT_CYLINDRICAL_EQUAL_AREA
    Proj4 = +proj=cea
    """
    return {
        "grid_mapping_name": "lambert_cylindrical_equal_area",
        "standard_parallel": spatial_ref.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1),
        "longitude_of_central_meridian": spatial_ref.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN),
        "false_easting": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_EASTING),
        "false_northing": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_NORTHING),
    }


def __translate_spatial_reference_ortho(spatial_ref: osr.SpatialReference) -> dict:
    """
    Convert a Orthographic projection to netcdf compliant attributes
    GDAL id = osr.SRS_PT_ORTHOGRAPHIC
    Proj4 = +proj=ortho
    """
    return {
        "grid_mapping_name": "orthographic",
        "longitude_of_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN),
        "latitude_of_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_LATITUDE_OF_ORIGIN),
        "false_easting": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_EASTING),
        "false_northing": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_NORTHING),
    }


def __translate_spatial_reference_stere(spatial_ref: osr.SpatialReference) -> dict:
    """
    Convert a Stereographic projection to netcdf compliant attributes
    GDAL id = osr.SRS_PT_STEREOGRAPHIC
    Proj4 = +proj=stere
    """
    return {
        "grid_mapping_name": "stereographic",
        "longitude_of_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN),
        "latitude_of_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_LATITUDE_OF_ORIGIN),
        "scale_factor_at_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_SCALE_FACTOR),
        "false_easting": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_EASTING),
        "false_northing": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_NORTHING),
    }


def __translate_spatial_reference_polar_stere(spatial_ref: osr.SpatialReference) -> dict:
    """
    Convert a Polar stereographic projection to netcdf compliant attributes
    GDAL id = osr.SRS_PT_POLAR_STEREOGRAPHIC
    Proj4 = +proj=stere
    """
    latitude_of_origin = spatial_ref.GetProjParm(osr.SRS_PP_LATITUDE_OF_ORIGIN)
    result = {
        "grid_mapping_name": "polar_stereographic",
        "latitude_of_projection_origin": 90.0 if latitude_of_origin >= 0.0 else -90.0,
        "straight_vertical_longitude_from_pole": spatial_ref.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN),
        "false_easting": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_EASTING),
        "false_northing": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_NORTHING),
    }

    if latitude_of_origin in [-90.0, 90.0]:
        # Polar stereographic PS-A
        result["scale_factor_at_projection_origin"] = spatial_ref.GetProjParm(osr.SRS_PP_SCALE_FACTOR)
    else:
        # Polar stereographic PS-B
        result["standard_parallel"] = latitude_of_origin

    return result


def __translate_spatial_reference_tmerc(spatial_ref: osr.SpatialReference) -> dict:
    """
    Convert a Transverse Mercator projection to netcdf compliant attributes
    GDAL id = osr.SRS_PT_TRANSVERSE_MERCATOR
    Proj4 = +proj=tmerc
    """
    return {
        "grid_mapping_name": "transverse_mercator",
        "scale_factor_at_central_meridian": spatial_ref.GetProjParm(osr.SRS_PP_SCALE_FACTOR),
        "longitude_of_central_meridian": spatial_ref.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN),
        "latitude_of_projection_origin": spatial_ref.GetProjParm(osr.SRS_PP_LATITUDE_OF_ORIGIN),
        "false_easting": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_EASTING),
        "false_northing": spatial_ref.GetProjParm(osr.SRS_PP_FALSE_NORTHING),
    }


SPATIAL_REFERENCE_TRANSLATORS = {
    osr.SRS_PT_MERCATOR_1SP: __translate_spatial_reference_epsg_9804,
    osr.SRS_PT_MERCATOR_2SP: __translate_spatial_reference_epsg_9805,
    osr.SRS_PT_ALBERS_CONIC_EQUAL_AREA: __translate_spatial_reference_aea,
    osr.SRS_PT_AZIMUTHAL_EQUIDISTANT: __translate_spatial_reference_aeqd,
    osr.SRS_PT_LAMBERT_AZIMUTHAL_EQUAL_AREA: __translate_spatial_reference_laea,
    osr.SRS_PT_LAMBERT_CONFORMAL_CONIC_1SP: __translate_spatial_reference_lcc_1sp,
    osr.SRS_PT_LAMBERT_CONFORMAL_CONIC_2SP: __translate_spatial_reference_lcc_2sp,
    osr.SRS_PT_CYLINDRICAL_EQUAL_AREA: __translate_spatial_reference_cea,
    osr.SRS_PT_ORTHOGRAPHIC: __translate_spatial_reference_ortho,
    osr.SRS_PT_STEREOGRAPHIC: __translate_spatial_reference_stere,
    osr.SRS_PT_POLAR_STEREOGRAPHIC: __translate_spatial_reference_polar_stere,
    osr.SRS_PT_TRANSVERSE_MERCATOR: __translate_spatial_reference_tmerc,
}


def is_projection_supported(projection: str) -> bool:
    """
    Return true if the specified projection is translatable
    """
    return projection in SPATIAL_REFERENCE_TRANSLATORS


def is_spatial_reference_supported(spatial_ref: osr.SpatialReference) -> bool:
    """
    Return true if the specified projection is translatable
    """
    return spatial_ref.GetAttrValue("PROJECTION") in SPATIAL_REFERENCE_TRANSLATORS


def translate_spatial_reference(spatial_ref: osr.SpatialReference) -> dict:
    """
    Convert a SpatialReference to netcdf compliant attributes
    """
    projection = spatial_ref.GetAttrValue("PROJECTION")
    if is_projection_supported(projection):
        return SPATIAL_REFERENCE_TRANSLATORS[projection](spatial_ref)
    # Projection not supported
    raise ValueError(f"Specified projection is not compliant with the Netcdf convention : {projection}")


def create_longitude_variable(
    parent_group: nc.Group, variable_name: str, dimension_name: str, longitudes: np.ndarray
) -> nc.Variable:
    """
    Create a longitude variable in the specified parent group with the specified name and dimension, and fill it with the provided longitudes values. The variable is compliant with CF and SeaDataNet conventions.

    Arguments:
        parent_group -- the netcdf group in which the variable will be created
        variable_name -- name of the longitude variable to create
        dimension_name -- name of the dimension to create for the longitude variable
        longitudes -- array of longitude values to fill in the variable
    """
    parent_group.createDimension(dimension_name, len(longitudes))
    result = parent_group.createVariable(variable_name, "f8", ("lon"))
    result.standard_name = "longitude"
    result.long_name = "longitude"
    result.units = "degrees_east"
    result.valid_range = [-180.0, 180.0]
    result.sdn_parameter_urn = "SDN:P01::ALONZZ01"
    result.sdn_parameter_uri = "https://vocab.nerc.ac.uk/collection/P01/current/ALONZZ01/"
    result.sdn_parameter_name = "Longitude east"
    result.sdn_uom_urn = "SDN:P06::DEGE"
    result.sdn_uom_name = "Degrees east"
    result.sdn_uom_uri = "https://vocab.nerc.ac.uk/collection/P06/current/DEGE/"
    result[:] = longitudes
    return result


def create_latitude_variable(
    parent_group: nc.Group, variable_name: str, dimension_name: str, latitudes: np.ndarray
) -> nc.Variable:
    """
    Create a longitude variable in the specified parent group with the specified name and dimension, and fill it with the provided longitudes values. The variable is compliant with CF and SeaDataNet conventions.

    Arguments:
        parent_group -- the netcdf group in which the variable will be created
        variable_name -- name of the longitude variable to create
        dimension_name -- name of the dimension to create for the longitude variable
        longitudes -- array of longitude values to fill in the variable
    """
    parent_group.createDimension(dimension_name, len(latitudes))
    result = parent_group.createVariable(variable_name, "f8", ("lat"))
    result.standard_name = "latitude"
    result.long_name = "latitude"
    result.units = "degrees_north"
    result.valid_range = [-90.0, 90.0]
    result.sdn_parameter_urn = "SDN:P01::ALATZZ01"
    result.sdn_parameter_uri = "https://vocab.nerc.ac.uk/collection/P01/current/ALATZZ01/"
    result.sdn_parameter_name = "Latitude north"
    result.sdn_uom_urn = "SDN:P06::DEGN"
    result.sdn_uom_name = "Degrees north"
    result.sdn_uom_uri = "https://vocab.nerc.ac.uk/collection/P06/current/DEGN/"
    result.units = "degrees_north"
    result[:] = latitudes
    return result


def create_crs_variable(parent_group: nc.Group, variable_name: str) -> nc.Variable:
    """Create a coordinate reference system variable in the specified parent group with the specified name. The variable is compliant with CF and SeaDataNet conventions."""
    crs = parent_group.createVariable(variable_name, "int", ())
    crs.grid_mapping_name = "latitude_longitude"
    crs.longitude_of_prime_meridian = 0.0
    crs.semi_major_axis = 6378137.0
    crs.inverse_flattening = 298.257223563
    crs.epsg_code = "EPSG:4326"
    crs.spatial_ref = "GEOGCS['WGS 84',DATUM['WGS_1984',SPHEROID['WGS 84',6378137,298.257223563,AUTHORITY['EPSG','7030']],AUTHORITY['EPSG','6326']],PRIMEM['Greenwich',0,AUTHORITY['EPSG','8901']],UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']],AUTHORITY['EPSG','4326']]"
