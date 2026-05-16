from typing import Dict
import xarray as xr
import netCDF4 as nc


def _convert_time_to_ms(ds: xr.Dataset, time_variable: [str]):
    """Xarray does not handle nanosecond, this function will update and decode time values to ms"""
    if not time_variable:
        time_variable = ["ping_time"]
    xr.set_options(keep_attrs=True)
    for t in time_variable:
        # adjust ping_time, ns is not managed by xarray
        var = ds.getattr(t)
        source_data = var.to_numpy()
        # keep track of time values
        ds.coords[t] = var // 1_000_000  # convert to ms
        ds.coords[t].attrs["units"] = "milliseconds since 1970-01-01 00:00:00Z"
    ds = xr.decode_cf(ds)

    return ds


def get_nc_attribute(nc_variable_or_group: nc.Dataset) -> dict:
    """Read netcdf attributes and returns them in a dictionnary"""
    desc = {}
    for att in nc_variable_or_group.ncattrs():
        desc[att] = nc_variable_or_group.getncattr(att)
    return desc


def __get_dimension_in_group(
    dimensions: Dict[str, nc.Dimension],
    group: nc.Dataset,
) -> Dict[str, nc.Dimension]:
    missing_dimension_names = [x for x in dimensions if dimensions[x] is None]
    for dim in missing_dimension_names:
        if dim in group.dimensions:
            dimensions[dim] = group.dimensions[dim]
    missing_dimension_names = [x for x in dimensions if dimensions[x] is None]
    if len(missing_dimension_names) > 0:
        # recurse upward
        __get_dimension_in_group(dimensions, group.parent)
    return dimensions


def get_dimensions(variable: nc.Variable) -> Dict[str, nc.Dimension]:
    """Recurse netcdf groups upward to find variable associated dimensions"""
    dimension_names = variable.dimensions
    parent = variable.group()
    r = {key: None for key in dimension_names}
    __get_dimension_in_group(dimensions=r, group=parent)
    return r
