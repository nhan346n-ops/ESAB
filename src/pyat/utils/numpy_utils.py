import datetime
import math
import tempfile
import warnings
from typing import Optional, Tuple, Union

import numba
import numpy as np
import scipy.interpolate
from osgeo import osr
from pyproj import Transformer, crs

import pyat.utils.argument_utils as arg_util


def disable_warning():
    """disable all numpy warnings"""
    warnings.filterwarnings("ignore")


def interp1d_nan(values: np.ndarray):
    """Function to replace nan values by interpolated values in a 1D array"""
    indexes = np.arange(values.shape[0])
    (valid,) = np.where(np.isfinite(values))
    f = scipy.interpolate.interp1d(valid, values[valid], bounds_error=False, fill_value="extrapolate")
    return f(indexes)


def linear_interp_data(values, values_times, times, extrapolate=False):
    """
    Returns linearly interpolated data values whether values are masked or not
    No extrapolation by default : defaults to NaN outside data range.
    """
    if extrapolate:
        data_time_interpolator = scipy.interpolate.interp1d(
            values_times, values, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
    else:
        data_time_interpolator = scipy.interpolate.interp1d(values_times, values, kind="linear", bounds_error=False)
    # cannot apply function to masked array, so fill masked values with nan
    if isinstance(times, np.ma.MaskedArray):
        if np.issubdtype(times.dtype, np.floating):
            fill_val = np.nan
        else:
            # Use the largest possible value for that specific integer type
            fill_val = np.iinfo(times.dtype).max
        times = times.filled(fill_val)

    return data_time_interpolator(times)


def safe_interp(x, y, x_new, axis=-1, fill_value="extrapolate"):
    """
    Safely interpolate y(x) at new points x_new, handling masked arrays or NaNs.

    Parameters
    ----------
    x : 1D array-like
        Original x-coordinates.
    y : array-like
        Original y values. Can be masked, contain NaNs, or be multi-dimensional.
        Interpolation is applied along the specified axis.
    x_new : array-like
        New x-coordinates for interpolation.
    axis : int
        Axis along which to interpolate.
    fill_value : 'extrapolate' or float
        Value to use for out-of-bounds interpolation.

    Returns
    -------
    y_new : ndarray
        Interpolated values at x_new. Shape matches y except along interpolation axis.
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    x_new = np.asanyarray(x_new)

    # Step 1: Compute mask for invalid values
    if np.ma.isMaskedArray(y):
        mask = np.ma.getmaskarray(y)
    else:
        mask = np.zeros_like(y, dtype=bool)
    mask |= ~np.isfinite(y)

    # Step 2: Fill masked/NaN values with NaN for easy handling
    y_safe = np.where(mask, np.nan, y)

    # Step 3: Interpolation function along axis
    def interp_1d(row):
        valid = ~np.isnan(row)
        if np.count_nonzero(valid) < 2:
            # Not enough points to interpolate
            return np.full_like(x_new, np.nan, dtype=row.dtype)
        # Use np.interp for linear interpolation
        y_row = np.interp(
            x_new,
            x[valid],
            row[valid],
            left=np.nan if fill_value == "extrapolate" else fill_value,
            right=np.nan if fill_value == "extrapolate" else fill_value,
        )
        return y_row

    # Step 4: Apply along the interpolation axis
    y_new = np.apply_along_axis(interp_1d, axis, y_safe)

    return y_new


@numba.guvectorize(["void(float64[:,:], float64[:,:])"], "(r, c),(m, c2)", target="parallel", nopython=True, cache=True)
def minMaxOnFloat(inArray: np.ndarray, result: np.ndarray):
    """
    Return the minimum and the maximum values of an array along the column axis
    :param inArray : the input array to process
    :param result : the resulting array containing the minimum and the maximum for each column
    """
    for colIndex in range(result.shape[1]):
        for rowIndex in range(inArray.shape[0]):
            colValue = float(inArray[rowIndex, colIndex])
            if not np.isnan(colValue):
                # Can't use min() or np.fmin() because they raise a RuntimeWarning (invalid value encountered) ?
                result[0, colIndex] = (
                    colValue if np.isnan(result[0, colIndex]) else np.minimum(result[0, colIndex], colValue)
                )
                result[1, colIndex] = (
                    colValue if np.isnan(result[1, colIndex]) else np.maximum(result[1, colIndex], colValue)
                )


@numba.jit(nopython=True, parallel=True, cache=True)
def aggregate(
    inValues: np.ndarray, xIndex: int, yIndex: int, valueIndex: int, outArray: np.ndarray, factor: float = 1.0
):
    """
    Return the minimum and the maximum values of an array along the column axis
    :param inValues : values to aggregate.
    :param xIndex : column index in inValues, containing the column index of the destination cell
    :param yIndex : column index in inValues, containing the row index of the destination cell
    :param valueIndex : column index in inValues, containing the float values to apply to the destination cell
    :param outArray : the receiving array where values are aggregated
    :param factor : factor applied to values
    """
    for rowIndex in numba.prange(inValues.shape[0]):
        if not np.isnan(inValues[rowIndex, 0]) and not np.isnan(inValues[rowIndex, 1]):
            x = int(inValues[rowIndex, xIndex])
            y = int(inValues[rowIndex, yIndex])
            if not np.isnan(x) and not np.isnan(y) and not np.isnan(inValues[rowIndex, valueIndex]):
                outArray[y, x] = inValues[rowIndex, valueIndex] * factor


@numba.njit(cache=True, fastmath=False)
def compute_weighted_statistics(
    in_array: np.ndarray,
    x_array: np.ndarray,
    y_array: np.ndarray,
    in_weights: np.ndarray,
    out_x_array: Optional[np.ndarray] = None,
    out_y_array: Optional[np.ndarray] = None,
    out_last_array: Optional[np.ndarray] = None,
    out_min_array: Optional[np.ndarray] = None,
    out_mean_array: Optional[np.ndarray] = None,
    out_max_array: Optional[np.ndarray] = None,
    out_weighted_count_array: Optional[np.ndarray] = None,
    out_weighted_sum_array: Optional[np.ndarray] = None,
    out_filtered_array: Optional[np.ndarray] = None,
) -> None:
    """
    Project the values contained in in_array to the out_*_array using weights.
    Indexes (column and row) used to determine the cell in out_*_array are found in x_array and y_array.

    :param in_array : float values to aggregate.
    :param x_array : column index in in_array, containing the column index of the destination cell
    :param y_array : row index in in_array, containing the row index of the destination cell
    :param in_weights : weight values to apply to in_array values.
    :param out_x_array : the receiving array containing all x mean values per cell
    :param out_y_array : the receiving array containing all y mean values per cell
    :param out_last_array : the receiving array containing all last projected values per cell
    :param out_min_array : the receiving array containing all min values per cell
    :param out_mean_array : the receiving array containing all mean values per cell
    :param out_max_array : the receiving array containing all max values per cell
    :param out_weighted_count_array : the receiving array containing the sum of weights per cell
    :param out_filtered_array : the receiving array containing the number of values equals to nan
    :param out_sum_array : the receiving array containing all sum values per cell (weighted if in_weights is provided)

    """
    shape = (0, 0)
    if out_last_array is not None:
        shape = out_last_array.shape
    if out_min_array is not None:
        shape = out_min_array.shape
    if out_mean_array is not None:
        shape = out_mean_array.shape
    if out_max_array is not None:
        shape = out_max_array.shape
    if out_weighted_count_array is not None:
        shape = out_weighted_count_array.shape
    if out_weighted_sum_array is not None:
        shape = out_weighted_sum_array.shape
    if out_filtered_array is not None:
        shape = out_filtered_array.shape
    if shape == (0, 0):
        raise ValueError("At least one out array must be specified")
    if out_mean_array is not None and out_weighted_count_array is None:
        raise ValueError("out_weighted_count_array is mandatory to compute the mean values")

    for value, x, y, weight in zip(in_array.flat, x_array.flat, y_array.flat, in_weights.flat):
        if np.isnan(weight) or weight <= 0.0:
            continue
        col = math.floor(x)
        row = math.floor(y)
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            if not np.isnan(value):
                if out_last_array is not None:
                    out_last_array[row, col] = value
                if out_mean_array is not None and out_weighted_count_array is not None:
                    if np.isnan(out_mean_array[row, col]):
                        out_mean_array[row, col] = value
                    else:
                        out_mean_array[row, col] = (
                            (np.float64(out_mean_array[row, col]) * out_weighted_count_array[row, col])
                            + (np.float64(value) * weight)
                        ) / (out_weighted_count_array[row, col] + weight)
                if out_x_array is not None and out_weighted_count_array is not None:
                    if np.isnan(out_x_array[row, col]):
                        out_x_array[row, col] = x
                    else:
                        out_x_array[row, col] = (
                            (np.float64(out_x_array[row, col]) * out_weighted_count_array[row, col])
                            + (np.float64(x) * weight)
                        ) / (out_weighted_count_array[row, col] + weight)
                if out_y_array is not None and out_weighted_count_array is not None:
                    if np.isnan(out_y_array[row, col]):
                        out_y_array[row, col] = y
                    else:
                        out_y_array[row, col] = (
                            (np.float64(out_y_array[row, col]) * out_weighted_count_array[row, col])
                            + (np.float64(y) * weight)
                        ) / (out_weighted_count_array[row, col] + weight)
                if out_weighted_count_array is not None:
                    if out_weighted_count_array[row, col] <= 0:
                        out_weighted_count_array[row, col] = weight
                    else:
                        out_weighted_count_array[row, col] = out_weighted_count_array[row, col] + weight
                if out_min_array is not None:
                    if np.isnan(out_min_array[row, col]):
                        out_min_array[row, col] = value
                    else:
                        out_min_array[row, col] = min(out_min_array[row, col], value)
                if out_max_array is not None:
                    if np.isnan(out_max_array[row, col]):
                        out_max_array[row, col] = value
                    else:
                        out_max_array[row, col] = max(out_max_array[row, col], value)
                if out_weighted_sum_array is not None:
                    if np.isnan(out_weighted_sum_array[row, col]):
                        out_weighted_sum_array[row, col] = np.float64(value) * weight
                    else:
                        out_weighted_sum_array[row, col] = out_weighted_sum_array[row, col] + (
                            np.float64(value) * weight
                        )
            elif out_filtered_array is not None:
                if out_filtered_array[row, col] <= 0:
                    out_filtered_array[row, col] = 1
                else:
                    out_filtered_array[row, col] = out_filtered_array[row, col] + 1


@numba.njit(cache=True, fastmath=False)
def compute_statistics(
    in_array: np.ndarray,
    x_array: np.ndarray,
    y_array: np.ndarray,
    out_x_array: Optional[np.ndarray] = None,
    out_y_array: Optional[np.ndarray] = None,
    out_last_array: Optional[np.ndarray] = None,
    out_min_array: Optional[np.ndarray] = None,
    out_mean_array: Optional[np.ndarray] = None,
    out_max_array: Optional[np.ndarray] = None,
    out_count_array: Optional[np.ndarray] = None,
    out_sum_array: Optional[np.ndarray] = None,
    out_filtered_array: Optional[np.ndarray] = None,
) -> None:
    """
    Project the values contained in in_array to the out_*_array.
    Indexes (column and row) used to determine the cell in out_*_array are found in x_array and y_array.

    :param in_array : float values to aggregate.
    :param x_array : column index in in_array, containing the column index of the destination cell
    :param y_array : row index in in_array, containing the row index of the destination cell
    :param out_x_array : the receiving array containing all x mean values per cell
    :param out_y_array : the receiving array containing all y mean values per cell
    :param out_last_array : the receiving array containing all last projected values per cell
    :param out_min_array : the receiving array containing all min values per cell
    :param out_mean_array : the receiving array containing all mean values per cell
    :param out_max_array : the receiving array containing all max values per cell
    :param out_count_array : the receiving array containing the number of values per cell, or the sum of weights if in_weights is provided
    :param out_filtered_array : the receiving array containing the number of values equals to nan
    :param out_sum_array : the receiving array containing all sum values per cell (weighted if in_weights is provided)
    """
    # Check out array
    shape = (0, 0)
    if out_last_array is not None:
        shape = out_last_array.shape
    if out_min_array is not None:
        shape = out_min_array.shape
    if out_mean_array is not None:
        shape = out_mean_array.shape
    if out_max_array is not None:
        shape = out_max_array.shape
    if out_count_array is not None:
        shape = out_count_array.shape
    if out_sum_array is not None:
        shape = out_sum_array.shape
    if out_filtered_array is not None:
        shape = out_filtered_array.shape
    if shape == (0, 0):
        raise ValueError("At least one out array must be specified")
    if out_mean_array is not None and out_count_array is None:
        raise ValueError("out_count_array is mandatory to compute the mean values")
    for value, x, y in zip(in_array.flat, x_array.flat, y_array.flat):
        col = math.floor(x)
        row = math.floor(y)
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            if not np.isnan(value):
                if out_last_array is not None:
                    out_last_array[row, col] = value
                if out_mean_array is not None and out_count_array is not None:
                    if np.isnan(out_mean_array[row, col]):
                        out_mean_array[row, col] = value
                    else:
                        out_mean_array[row, col] = (
                            (np.float64(out_mean_array[row, col]) * out_count_array[row, col]) + np.float64(value)
                        ) / (out_count_array[row, col] + 1)
                if out_x_array is not None and out_count_array is not None:
                    if np.isnan(out_x_array[row, col]):
                        out_x_array[row, col] = x
                    else:
                        out_x_array[row, col] = (
                            (np.float64(out_x_array[row, col]) * out_count_array[row, col]) + np.float64(x)
                        ) / (out_count_array[row, col] + 1)
                if out_y_array is not None and out_count_array is not None:
                    if np.isnan(out_y_array[row, col]):
                        out_y_array[row, col] = y
                    else:
                        out_y_array[row, col] = (
                            (np.float64(out_y_array[row, col]) * out_count_array[row, col]) + np.float64(y)
                        ) / (out_count_array[row, col] + 1)
                if out_count_array is not None:
                    if out_count_array[row, col] <= 0:
                        out_count_array[row, col] = 1
                    else:
                        out_count_array[row, col] = out_count_array[row, col] + 1
                if out_min_array is not None:
                    if np.isnan(out_min_array[row, col]):
                        out_min_array[row, col] = value
                    else:
                        out_min_array[row, col] = min(out_min_array[row, col], value)
                if out_max_array is not None:
                    if np.isnan(out_max_array[row, col]):
                        out_max_array[row, col] = value
                    else:
                        out_max_array[row, col] = max(out_max_array[row, col], value)
                if out_sum_array is not None:
                    if np.isnan(out_sum_array[row, col]):
                        out_sum_array[row, col] = value
                    else:
                        out_sum_array[row, col] = out_sum_array[row, col] + value
            elif out_filtered_array is not None:
                if out_filtered_array[row, col] <= 0:
                    out_filtered_array[row, col] = 1
                else:
                    out_filtered_array[row, col] = out_filtered_array[row, col] + 1


@numba.njit(cache=True, fastmath=False)
def compute_standard_deviation_first_pass(
    in_values: np.ndarray,
    x_array: np.ndarray,
    y_array: np.ndarray,
    out_tmp_array: np.ndarray,
):
    """
    Project the values of the in_values to the out_stddev_array and prepare the computing of the standard deviation for each cell
    Indexes (column and row) used to determine the cell in out_stddev_array are found in x_array and y_array

    :param in_values : float values to project.
    :param x_array : column index in in_array, containing the column index of the destination cell
    :param y_array : row index in in_array, containing the row index of the destination cell
    :param out_tmp_array : the receiving array
    """
    out_shape = out_tmp_array.shape

    for in_value, col, row in zip(in_values.flat, x_array.flat, y_array.flat):
        if not np.isnan(in_value):
            if 0 <= row < out_shape[0] and 0 <= col < out_shape[1]:
                if np.isnan(out_tmp_array[row, col]):
                    out_tmp_array[row, col] = np.float64(in_value) ** 2
                else:
                    out_tmp_array[row, col] += np.float64(in_value) ** 2


def compute_standard_deviation_second_pass(
    in_count_array: np.ndarray,
    in_mean_array: np.ndarray,
    in_square_array: np.ndarray,
    out_stddev_array: np.ndarray,
):
    """
    Finalize the computing of the standard deviation for each cell
    Indexes (column and row) used to determine the cell in out_stddev_array are found in x_array and y_array

    :param in_count_array : number of values
    :param in_mean_array : mean values
    :param in_square_array : sum of squared values
    :param out_stddev_array : the receiving array containing all sum of squared values
    """
    out_stddev_array[in_count_array <= 0] = np.nan
    squared_mean = np.square(in_mean_array)  # mean²
    np.divide(in_square_array, in_count_array, out=in_square_array)  # sum(value²)/n
    np.subtract(in_square_array, squared_mean, out=in_square_array)  # sum(value²)/n - mean²
    np.sqrt(in_square_array, out=out_stddev_array)  # sqrt(sum(value²)/n - mean²)


# pylint: disable=too-many-boolean-expressions, consider-using-with
# In fastmath mode, np.isnan does not work !
@numba.njit(cache=True, fastmath=False)
def project_into_grid_keep_last(
    in_array: np.ndarray,
    x_array: np.ndarray,
    y_array: np.ndarray,
    out_array: np.ndarray,
    missing_value: Union[float, int, None] = None,
    factor: float = 1.0,
) -> None:
    """
    Project the values contained in in_array to the out_array.
    Only the last value is kept in the cell
    Indexes (column and row) used to determine the cell in out_array are found in x_array and y_array

    :param in_array : float values to aggregate.
    :param x_array : column index in in_array, containing the column index of the destination cell
    :param y_array : row index in in_array, containing the row index of the destination cell
    :param out_array : the receiving array containing the last value per cell
    :param factor : factor applied to values
    """
    out_shape = out_array.shape
    for in_value, col, row in zip(in_array.flat, x_array.flat, y_array.flat):
        # Sanity check
        if not (0 <= row < out_shape[0] and 0 <= col < out_shape[1]):
            continue

        # Really a value ?
        if np.isnan(in_value):
            continue

        # Any missing value to check ?
        if missing_value is None or np.isnan(missing_value):
            out_array[row, col] = in_value * factor
            continue

        # Check if missing value
        value = in_value * factor
        if value != missing_value:
            out_array[row, col] = value


def to_memmap(in_array: np.ndarray) -> np.ndarray:
    """
    Utility method to create and initialize memory-map to an array stored in a binary file on disk .
    """
    map_file = tempfile.TemporaryFile(suffix=".memmap")
    result = np.memmap(map_file, shape=in_array.shape, dtype=in_array.dtype, mode="w+")
    # Write data to memmap array
    result[:] = in_array[:]

    return result


def to_utc(in_datetime: datetime.datetime) -> np.datetime64:
    """Convert a naive or aware datetime to np datetime64 in UTC timezone
    @:param in_datetime: the naive/aware datetime to convert
    @:return datetime64 object
    """
    return np.datetime64(datetime.datetime.utcfromtimestamp(in_datetime.timestamp()))


def project_coords(
    xs: np.ndarray,
    ys: np.ndarray,
    geobox: arg_util.Geobox,
    spatial_resolution: float,
    spatial_reference: osr.SpatialReference = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project all coordinates to the grid to obtain the cell position.
    xs are the longitudes when spatial_reference is not projected (LonLat). Otherwise, xs are abscissas
    xy are the latitudes when spatial_reference is not projected (LonLat). Otherwise, xs are ordinates
    geobox represents the coordinates of the centers of the cells in the corners
    spatial_reference is the SRS of the xs and ys. None means that the CRS is the same than GeoBox and no transformation is required
    returns (columns, rows) calculated by the projection as float
    """
    # Compare xy and geobox CRS. If not same, transformation is required
    if spatial_reference is not None:
        xy_crs = crs.CRS.from_proj4(spatial_reference.ExportToProj4())
        geobox_crs = crs.CRS.from_proj4(geobox.spatial_reference.ExportToProj4())
        if not xy_crs.is_exact_same(geobox_crs):
            # Transform input coordinates to the target CRS
            transformer = Transformer.from_crs(
                crs.CRS.from_epsg(4326),
                crs.CRS.from_proj4(geobox.spatial_reference.ExportToProj4()),
                always_xy=True,
            )
            # make a tuple to avoid a false positive with pylint unpacking-non-sequence
            xs, ys = tuple(transformer.transform(xs, ys, radians=False))

    # Project coords of target CRS to the cell position in the DTM grid
    columns = (
        __project_longitude_to_grid(xs, geobox.left, geobox.right, spatial_resolution)
        if geobox.spatial_reference.IsGeographic()
        else __project_value_to_axis(xs, geobox.left, spatial_resolution)
    )
    rows = __project_value_to_axis(ys, geobox.lower, spatial_resolution)

    return (columns, rows)


def project_coords_as_index(
    xs: np.ndarray,
    ys: np.ndarray,
    geobox: arg_util.Geobox,
    spatial_resolution: float,
    spatial_reference: osr.SpatialReference = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project all coordinates to the grid to obtain the cell position.
    xs are the longitudes when spatial_reference is not projected (LonLat). Otherwise, xs are abscissas
    xy are the latitudes when spatial_reference is not projected (LonLat). Otherwise, xs are ordinates
    geobox represents the coordinates of the centers of the cells in the corners
    spatial_reference is the SRS of the xs and ys. None means that the CRS is the same than GeoBox and no transformation is required
    returns (columns, rows) calculated by the projection as int
    """
    x, y = project_coords(xs, ys, geobox, spatial_resolution, spatial_reference)
    return np.floor(x).astype(int), np.floor(y).astype(int)


@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)])
def __project_value_to_axis(value: np.ndarray, axis_origin: float, cell_size: float) -> np.ndarray:
    return (value - axis_origin) / cell_size


@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64, numba.float64)])
def __project_longitude_to_grid(long: np.ndarray, west: float, east: float, spatial_resolution: float) -> np.ndarray:
    # Check if longitude span the 180th meridian
    if west > 0 > east and long < west:
        long += 360.0
    return (long - west) / spatial_resolution
