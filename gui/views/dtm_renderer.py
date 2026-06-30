"""Utility: render DTM layers to RGBA PNG images for map display.

NaN cells are made transparent so the map background shows through.
Large grids are block-averaged so the longest axis stays at
``_TARGET_PX`` pixels — much sharper than step-decimation.
Elevation layers can be rendered with hillshading for a 3-D effect.
"""
import base64
import io
import os
import tempfile
import uuid
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TARGET_PX = 16000


def _decimate(data: np.ndarray, max_px: int) -> np.ndarray:
    """Block-average decimation — smoother than step-sampling, avoids aliasing.

    NaN cells are excluded from each block average so NaN regions stay NaN.
    """
    ny, nx = data.shape
    longest = max(nx, ny)
    if longest <= max_px:
        return data
    step = int(np.ceil(longest / max_px))
    ny_red = ny // step
    nx_red = nx // step
    trimmed = data[:ny_red * step, :nx_red * step]
    blocks = trimmed.reshape(ny_red, step, nx_red, step)
    return np.nanmean(blocks, axis=(1, 3))


def _shoulder_compression(x: np.ndarray, strength: float, threshold: float = 0.7) -> np.ndarray:
    """Piecewise power-function shoulder compression for suppressing bright artifacts.

    Only affects the upper ``threshold`` fraction of the normalised [0, 1] range,
    leaving mid-tones and shadows untouched — ideal for darkening near-nadir
    bright trails in backscatter mosaics.

    At full strength the output ceiling drops from 1.0 → 0.85 so that even the
    very brightest pixels (which saturate at 1.0 after percentile-based
    normalisation) get pulled down, eliminating residual bright spots.

    Args:
        x: Normalized data in [0, 1] (NaN-safe, NaNs pass through unchanged).
        strength: Compression strength in [0, 1]; 0 = identity, 1 = maximum.
        threshold: Normalised value above which compression starts (default 0.7).

    Returns:
        Compressed array in [0, 1], same shape as ``x``.
    """
    if strength <= 0.0:
        return np.clip(x, 0.0, 1.0)

    # Power exponent: 1 (off) → 5 (smoother roll-off at max strength)
    # Lower exponent avoids the harsh transition near 1.0 while maintaining smooth compression.
    p = 1.0 + 4.0 * strength

    # Reduce the output ceiling so the very brightest pixels also get dimmed.
    # At max strength the ceiling drops from 1.0 → 0.72 (close to threshold)
    # to completely suppress residual highlights.
    max_out = 1.0 - 0.28 * strength

    # Clip input BEFORE the mapping to keep NaN/Inf safe.
    xc = np.clip(x, 0.0, 1.0)
    above = xc > threshold
    if not above.any():
        return xc

    result = xc.copy()
    # Remap [threshold, 1] → [0, 1], apply power, remap back
    u = (result[above] - threshold) / (1.0 - threshold)
    v = u ** p
    result[above] = threshold + v * (max_out - threshold)
    # Final safety clip — values above max_out are pulled down.
    np.clip(result, 0.0, 1.0, out=result)
    return result


def _hillshade(data: np.ndarray, azimuth: float = 315, altitude: float = 35) -> np.ndarray:
    """Compute shaded relief from elevation data.

    Args:
        data: 2-D elevation array (arbitrary units, but consistent scale).
        azimuth: Light azimuth in degrees (0=North, clockwise).
        altitude: Light altitude in degrees above horizon.

    Returns:
        2-D array in [0, 1] where 1 = fully lit, NaN preserved.
    """
    filled = np.where(np.isfinite(data), data, 0.0)
    dx, dy = np.gradient(filled)
    azi_rad = np.deg2rad(azimuth)
    alt_rad = np.deg2rad(altitude)
    shaded = np.sin(alt_rad) - np.cos(alt_rad) * (dx * np.cos(azi_rad) + dy * np.sin(azi_rad))
    shaded = np.clip(shaded, 0.0, 1.0)
    shaded[~np.isfinite(data)] = np.nan
    return shaded


def _make_rgba(data: np.ndarray, cmap_name: str,
               vmin: float, vmax: float,
               gamma: float = 1.0,
               shoulder: float = 0.0,
               hillshade: bool = False) -> np.ndarray:
    """Convert a 2-D float array to an RGBA uint8 image.
    NaN cells → alpha=0 (transparent).
    ``gamma`` is applied as V_out = V_in^(1/gamma).
    ``shoulder`` controls S-curve shoulder compression applied *before* gamma
    to selectively roll off bright values without affecting mid-tones/shadows.
    When ``hillshade=True`` the RGB channels are multiplied by a shaded-relief
    layer computed from the data for a realistic 3-D terrain effect.
    """
    normed = (data.astype(np.float64) - vmin) / (vmax - vmin)
    normed = np.clip(normed, 0.0, 1.0)
    # S-curve shoulder compression — targets upper range only
    normed = _shoulder_compression(normed, shoulder)
    if gamma != 1.0:
        normed = normed ** (1.0 / gamma)
    rgba = plt.colormaps[cmap_name](normed)

    if hillshade:
        shade = _hillshade(data)
        rgba[:, :, 0] *= shade
        rgba[:, :, 1] *= shade
        rgba[:, :, 2] *= shade

    rgba[np.isnan(data), 3] = 0.0
    return (rgba * 255).astype(np.uint8)


def _read_dtm_layer(dtm_path: str, layer_name: str,
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None):
    """Read a DTM layer, return (lon, lat, data, bounds, vmin, vmax).

    Supports both geographic (lon/lat) and projected (x/y) DTMs.
    For projected DTMs, ``lon``/``lat`` arrays are the projected
    coordinates (for internal rendering), and ``bounds`` is the
    geographic bounds in (min_lon, min_lat, max_lon, max_lat)
    transformed via the DTM's CRS to WGS84.
    """
    from netCDF4 import Dataset
    import numpy as np
    with Dataset(dtm_path, "r") as ds:
        has_geo = "lon" in ds.variables and "lat" in ds.variables
        has_proj = "x" in ds.variables and "y" in ds.variables
        data = ds.variables[layer_name][:]

        if has_geo:
            lon = ds.variables["lon"][:].astype(float)
            lat = ds.variables["lat"][:].astype(float)
            min_lon, max_lon = float(lon.min()), float(lon.max())
            min_lat, max_lat = float(lat.min()), float(lat.max())
        elif has_proj:
            x_arr = ds.variables["x"][:].astype(float)
            y_arr = ds.variables["y"][:].astype(float)
            lon, lat = x_arr, y_arr  # use x/y as rendering axes
            # Transform bounds to WGS84 lat/lon for the Leaflet overlay
            try:
                crs_wkt = getattr(ds.variables.get("crs"), "crs_wkt", None)
                if crs_wkt:
                    from pyproj import CRS, Transformer
                    src_crs = CRS.from_wkt(crs_wkt)
                    tgt_crs = CRS.from_epsg(4326)
                    trans = Transformer.from_crs(src_crs, tgt_crs, always_xy=True)
                    sw = trans.transform(float(x_arr[0]), float(y_arr[0]))
                    ne = trans.transform(float(x_arr[-1]), float(y_arr[-1]))
                    min_lon = min(sw[0], ne[0])
                    max_lon = max(sw[0], ne[0])
                    min_lat = min(sw[1], ne[1])
                    max_lat = max(sw[1], ne[1])
                else:
                    min_lon = min(float(x_arr[0]), float(x_arr[-1]))
                    max_lon = max(float(x_arr[0]), float(x_arr[-1]))
                    min_lat = min(float(y_arr[0]), float(y_arr[-1]))
                    max_lat = max(float(y_arr[0]), float(y_arr[-1]))
            except Exception:
                min_lon = min(float(x_arr[0]), float(x_arr[-1]))
                max_lon = max(float(x_arr[0]), float(x_arr[-1]))
                min_lat = min(float(y_arr[0]), float(y_arr[-1]))
                max_lat = max(float(y_arr[0]), float(y_arr[-1]))
        else:
            raise IOError("DTM file has neither lon/lat nor x/y variables")

    if hasattr(data, 'mask') and isinstance(data, np.ma.MaskedArray):
        data = np.ma.filled(data.astype(float), np.nan)
    else:
        data = data.astype(float)
    data[~np.isfinite(data)] = np.nan

    if data.shape[0] == len(lon) and data.shape[1] == len(lat):
        data = data.T
    elif not (data.shape[0] == len(lat) and data.shape[1] == len(lon)):
        pass

    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanpercentile(valid, 2)) if vmin is None else vmin
        vmax = float(np.nanpercentile(valid, 98)) if vmax is None else vmax
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0

    return lon, lat, data, (min_lon, min_lat, max_lon, max_lat), vmin, vmax


def _make_png(data: np.ndarray, cmap_name: str,
              vmin: float, vmax: float,
              gamma: float = 1.0,
              shoulder: float = 0.0,
              hillshade: bool = False) -> bytes:
    data = _decimate(data, _TARGET_PX)

    # Flip vertically: PNG row 0 = top of image = north
    # (netCDF stores row 0 = south by CF convention)
    data = np.flipud(data)

    rgba = _make_rgba(data, cmap_name, vmin, vmax, gamma, shoulder, hillshade)
    buf = io.BytesIO()
    try:
        from PIL import Image
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=1)
    except ImportError:
        fig, ax = plt.subplots(figsize=(rgba.shape[1] / 100, rgba.shape[0] / 100), dpi=100)
        ax.imshow(rgba, interpolation="nearest", aspect="auto")
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        fig.savefig(buf, format="png", dpi=100, transparent=True)
        plt.close(fig)
    return buf.getvalue()


def dtm_layer_to_data_url(
    dtm_path: str, layer_name: str = "backscatter",
    cmap: str = "gray",
    vmin: Optional[float] = None, vmax: Optional[float] = None,
    gamma: float = 1.0,
    shoulder: float = 0.0,
    hillshade: bool = False,
) -> Tuple[str, Tuple[float, float, float, float]]:
    """Read a DTM layer and return as (data_url, bounds)."""
    lon, lat, data, bounds, vmin, vmax = _read_dtm_layer(dtm_path, layer_name, vmin, vmax)
    png = _make_png(data, cmap, vmin, vmax, gamma, shoulder, hillshade)
    b64 = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}", bounds


def dtm_layer_to_file(
    dtm_path: str, layer_name: str = "backscatter",
    cmap: str = "gray",
    vmin: Optional[float] = None, vmax: Optional[float] = None,
    gamma: float = 1.0,
    shoulder: float = 0.0,
    hillshade: bool = False,
) -> Tuple[str, str, Tuple[float, float, float, float]]:
    """Read a DTM layer, save PNG file, return (out_path, data_url, bounds)."""
    lon, lat, data, bounds, vmin, vmax = _read_dtm_layer(dtm_path, layer_name, vmin, vmax)
    png = _make_png(data, cmap, vmin, vmax, gamma, shoulder, hillshade)
    out = os.path.join(tempfile.gettempdir(), f"pyat_{layer_name}_{uuid.uuid4().hex[:8]}.png")
    with open(out, "wb") as f:
        f.write(png)
    b64 = base64.b64encode(png).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"
    return out, data_url, bounds
