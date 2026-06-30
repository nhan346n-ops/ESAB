"""
GXF-3 exporter for NetCDF4 DTM grids.
"""

from typing import Optional

import numpy as np
from osgeo import osr
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.utils.process_utils as process_util
import pyat.utils.argument_utils as arg_util
import pyat.utils.pyat_logger as log
from pyat.dtm.dtm_standard_constants import ELEVATION_NAME


def _get_format(var):
    """
    Returns a format string for the given variable type, used for writing header fields in GXF export.
    """
    if isinstance(var, str):
        return "%s"
    if isinstance(var, (np.integer, int)):
        return "%d"
    if isinstance(var, (np.floating, float)):
        # prefer integer format when value is integral
        if np.all(np.equal(np.mod(var, 1), 0)):
            return "%d"
        return "%.6f"
    # fallback
    return "%.6f"


def _next_pow10(n: float) -> int:
    """
    Returns the exponent of the next power of 10 greater than n.
    """
    return int(np.ceil(np.log10(np.abs(n))))


def _get_dummy(zmin: float, zmax: float) -> int:
    """
    Returns dummy value for NaN replacement in GXF export, ensuring it does not overlap with actual data.
    """
    if zmin < 0 < zmax:
        dummy = 10 ** _next_pow10(zmax) - 1
        if dummy <= zmax:
            dummy = 10 ** (_next_pow10(zmax) + 1) - 1
    else:
        dummy = 0
    return int(dummy)


def dtm_crs_to_map_projection(dtm_crs: osr.SpatialReference) -> str:
    """
    Convert an `osr.SpatialReference` into a GXF #MAP_PROJECTION block body.
    Returns the formatted string (without the leading `#MAP_PROJECTION` line)
    """
    # Datum / proj name
    try:
        datum = dtm_crs.GetAttrValue("DATUM") or dtm_crs.GetAttrValue("GEOGCS") or ""
    except Exception:
        datum = ""

    # Semi-major axis
    try:
        semi_major = float(dtm_crs.GetSemiMajor())
    except Exception:
        semi_major = 0.0

    # Eccentricity computed from inverse flattening when available
    try:
        inv_f = float(dtm_crs.GetInvFlattening())
        if not np.isclose(inv_f, 0.0, rtol=1e-09, atol=1e-09):
            f = 1.0 / inv_f
            eccentricity = float(np.sqrt(2.0 * f - f * f))
        else:
            eccentricity = 0.0
    except Exception:
        eccentricity = 0.0

    # Prime meridian (fallback to 0)
    # PRIMEM attribute stores [name, longitude, ...]
    try:
        pm_str = dtm_crs.GetAttrValue("PRIMEM", 1)
        prime_meridian = float(pm_str) if pm_str else 0.0
    except Exception:
        prime_meridian = 0.0

    # Projection method and parameters
    proj_method = "Geographic"
    proj_parameters = ""
    proj_name = datum if datum else (dtm_crs.GetAttrValue("PROJCS") or "")

    try:
        # Non projected CRS
        if dtm_crs.IsGeographic():
            proj_method = "Geographic"
            proj_parameters = ""
            proj_name = datum or proj_name
        else:
            # Projected CRS : try to get projection name from the SRS
            try:
                proj_attr = dtm_crs.GetAttrValue("PROJECTION") or ""
            except Exception:
                proj_attr = ""

            proj_attr_low = (proj_attr or "").lower()

            # Mercator
            if "mercator" in proj_attr_low:
                proj_method = "Mercator (2SP)"
                lat_ts = dtm_crs.GetProjParm("standard_parallel_1") or dtm_crs.GetProjParm("latitude_of_origin") or 0.0
                lon0 = dtm_crs.GetProjParm("central_meridian") or dtm_crs.GetProjParm("longitude_of_origin") or 0.0
                x0 = dtm_crs.GetProjParm("false_easting") or 0.0
                y0 = dtm_crs.GetProjParm("false_northing") or 0.0
                proj_parameters = f",{lat_ts},{lon0},{x0},{y0}"
                proj_name = proj_name or f"{datum} / World Mercator"

            # Transverse Mercator / UTM-like
            elif "transverse" in proj_attr_low or "utm" in proj_attr_low:
                proj_method = "Transverse Mercator"
                lat0 = dtm_crs.GetProjParm("latitude_of_origin") or 0.0
                lon0 = dtm_crs.GetProjParm("central_meridian") or dtm_crs.GetProjParm("longitude_of_origin") or 0.0
                scale = dtm_crs.GetProjParm("scale_factor") or 1.0
                x0 = dtm_crs.GetProjParm("false_easting") or 0.0
                y0 = dtm_crs.GetProjParm("false_northing") or 0.0
                proj_parameters = f",{lat0},{lon0},{scale},{x0},{y0}"
                proj_name = proj_name or dtm_crs.GetAttrValue("PROJCS") or proj_name

            # Lambert Conic Conformal (2SP)
            elif "lambert" in proj_attr_low:
                proj_method = "Lambert Conic Conformal (2SP)"
                sp1 = dtm_crs.GetProjParm("standard_parallel_1") or 0.0
                sp2 = dtm_crs.GetProjParm("standard_parallel_2") or 0.0
                lat0 = dtm_crs.GetProjParm("latitude_of_origin") or 0.0
                lon0 = dtm_crs.GetProjParm("central_meridian") or 0.0
                x0 = dtm_crs.GetProjParm("false_easting") or 0.0
                y0 = dtm_crs.GetProjParm("false_northing") or 0.0
                proj_parameters = f",{sp1},{sp2},{lat0},{lon0},{x0},{y0}"
                proj_name = proj_name or dtm_crs.GetAttrValue("PROJCS") or proj_name

            # Polar stereographic
            elif "stere" in proj_attr_low or "polar" in proj_attr_low:
                proj_method = "Polar Stereographic"
                lat0 = dtm_crs.GetProjParm("latitude_of_origin") or 0.0
                lon0 = dtm_crs.GetProjParm("central_meridian") or 0.0
                scale = dtm_crs.GetProjParm("scale_factor") or 1.0
                x0 = dtm_crs.GetProjParm("false_easting") or 0.0
                y0 = dtm_crs.GetProjParm("false_northing") or 0.0
                proj_parameters = f",{lat0},{lon0},{scale},{x0},{y0}"
                proj_name = proj_name or dtm_crs.GetAttrValue("PROJCS") or proj_name

            else:
                raise NotImplementedError(f"Projection method not recognized for GXF export: {proj_attr}")
    except Exception as exc:
        raise NotImplementedError(f"Projection method not recognized for GXF export: {proj_attr}") from exc

    # Format according to Matlab implementation: three lines following '#MAP_PROJECTION'
    return f'"{proj_name}"\n"{datum}",{semi_major},{eccentricity},{prime_meridian}\n"{proj_method}"{proj_parameters}\n'


def export_gxf(
    path: str,
    y_axis: np.ndarray,
    x_axis: np.ndarray,
    elevation: np.ndarray,
    map_projection: str = None,
    file_type: str = "kingdom",
    precision: float = 0.001,
    monitor: ProgressMonitor = DefaultMonitor,
) -> None:
    """
    Export DTM grid to GXF-3 format.
    """
    file_type = file_type.lower()
    if file_type not in ["uncompressed", "kingdom"]:
        raise NotImplementedError("Only uncompressed or kingdom GXF is implemented")

    # grid dimensions and spacing
    points = int(len(x_axis))
    rows = int(len(y_axis))
    rw_sep = float(abs(y_axis[1] - y_axis[0])) if len(y_axis) > 1 else 0.0
    pt_sep = float(abs(x_axis[1] - x_axis[0])) if len(x_axis) > 1 else 0.0

    # Lower-left origin
    xorigin = float(np.min(x_axis))
    yorigin = float(np.min(y_axis))

    # set sense according to file type, flipping Z if kingdom
    if file_type == "kingdom":
        sense = 1
    else:
        sense = -2

    # Z min/max ignoring NaN
    zmax = float(np.nanmax(elevation))
    zmin = float(np.nanmin(elevation))

    # nodata value handling: replace NaN with a dummy value outside the range of actual data
    dummy = _get_dummy(zmin, zmax)
    elevation = np.where(np.isnan(elevation), dummy, elevation)

    # Header fields
    header = [
        ("TITLE", '"Grid exported by pyat"'),
        ("POINTS", points),
        ("ROWS", rows),
        ("RWSEPARATION", rw_sep),
        ("PTSEPARATION", pt_sep),
        ("SENSE", sense),
        ("XORIGIN", xorigin),
        ("YORIGIN", yorigin),
        ("ZMAXIMUM", zmax),
        ("ZMINIMUM", zmin),
        ("DUMMY", dummy),
    ]

    fmt_cache = {}

    def _fmt(v):
        t = type(v)
        if t not in fmt_cache:
            fmt_cache[t] = _get_format(v)
        return (fmt_cache[t], v)

    monitor.begin_task("Exporting to GXF", 100)

    # write the GXF file
    with open(path, "wt", encoding="utf-8") as gxf:
        # write header fields
        for name, val in header:
            gxf.write(f"#{name}\n")
            fmt, v = _fmt(val)
            gxf.write((fmt % v) + "\n")

        # write projection description
        if map_projection:
            # map_projection is expected to contain the three lines following the
            # '#MAP_PROJECTION' header (already formatted). Write the header
            # then the provided block.
            gxf.write("#MAP_PROJECTION\n")
            gxf.write(map_projection)

        # Write GRID, packing up to 80 characters per line
        gxf.write("#GRID\n")
        for i in range(rows):
            if i % max(1, rows // 10) == 0:
                monitor.worked(10)
            j = 0
            while j < points:
                txt_parts = []
                nchar = 0
                while j < points:
                    word = (
                        f"{elevation[i, j]:.{int(-np.log10(precision))}f}"
                        if np.issubdtype(type(elevation[i, j]), np.floating)
                        else f"{int(elevation[i, j])}"
                    )
                    # strip trailing zeros and decimal points for shorter files
                    word = word.rstrip("0").rstrip(".") if "." in word else word
                    part_len = len(word) + 1
                    if nchar + part_len > 80 and len(txt_parts) > 0:
                        break
                    txt_parts.append(word)
                    nchar += part_len
                    j += 1
                # write line with space-separated values
                gxf.write(" ".join(txt_parts).rstrip() + "\n")
    monitor.done()


class Dtm2GXF:
    """Wrapper to export one or more DTM files to GXF format.

    Usage mirrors `Dtm2Ascii` in `dtm_to_ascii.py`.
    """

    def __init__(
        self,
        i_paths: list,
        o_paths: Optional[list] = None,
        overwrite: bool = False,
        file_type: str = "kingdom",
        precision: float = 0.001,
        monitor=DefaultMonitor,
    ):
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.overwrite = overwrite
        self.file_type = file_type
        self.precision = precision
        self.monitor = monitor
        self.logger = log.logging.getLogger(self.__class__.__name__)

    def __process_data(self, i_dtm_driver: dtm_driver.DtmDriver, monitor: ProgressMonitor) -> None:
        ind = self.i_paths.index(i_dtm_driver.dtm_file.file_path)
        o_path = arg_util.create_output_path(
            i_dtm_driver.dtm_file.file_path,
            extension=".gxf",
            overwrite=self.overwrite,
            o_path=(None if not self.o_paths else self.o_paths[ind]),
        )

        self.logger.info(f"Creating file {o_path}")
        x_variable = i_dtm_driver.get_x_axis()
        x_variable.set_auto_mask(False)
        x_axis = x_variable[:]
        y_variable = i_dtm_driver.get_y_axis()
        y_variable.set_auto_mask(False)
        y_axis = y_variable[:]

        elevation_variable = None
        try:
            elevation_variable = i_dtm_driver[ELEVATION_NAME]
            elevation_variable.set_auto_mask(False)
        except Exception as exc:
            raise RuntimeError("No elevation variable found in DTM") from exc
        elevation = elevation_variable[:]

        # get CRS from DTM file
        dtm_crs = i_dtm_driver.dtm_file.spatial_reference

        export_gxf(
            o_path,
            y_axis,
            x_axis,
            elevation,
            map_projection=dtm_crs_to_map_projection(dtm_crs),
            file_type=self.file_type,
            precision=self.precision,
            monitor=monitor.split(1),
        )

    def __call__(self) -> None:
        """
        Process each input DTM file and export to GXF.
        """
        self.monitor.begin_task("Exporting DTM to GXF", len(self.i_paths))
        process_util.process_each_input_file_in_read_mode(
            self.i_paths,
            self.__class__.__name__,
            self.logger,
            self.monitor,
            self.__process_data,
        )
