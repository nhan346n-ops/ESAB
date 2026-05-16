"""XSF metadata reader using netCDF4."""
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np

from ..utils.config import ATT_PROCESSING_STATUS_BACKSCATTER_CORRECTION


@dataclass
class XsfMetadata:
    """Metadata extracted from an XSF (SONAR-netCDF4) file."""
    filepath: str
    filename: str
    filesize_mb: float
    title: str = ""
    sounder_type: str = ""
    sonar_convention_version: str = ""
    xsf_version: str = ""
    date_created: str = ""
    backscatter_correction: int = 0  # 0=unprocessed, 1=processed
    processing_status_raw: str = ""
    nav_bounds: Optional[Dict[str, float]] = None
    extra_attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_processed(self) -> bool:
        return self.backscatter_correction == 1


def read_xsf_metadata(filepath: str) -> XsfMetadata:
    """Read metadata from an XSF file using netCDF4.

    Reads only global attributes — does NOT load variable data.
    Safe for very large files (500MB+).
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ImportError("netCDF4 is required to read XSF files")

    filename = os.path.basename(filepath)
    filesize_mb = os.path.getsize(filepath) / (1024 * 1024)

    metadata = XsfMetadata(
        filepath=filepath,
        filename=filename,
        filesize_mb=round(filesize_mb, 1),
    )

    try:
        with Dataset(filepath, "r") as ds:
            # Global attributes
            metadata.title = _get_attr(ds, "title")
            metadata.sounder_type = _get_attr(ds, "keywords", "").replace("()", "").strip()
            metadata.sonar_convention_version = _get_attr(ds, "sonar_convention_version")
            metadata.xsf_version = _get_attr(ds, "xsf_convention_version")
            metadata.date_created = _get_attr(ds, "date_created")
            metadata.processing_status_raw = _get_attr(ds, "processing_status", "{}")

            # Parse processing_status JSON
            try:
                status = json.loads(metadata.processing_status_raw)
                metadata.backscatter_correction = status.get(
                    ATT_PROCESSING_STATUS_BACKSCATTER_CORRECTION, 0
                )
            except (json.JSONDecodeError, TypeError):
                metadata.backscatter_correction = 0

            # Collect extra global attrs
            for attr_name in ds.ncattrs():
                if attr_name not in (
                    "title", "keywords", "sonar_convention_version",
                    "xsf_convention_version", "date_created", "processing_status"
                ):
                    metadata.extra_attrs[attr_name] = _get_attr(ds, attr_name)

            # Try to extract navigation bounds from /Sonar group if available
            _try_read_nav_bounds(ds, metadata)

    except Exception as e:
        metadata.extra_attrs["_read_error"] = str(e)

    return metadata


def _get_attr(ds, attr_name: str, default: str = "") -> str:
    """Safely get a netCDF global attribute."""
    try:
        val = getattr(ds, attr_name, default)
        return str(val) if val is not None else default
    except Exception:
        return default


def _try_read_nav_bounds(ds, metadata: XsfMetadata) -> None:
    """Attempt to read navigation bounds from the XSF file."""
    try:
        if "Sonar" not in ds.groups:
            return
        sonar = ds.groups["Sonar"]
        nav_group = None
        for gname in sonar.groups:
            if gname.lower().startswith("beam_group"):
                bg = sonar.groups[gname]
                if "Navigation" in bg.groups:
                    nav_group = bg.groups["Navigation"]
                    break

        if nav_group is None:
            return

        # Read longitude/latitude arrays
        lon_var = None
        lat_var = None
        for vname in nav_group.variables:
            vlow = vname.lower()
            if "longitude" in vlow and lon_var is None:
                lon_var = nav_group.variables[vname]
            if "latitude" in vlow and lat_var is None:
                lat_var = nav_group.variables[vname]

        if lon_var is not None and lat_var is not None:
            lon_data = lon_var[:].ravel()
            lat_data = lat_var[:].ravel()
            # Remove fill values
            mask = np.isfinite(lon_data) & np.isfinite(lat_data)
            if mask.any():
                metadata.nav_bounds = {
                    "lon_min": float(np.min(lon_data[mask])),
                    "lon_max": float(np.max(lon_data[mask])),
                    "lat_min": float(np.min(lat_data[mask])),
                    "lat_max": float(np.max(lat_data[mask])),
                }
    except Exception:
        pass  # Navigation bounds are optional


def scan_directory(directory: str) -> List[XsfMetadata]:
    """Scan a directory for XSF files and read their metadata."""
    results = []
    if not os.path.isdir(directory):
        return results

    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".nc"):
            filepath = os.path.join(directory, fname)
            try:
                meta = read_xsf_metadata(filepath)
                results.append(meta)
            except Exception:
                continue

    return results
