"""JSON configuration builder for pyat backend tools.

Generates JSON parameter files matching the format expected by:
  python -m pyat <config.json>
"""
import json, os
from typing import Dict, Any, List, Optional

from ..utils.config import (
    SRC_PATH, JOBS_DIR, GWS_CONF_PATH, get_timestamp, ensure_dirs,
)


def build_args_json(
    config_file_relpath: str,
    params: Dict[str, Any],
    tool_name: str,
) -> str:
    """Build an arguments JSON file for pyat execution.

    Args:
        config_file_relpath: Relative path from GWS_CONF_DIR to config JSON
            (e.g., "sonar/bs/bs_sliding_angular_renormalization.json")
        params: Dictionary of parameter values.
        tool_name: Tool name for file naming.

    Returns:
        Absolute path to the generated JSON file.
    """
    ensure_dirs()

    # Resolve to absolute path for pyat to find it regardless of CWD
    abs_config_path = str(SRC_PATH / "gws" / "conf" / config_file_relpath)

    arguments = {"configuration_file": abs_config_path}
    arguments.update(params)

    timestamp = get_timestamp()
    safe_name = tool_name.replace(" ", "_").lower()
    filename = f"{timestamp}_{safe_name}.json"
    filepath = str(JOBS_DIR / filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(arguments, f, indent=2, ensure_ascii=False, default=str)

    return filepath


def load_config_template(config_name: str) -> Optional[Dict[str, Any]]:
    """Load an existing GWS config template.

    Args:
        config_name: Config filename (e.g., "bs_angular_renormalization.json").

    Returns:
        Parsed JSON dict or None if not found.
    """
    config_path = GWS_CONF_PATH / config_name
    if not config_path.exists():
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_tool1_json(
    input_files: List[str],
    sounder_type: str = "AUTO",
    projection: str = "自动检测",
    resolution: str = "2.0",
    gap_fill: str = "None",
    elev_min: Optional[float] = None,
    elev_max: Optional[float] = None,
    use_snippets: bool = False,
    use_svp: bool = True,
    use_insonified_area: bool = True,
    remove_compensation: bool = True,
    remove_calibration: bool = True,
    output_dir: Optional[str] = None,
) -> str:
    """Build JSON config for Tool 1: Export Reference DTM.

    Generates arguments that will be processed by pyat's
    avg_backscatter_model.json config template internally.
    """
    params = {
        "i_paths": input_files,
        "sounder_type": sounder_type,
        "use_snippets": use_snippets,
        "use_svp": use_svp,
        "use_insonified_area": use_insonified_area,
        "remove_compensation": remove_compensation,
        "remove_calibration": remove_calibration,
        "projection": projection,
        "resolution": resolution,
        "gap_fill": gap_fill,
    }
    if elev_min is not None:
        params["elevation_min"] = elev_min
    if elev_max is not None:
        params["elevation_max"] = elev_max
    if output_dir:
        params["o_dir"] = output_dir

    return build_args_json(
        config_file_relpath="sonar/bs/avg_backscatter_model.json",
        params=params,
        tool_name="tool1_ref_dtm",
    )


def build_tool2a_json(
    input_files: List[str],
    bathy_nc: str,
    sounder_type: str = "AUTO",
    sliding_window: int = 10,
    ref_angle_min: float = 30.0,
    ref_angle_max: float = 60.0,
    use_snippets: bool = False,
    use_svp: bool = True,
    use_insonified_area: bool = True,
    remove_calibration: bool = True,
    output_bsar: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Build JSON config for Tool 2A: Sliding Angular Renormalization."""
    # Generate o_paths: same dir as input, _bs_sliding suffix
    o_paths = []
    for fp in input_files:
        base, ext = os.path.splitext(fp)
        out = f"{base}_bs_sliding{ext}"
        if output_dir:
            out = os.path.join(output_dir, os.path.basename(out))
        o_paths.append(out)

    params = {
        "i_paths": input_files,
        "o_paths": o_paths,
        "sounder_type": sounder_type,
        "sliding_window": sliding_window,
        "ref_angle_min": ref_angle_min,
        "ref_angle_max": ref_angle_max,
        "use_snippets": use_snippets,
        "use_svp": use_svp,
        "use_insonified_area": use_insonified_area,
        "remove_calibration": remove_calibration,
    }
    if output_bsar:
        params["o_bsar"] = output_bsar

    return build_args_json(
        config_file_relpath="sonar/bs/bs_sliding_angular_renormalization.json",
        params=params,
        tool_name="tool2a_sliding",
    )


def build_tool2b_step2a_json(
    input_files: List[str],
    bsar_nc: str,
    bathy_nc: str = "",
    reference_level: float = -20.0,
    apply_compensation: bool = True,
    use_snippets: bool = False,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Build JSON config for Statistical Angular Response (BSAR).

    Applies a pre-computed BSAR model to XSF files and produces corrected
    XSF files (``_bs_renorm`` suffix) with ``backscatterCorrection = ON``.
    Backend: sonar/bs/bs_angular_renormalization.json → xsf_constant_process
    """
    # Auto-generate output paths: same dir as input, _bs_renorm suffix
    o_paths = []
    for fp in input_files:
        base, ext = os.path.splitext(fp)
        out = f"{base}_bs_renorm{ext}"
        if output_dir:
            out = os.path.join(output_dir, os.path.basename(out))
        o_paths.append(out)

    params = {
        "i_paths": input_files,
        "o_paths": o_paths,
        "mean_model_file": bsar_nc,
        "reference_level": reference_level,
        "apply_compensation": apply_compensation,
        "use_snippets": use_snippets,
    }
    if bathy_nc:
        params["i_dtm"] = bathy_nc
    if overwrite:
        params["overwrite"] = True

    return build_args_json(
        config_file_relpath="sonar/bs/bs_angular_renormalization.json",
        params=params,
        tool_name="tool2b_bsar_apply",
    )


def build_tool2b_step2b_json(
    input_files: List[str],
    bathy_nc: str = "",
    sounder_type: str = "AUTO",
    use_snippets: bool = False,
    use_svp: bool = True,
    use_insonified_area: bool = True,
    remove_compensation: bool = True,
    remove_calibration: bool = True,
    integration_method: str = "MEAN",
    linear_scale: str = "AMPLITUDE",
    output_bsar: str = "",
) -> str:
    """Build JSON config for Static Angular Renormalisation — BSAR model computation.

    Backend: sonar/bs/avg_backscatter_model.json → stats_computer.compute_mean_model_process
    """
    params = {
        "i_paths": input_files,
        "sounder_type": sounder_type,
        "use_snippets": use_snippets,
        "use_svp": use_svp,
        "use_insonified_area": use_insonified_area,
        "remove_compensation": remove_compensation,
        "remove_calibration": remove_calibration,
        "integration_method": integration_method,
        "linear_scale": linear_scale,
    }
    if bathy_nc:
        params["i_dtm"] = bathy_nc
    if output_bsar:
        params["o_path"] = output_bsar

    return build_args_json(
        config_file_relpath="sonar/bs/avg_backscatter_model.json",
        params=params,
        tool_name="tool2b_bsar_compute",
    )


def build_sounder_to_dtm_json(
    input_files: List[str],
    output_files: List[str],
    target_resolution: str = "0.000277778",
    target_spatial_reference: str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
    layers: Optional[List[str]] = None,
    gap_filling: bool = False,
    mask_size: int = 3,
    valid_sounds_only: bool = True,
    spatial_antialiasing: bool = False,
    min_elevation: Optional[float] = None,
    max_elevation: Optional[float] = None,
    min_sounds: Optional[int] = None,
    overwrite: bool = False,
    title: str = "",
    institution: str = "",
    source: str = "",
    references: str = "",
    comment: str = "",
    quality_indicator: bool = False,
    coord: Optional[str] = None,
) -> str:
    """Build JSON config for SounderToDtm exporter.

    Maps to: src/gws/conf/dtm/convert/sounder_to_dtm.json
    Parameters include: resolution, projection, gap filling, spatial antialiasing,
                        valid sounds only, min/max elevation, min sounds, layers.
    """
    params = {
        "i_paths": input_files,
        "o_paths": output_files,
        "overwrite": overwrite,
        "target_resolution": target_resolution,
        "target_spatial_reference": target_spatial_reference,
        "gap_filling": gap_filling,
        "mask_size": mask_size,
        "valid_sounds_only": valid_sounds_only,
        "spatial_antialiasing": spatial_antialiasing,
    }
    if coord:
        params["coord"] = coord
    if layers:
        params["layers"] = layers
    if min_elevation is not None:
        params["min_elevation"] = min_elevation
    if max_elevation is not None:
        params["max_elevation"] = max_elevation
    if min_sounds is not None:
        params["min_sounds"] = min_sounds
    if title:
        params["title"] = title
    if institution:
        params["institution"] = institution
    if source:
        params["source"] = source
    if references:
        params["references"] = references
    if comment:
        params["comment"] = comment
    if quality_indicator:
        params["quality_indicator"] = quality_indicator

    return build_args_json(
        config_file_relpath="dtm/convert/sounder_to_dtm.json",
        params=params,
        tool_name="sounder_to_dtm",
    )


# ── Water Column tools ─────────────────────────────────────────────

WC_CONFIG_MAP = {
    "horizontal": "sonar/wc/horizontal_section.json",
    "longitudinal": "sonar/wc/longitudinal_section.json",
    "polar": "sonar/wc/polar_echograms.json",
    "vertical": "sonar/wc/vertical_integration.json",
}

WC_SUFFIX_MAP = {
    "horizontal": "WCHorizontalEcho.g3d.nc",
    "longitudinal": "WCLongitudinalEcho.g3d.nc",
    "polar": "PolarEchograms.g3d.nc",
    "vertical": "WCVerticalEcho.tiff",
}


def build_wc_json(
    mode: str,
    input_files: List[str],
    output_dir: str = "",
    output_prefix: str = "wc_",
    overwrite: bool = True,
    **params,
) -> str:
    """Build JSON config for Water Column tools.

    Args:
        mode: One of "horizontal" / "longitudinal" / "polar" / "vertical".
        input_files: List of XSF file paths.
        output_dir: Output directory (empty = same as input).
        output_prefix: Prefix for output filenames.
        overwrite: Allow overwrite existing output.
        **params: Mode-specific parameters (delta_elevation, grid_count, ...).
    """
    config_relpath = WC_CONFIG_MAP[mode]
    suffix = WC_SUFFIX_MAP.get(mode, "WC.nc")

    # Generate o_paths: one output per input file
    o_paths = []
    for fp in input_files:
        base = os.path.splitext(os.path.basename(fp))[0]
        if output_dir:
            o_paths.append(os.path.join(output_dir,
                f"{output_prefix}{base}_{suffix}"))
        else:
            in_dir = os.path.dirname(fp) or "."
            o_paths.append(os.path.join(in_dir,
                f"{output_prefix}{base}_{suffix}"))

    args = {
        "i_paths": input_files,
        "o_paths": o_paths,
        "overwrite": overwrite,
    }
    # Merge mode-specific params (keys already match GWS config names)
    for k, v in params.items():
        if v is None or v == "":
            continue
        # parameters that use 0 or 0.0 to mean "auto"
        auto_zero_params = (
            "delta_elevation", "delta_across", "delta_along",
            "sample_resolution", "height", "grid_count", "target_resolution"
        )
        if k in auto_zero_params and v in (0, 0.0):
            continue
            
        # vertical integration does not accept layers
        if mode == "vertical" and k == "layers":
            continue

        # Always include these params even if 0/False (GWS eval may use them)
        always_include = (
            "normalization_offset", "vertical_offset", 
            "interpolate", "enable_normalization", "vertical_reference",
        )
        if k in always_include:
            args[k] = v
        elif k in auto_zero_params:
            args[k] = v
        elif v is not None and v != "" and v != 0.0 and v is not False:
            args[k] = v
        elif k in ("filters", "layers", "coord") and v:
            args[k] = v

    return build_args_json(
        config_file_relpath=config_relpath,
        params=args,
        tool_name=f"wc_{mode}",
    )
