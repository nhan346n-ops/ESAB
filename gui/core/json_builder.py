"""JSON configuration builder for pyat backend tools.

Generates JSON parameter files matching the format expected by:
  python -m pyat <config.json>
"""
import json
from typing import Dict, Any, List, Optional

from ..utils.config import (
    JOBS_DIR, GWS_CONF_PATH, get_timestamp, ensure_dirs,
)


def build_args_json(
    config_file_relpath: str,
    params: Dict[str, Any],
    tool_name: str,
) -> str:
    """Build an arguments JSON file for pyat execution.

    Args:
        config_file_relpath: Relative path to function config JSON
            (e.g., "sonar/bs/bs_sliding_angular_renormalization.json")
        params: Dictionary of parameter values.
        tool_name: Tool name for file naming.

    Returns:
        Absolute path to the generated JSON file.
    """
    ensure_dirs()

    arguments = {"configuration_file": config_file_relpath}
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
    projection: str = "Auto Detect",
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
    output_bsar: bool = False,
) -> str:
    """Build JSON config for Tool 2A: Sliding Angular Renormalization."""
    params = {
        "i_paths": input_files,
        "i_dtm": bathy_nc,
        "sounder_type": sounder_type,
        "sliding_window": sliding_window,
        "ref_angle_min": ref_angle_min,
        "ref_angle_max": ref_angle_max,
        "use_snippets": use_snippets,
        "use_svp": use_svp,
        "use_insonified_area": use_insonified_area,
        "remove_calibration": remove_calibration,
    }

    return build_args_json(
        config_file_relpath="sonar/bs/bs_sliding_angular_renormalization.json",
        params=params,
        tool_name="tool2a_sliding",
    )


def build_tool2b_step2a_json(
    input_files: List[str],
    bathy_nc: str,
    sounder_type: str = "AUTO",
    use_snippets: bool = False,
    use_svp: bool = True,
    use_insonified_area: bool = True,
    remove_compensation: bool = True,
    remove_calibration: bool = True,
    integration_method: str = "MEAN",
    linear_scale: str = "AMPLITUDE",
    mask_files: Optional[List[str]] = None,
    output_bsar: str = "",
) -> str:
    """Build JSON config for Tool 2B Step 2a: Statistical BSAR."""
    params = {
        "i_paths": input_files,
        "i_dtm": bathy_nc,
        "sounder_type": sounder_type,
        "use_snippets": use_snippets,
        "use_svp": use_svp,
        "use_insonified_area": use_insonified_area,
        "remove_compensation": remove_compensation,
        "remove_calibration": remove_calibration,
        "integration_method": integration_method,
        "linear_scale": linear_scale,
    }
    if mask_files:
        params["mask"] = mask_files
    if output_bsar:
        params["o_path"] = output_bsar

    return build_args_json(
        config_file_relpath="sonar/bs/avg_backscatter_model.json",
        params=params,
        tool_name="tool2b_bsar",
    )


def build_tool2b_step2b_json(
    input_files: List[str],
    bsar_nc: str,
    bathy_nc: str,
    reference_level: float = -20.0,
    apply_compensation: bool = True,
    use_snippets: bool = False,
) -> str:
    """Build JSON config for Tool 2B Step 2b: Apply BSAR Renormalization."""
    params = {
        "i_paths": input_files,
        "mean_model_file": bsar_nc,
        "i_dtm": bathy_nc,
        "reference_level": reference_level,
        "apply_compensation": apply_compensation,
        "use_snippets": use_snippets,
    }

    return build_args_json(
        config_file_relpath="sonar/bs/bs_angular_renormalization.json",
        params=params,
        tool_name="tool2b_apply_bsar",
    )


def build_tool3_json(
    input_files: List[str],
    projection: str = "Auto Detect",
    resolution: str = "2.0",
    gap_fill: str = "None",
    elev_min: Optional[float] = None,
    elev_max: Optional[float] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Build JSON config for Tool 3: Grid Backscatter Mosaic."""
    params = {
        "i_paths": input_files,
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
        tool_name="tool3_mosaic",
    )
