import json
import os

import sonarnative
from sonarnative import SpatializerHolder


def apply_filters_file(json_file: str | None, spatializer: SpatializerHolder):
    """
    parse json filters configuration file and apply defined filters on current spatializer
    """
    if not json_file or not os.path.exists(json_file):
        return
    with open(json_file, "r", encoding="utf-8") as json_config_file:
        conf = json.load(json_config_file)
        apply_filters(conf, spatializer)


def apply_filters(json_filters: str | None, spatializer: SpatializerHolder):
    """
    parse json filters configuration file and apply defined filters on current spatializer
    """
    if not json_filters or len(json_filters) == 0:
        return
    conf = json.loads(json_filters)
    if "acrossDistance" in conf:
        param = conf["acrossDistance"]
        enable = param["enable"]
        min_value = param["minValue"]
        max_value = param["maxValue"]
        native_param = sonarnative.AcrossDistanceParameter(enable, min_value, max_value)
        sonarnative.apply_across_distance_filter(spatializer, native_param)
    if "threshold" in conf:
        param = conf["threshold"]
        enable = param["enable"]
        min_value = param["minValue"]
        max_value = param["maxValue"]
        native_param = sonarnative.ThresholdParameter(enable, min_value, max_value)
        sonarnative.apply_threshold_filter(spatializer, native_param)
    if "beam" in conf:
        param = conf["beam"]
        enable = param["enable"]
        min_value = param["minValue"]
        max_value = param["maxValue"]
        native_param = sonarnative.BeamIndexParameter(enable, min_value, max_value)
        sonarnative.apply_beam_index_filter(spatializer, native_param)
    if "sample" in conf:
        param = conf["sample"]
        enable = param["enable"]
        min_value = param["minValue"]
        max_value = param["maxValue"]
        native_param = sonarnative.SampleIndexParameter(enable, min_value, max_value)
        sonarnative.apply_sample_index_filter(spatializer, native_param)
    if "depth" in conf:
        param = conf["depth"]
        enable = param["enable"]
        min_value = param["minValue"]
        max_value = param["maxValue"]
        native_param = sonarnative.DepthParameter(enable, min_value, max_value)
        sonarnative.apply_depth_filter(spatializer, native_param)
    if "sidelobe" in conf:
        param = conf["sidelobe"]
        enable = param["enable"]
        threshold = param["threshold"]
        native_param = sonarnative.SideLobeParameter(enable, threshold)
        sonarnative.apply_side_lobe_filter(spatializer, native_param)
    if "bottom" in conf:
        param = conf["bottom"]
        enable = param["enable"]
        tolerance_absolute = param["toleranceAbsolute"]
        tolerance_percent = param["tolerancePercent"]
        angle_coefficient = param["angleCoefficient"]
        tolerance_type = param["type"]
        if tolerance_type == "RANGEPERCENT":
            native_param = sonarnative.BottomFilterParameter.new_range_percent(
                angle_coefficient, tolerance_percent, enable
            )
            sonarnative.apply_bottom_filter(spatializer, native_param)
        elif tolerance_type == "SAMPLE":
            native_param = sonarnative.BottomFilterParameter.new_sample(angle_coefficient, tolerance_absolute, enable)
            sonarnative.apply_bottom_filter(spatializer, native_param)
    if "sampling" in conf:
        param = conf["sampling"]
        sampling = param["sampling"]
        native_param = sonarnative.SamplingParameter(sampling)
        sonarnative.apply_sampling_filter(spatializer, native_param)
    if "specular" in conf:
        param = conf["specular"]
        enable = param["enable"]
        below = param["below"]
        tolerance = param["tolerance"]
        native_param = sonarnative.SpecularFilterParameter(enable, below, tolerance)
        sonarnative.apply_specular_filter(spatializer, native_param)
    if "multiping" in conf:
        param = conf["multiping"]
        enable = param["enable"]
        index = param["index"]
        native_param = sonarnative.MultipingParameter(enable, index)
        sonarnative.apply_multiping_filter(spatializer, native_param)
