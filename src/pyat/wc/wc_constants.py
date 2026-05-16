from typing import List

BACKSCATTER_MEAN = "backscatter_mean"
BACKSCATTER_MAX = "backscatter_max"
BACKSCATTER_VALUE_COUNT = "backscatter_value_count"

BACKSCATTER_COMP_MEAN = "backscatter_comp_mean"
BACKSCATTER_COMP_MAX = "backscatter_comp_max"
BACKSCATTER_COMP_VALUE_COUNT = "backscatter_comp_value_count"

ELEVATION = "elevation"
ELEVATION_COUNT = "elevation_count"

def contains_compensated_layer(layers: List[str]) -> bool:
    return BACKSCATTER_COMP_MEAN in layers or BACKSCATTER_COMP_MAX in layers

def contains_raw_layer(layers: List[str]) -> bool:
    return BACKSCATTER_MEAN in layers or BACKSCATTER_MAX in layers

def contains_elevation_layer(layers: List[str]) -> bool:
    return ELEVATION in layers
