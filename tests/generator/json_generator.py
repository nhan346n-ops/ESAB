#! /usr/bin/env python3
# coding: utf-8


import json
import os

# Example for merge

params = {}
params["merge_type"] = "FILL"
params["spatial_reso"] = 3
params["i_files"] = []
directory = r"D:\utilisateurs\gguardia\test\dtm\netcdf4"

params["i_files"].append(os.path.join(directory, "generated_16x16_pattern_10.nc"))
params["i_files"].append(os.path.join(directory, "generated_16x16_pattern_20.nc"))


with open("params_merge.json", "w", encoding="utf-8") as f:
    json.dump(params, f, indent=4)
