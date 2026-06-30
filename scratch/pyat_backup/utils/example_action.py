#! /usr/bin/env python3
# coding: utf-8

from argparse import Action
import json

import pyat.dtm.dtm_standard_constants as dtm_constants


class ExampleAction(Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):

        params = vars(namespace)

        del params["example"]

        keys = params.keys()

        if "i_paths" in keys:
            params["i_paths"] = ["input.nc", "input_2.nc"]

        if "o_paths" in keys:
            params["o_paths"] = ["output.nc", "output_2.nc"]

        if "i_path" in keys:
            for ext in ["emo", "dtm", "nvi", "mbg"]:
                if ext in parser.prog:
                    params["i_path"] = "input." + ext
                    break

        if "o_path" in keys:
            if "shp" in parser.prog:
                params["o_path"] = "output.shp"
            else:
                params["o_path"] = "output.nc"

        if "layers" in keys:
            params["layers"] = {}
            for layer in dtm_constants.LAYERS:
                params["layers"][layer] = True

        if "coord" in keys:
            params["coord"] = {"north": None, "south": None, "west": None, "east": None}

        if "cdis" in keys:
            params["cdis"] = []
            params["cdis"].append({"old": "", "new": ""})
            params["cdis"].append({"old": "", "new": ""})

        # if "filters" in keys:
        #     params["filters"] = []
        #     params["filters"].append({"layer": dtm_constants.ELEVATION_NAME, "oper": cst.EQUAL, "a": 1})
        #     params["filters"].append({"layer": dtm_constants.ELEVATION_MIN, "oper": cst.BETWEEN, "a": 0, "b": 1})

        if "kml" in keys:
            params["kml"] = []
            params["kml"].append("input_1.kml")
            params["kml"].append("input_2.kml")

        if values:
            p_file = values
        else:
            p_file = self.default

        with open(p_file, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=4)

        parser.exit()
