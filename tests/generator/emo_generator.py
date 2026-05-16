#! /usr/bin/env python3
# coding: utf-8

import os
import numpy as np
from string import Template


class EmoGenerator:
    """Generator of emo file (*.emo)."""

    def __init__(self, directory: str):
        self.dir = directory
        self.t = Template(
            "$lon$sep$lat$sep$min$sep$max$sep$ele$sep$std$sep$v_c$sep$int$sep$smo$sep$smo2$sep$cdi$sep$src$sep\n"
        )

    def create_1(self, name: str, sep: str = ";", n: int = 16) -> str:

        # 1 ligne, 1 cellule
        lon = np.linspace(-3.99947917, -3.98385417, n)
        lat = np.linspace(47.00052083, 47.01614583, n)

        cdi = [""]
        dtm_source = [""]

        elev = np.full((n, n), 10.0)
        min = np.full((n, n), 9.0)
        max = np.full((n, n), 11.0)
        value_count = np.full((n, n), 1.0)
        stdev = np.full((n, n), 0.0)
        smoothed = np.full((n, n), np.nan)
        smoothed_diff = np.full((n, n), np.nan)
        interpolation = np.full((n, n), 0)

        o_path = os.path.join(self.dir, name + ".emo")
        with open(o_path, mode="w") as file:
            for row in range(n):
                for col in range(n):
                    file.write(
                        self.t.substitute(
                            lon=lon[col],
                            lat=lat[row],
                            sep=sep,
                            cdi=cdi[0],
                            src=dtm_source[0],
                            ele=elev[row, col],
                            min=min[row, col],
                            max=max[row, col],
                            v_c=value_count[row, col],
                            std=stdev[row, col],
                            smo=smoothed[row, col],
                            smo2=smoothed_diff[row, col],
                            int=interpolation[row, col],
                        )
                    )

        return o_path
