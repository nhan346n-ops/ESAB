#! /usr/bin/env python3
# coding: utf-8

import os
import math
import numpy as np
from string import Template
import pyat.xyz.xyz_constants as XyzConstants

minute = 1 / 60.0
second = minute ** 2
RESOLUTION = 3.75 * second
NORTH = "north"
SOUTH = "south"
EAST = "east"
WEST = "west"
GEOBOX_1 = {SOUTH: 47, NORTH: 47 + minute, EAST: -4 + minute, WEST: -4}


class XyzGenerator:
    """Generator of xyz file (*.xyz)."""

    def __init__(self, directory: str):
        self.dir = directory
        self.t = Template("$lon$sep$lat$sep$ele\n")

    def create_file(self, name: str, geobox: dict, sep: str = ";") -> str:

        # 1 ligne, 1 cellule
        size_lat = math.ceil((geobox[EAST] - (geobox[WEST] + 0.5 * RESOLUTION)) / RESOLUTION)
        size_lon = math.ceil((geobox[NORTH] - (geobox[SOUTH] + 0.5 * RESOLUTION)) / RESOLUTION)

        lon = np.arange(geobox[WEST] + 0.5 * RESOLUTION, geobox[EAST], RESOLUTION)
        lat = np.arange(geobox[SOUTH] + 0.5 * RESOLUTION, geobox[NORTH], RESOLUTION)

        elev = np.full((size_lon, size_lat), 10.0)

        o_path = os.path.join(self.dir, name + XyzConstants.EXTENSION)
        with open(o_path, mode="w") as file:
            for row in range(size_lat):
                for col in range(size_lon):
                    file.write(self.t.substitute(lon=lon[col], lat=lat[row], sep=sep, ele=elev[row, col]))

        return o_path
