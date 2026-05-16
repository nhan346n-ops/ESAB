#! /usr/bin/env python3
# coding: utf-8

import sonar_netcdf.process.sonar_file_merger as sfm

from pyat.xsf.netcdf_merger_bridge import NcMergerBridge


class XsfMergerBridge(NcMergerBridge):
    def __init__(self, **kwargs):
        super().__init__(nc_merger_class=sfm.SNMerger, **kwargs)
