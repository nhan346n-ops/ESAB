#! /usr/bin/env python3
# coding: utf-8

import os
import shutil
import tempfile
from typing import List
import numpy as np
from numpy.testing import assert_array_equal
import pyat.xsf.xsf_driver as xsf_driver
from pyat.xsf.xsf_updater import XsfUpdater, VARS_TO_TRANSFER
from tests.generator.xsf_generator import XsfGenerator


def generate_xsf(folder: str) -> str:
    """
    Creates a plain XSF file
    """
    generator = XsfGenerator(folder)
    return generator.initialize_file(
        latitude_min_deg=48.0,
        latitude_max_deg=48.005,
        longitude_min_deg=-4.005,
        longitude_max_deg=-4.0,
        ping_count=20,
        beam_count=20,
        min_depth_m=10.0,
        max_depth_m=20.0,
    )


def reset_var_to_transfert(xsf_path: str, values: List[int]) -> None:
    # Intialize variables to transfert
    with xsf_driver.open_xsf(xsf_path, "r+") as o_xsf_driver:
        for layer_path, value in zip(VARS_TO_TRANSFER, values):
            layer = o_xsf_driver.get_layer(layer_path)
            layer[:] = np.full(layer.shape, value)


def test_rectify_xsf():
    """
    Reports validity flags and bias corrections from a reference xsf to a target one
    """
    with tempfile.TemporaryDirectory() as o_path:
        # All layers are set with a different value in reference file
        ref_folder = tempfile.mkdtemp(dir=o_path)
        ref_xsf_path = generate_xsf(ref_folder)
        ref_values = range(1, len(VARS_TO_TRANSFER) + 1)
        reset_var_to_transfert(ref_xsf_path, ref_values)

        # All layers are set to 0 in input file
        i_xsf_path = os.path.join(o_path, os.path.basename(ref_xsf_path))
        shutil.copyfile(ref_xsf_path, i_xsf_path)
        reset_var_to_transfert(i_xsf_path, [0] * len(VARS_TO_TRANSFER))

        # Launch the process
        updater = XsfUpdater(i_paths=[i_xsf_path], o_paths=[i_xsf_path], i_ref=ref_folder, overwrite=True)
        updater()

        # Check all layers
        with xsf_driver.open_xsf(i_xsf_path) as i_xsf_driver:
            for layer_path, value in zip(VARS_TO_TRANSFER, ref_values):
                layer = i_xsf_driver.get_layer(layer_path)
                assert_array_equal(layer[:], np.full(layer.shape, value))
