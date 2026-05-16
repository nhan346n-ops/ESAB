#! /usr/bin/env python3
# coding: utf-8
import netCDF4 as nc
import numpy as np

import pyat.dtm.cdi.cdi_layer_util as cdi_util

# global variable to set to use or not exception
# pylint:disable=global-at-module-level
global _use_exception
_use_exception = True
# we should not use global variable but refactor with an statefull object "comparator"


def _check(condition, message):
    if _use_exception:
        assert condition, message
    else:
        if not condition:
            print(message)


def compare_variables_data(src: nc.Dataset, srcName: str, target: nc.Dataset, targetName: str, tolerance=10e-9):
    if srcName not in src.variables:
        print(f"Ignore variable {srcName} missing in source file")
        return
    a = src[srcName]
    if a.dtype == type("str") or a.dtype == "S1" or a.dtype == "U1":
        compare_cdi_variables(src, srcName, target, targetName)
        return
    _check(targetName in target.variables, f"variable {targetName} is expected and missing in target file")
    b = target[targetName]

    a_array = a[:]
    b_array = b[:]
    if hasattr(a_array, "mask"):
        _check(
            np.array_equal(a_array.mask, b_array.mask),
            f"Variable valid values mask differs between {a.name} and { b.name}",
        )

    if hasattr(a, "scale_factor"):
        # we take a tolerance of 10% of scale_factor
        tolerance = a.scale_factor * 0.1
    if len(a_array.shape) > 0:
        print(f">> {srcName} max difference = {np.max(np.abs(a_array - b_array))} ")
        _check(np.max(np.abs(a_array - b_array)) < tolerance, f"Variable values differs between {a.name} and {b.name}")


def compare_cdi_variables(src: nc.Dataset, srcName: str, target: nc.Dataset, targetName: str):
    if srcName not in src.variables:
        print(f"Ignore variable {srcName} missing in source file")
        return
    a = src[srcName]
    _check(targetName in target.variables, f"variable {targetName} is expected and missing in target file")
    b = target[targetName]
    values_a = a[:]
    if values_a.dtype in ("S1", "U1"):
        a_array = nc.chartostring(a[:])
    else:
        a_array = a[:]
    a_array = cdi_util.trim_string_array(a_array)
    values_b = b[:]
    if values_b.dtype in ("S1", "U1"):
        b_array = nc.chartostring(b[:])
    else:
        b_array = b[:]
    b_array = cdi_util.trim_string_array(b_array)
    _check(np.array_equal(a_array, b_array), f"Variable array differs between {a.name} and {b.name}")


def compare_variables(reference: nc.Dataset, target: nc.Dataset):
    in_src_only = [value for value in reference.variables if value not in target.variables]
    in_target_only = [value for value in target.variables if value not in reference.variables]
    _check(len(in_src_only) == 0, f"Variables declared in {reference.filepath()} : {in_src_only}")
    _check(len(in_target_only) == 0, f"Variables only declared in {target.filepath()} : {in_target_only}")

    # compare common variables
    intersect = [value for value in reference.variables if value in reference.variables]
    for name in intersect:
        # compare variable attributes
        print(f">Checking variable {name}")
        compare_attributes(reference.variables[name], target.variables[name])
        compare_variables_data(reference, name, target, name)


def compare_attributes(reference: nc.Dataset, target: nc.Dataset):
    ref_att = reference.ncattrs()
    target_att = target.ncattrs()

    in_src_only = [value for value in ref_att if value not in target_att]
    in_target_only = [value for value in target_att if value not in ref_att]
    _check(len(in_src_only) == 0, f"Attributes only declared in reference (first file) : {in_src_only}")
    _check(len(in_target_only) == 0, f"Attributes only declared in target (second file) : {in_target_only}")

    intersect = [value for value in ref_att if value in target_att]

    # now compare content
    for name in intersect:
        # check if we have an array
        _check(
            np.all(reference.getncattr(name) == target.getncattr(name))
            or (
                isinstance(reference.getncattr(name), (np.float64, np.float32))
                and np.all(np.isnan(reference.getncattr(name)) and np.all(np.isnan(target.getncattr(name))))
            ),
            f"Attribute {name} content differs {reference.getncattr(name)} vs {target.getncattr(name)}",
        )


def compare_dimensions(src: nc.Dataset, target: nc.Dataset):
    """compare all dimensions between an old dtm and a upgraded one, we do not recurse between groups"""

    in_src_only = [value for value in src.dimensions if value not in target.dimensions]
    in_target_only = [value for value in target.dimensions if value not in src.dimensions]
    intersect = [value for value in target.dimensions if value in src.dimensions]
    _check(len(in_src_only) == 0, f"Dimension only declared in {src.filepath()} : {in_src_only}")
    _check(len(in_target_only) == 0, f"Dimension only declared in {target.filepath()} : {in_target_only}")
    for dim in intersect:
        _check(
            len(src.dimensions[dim]) == len(target.dimensions[dim]),
            f"Dimension {dim} differs between files {len(src.dimensions[dim])} vs {len(target.dimensions[dim])} ({src.filepath()} vs {target.filepath()})",
        )


def compare(filename_A: str, filename_B: str):
    """
    Compare two netcdfile
    """

    # comparing file contents
    with nc.Dataset(filename_A) as src, nc.Dataset(filename_B) as target:
        print(">Check dimensions")
        compare_dimensions(src, target)
        print(">Check global Attributes")
        compare_attributes(src, target)
        print(">Check Variables")
        compare_variables(src, target)


if __name__ == "__main__":
    _use_exception = False

    compare("D:/D5.nc", "E:/D5.nc")
