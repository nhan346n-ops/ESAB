#! /usr/bin/env python3
# coding: utf-8

import argparse
import pyat.dtm.dtm_standard_constants as DtmConstants


def coord(string: str) -> dict:
    tmp = string.split(" ")

    if not len(tmp) in [3, 4]:
        msg = 'Coordinates must be like: "$north $south $west $east" '
        raise argparse.ArgumentTypeError(msg)

    result = {}
    result["north"] = tmp[0]
    result["south"] = tmp[1]
    result["west"] = tmp[2]
    result["east"] = tmp[3]
    return result


def layers(string: str) -> dict:
    tmp = string.split(" ")

    if not len(tmp) in range(1, 9):
        msg = 'Layers must be like: "$layer_1 $layer_2 ..." '
        raise argparse.ArgumentTypeError(msg)

    result = {}
    for layer in DtmConstants.LAYERS:
        if layer in tmp:
            result[layer] = True
        else:
            result[layer] = False
    return result


def filters(string: str) -> dict:
    tmp = string.split(" ")

    if not len(tmp) in [3, 4]:
        msg = 'Filter must be like: "$name_of_layer $name_of_operation $a ($b)"'
        raise argparse.ArgumentTypeError(msg)
    result = {}
    result["layer"] = tmp[0]
    result["oper"] = tmp[1]
    result["a"] = tmp[2]
    if len(tmp) > 3:
        result["b"] = tmp[3]
    return result


def cdi(string: str) -> dict:
    tmp = string.split(" ")

    if len(tmp) != 2:
        msg = 'Cdis must be like: "$cdi_to_change $new_name"'
        raise argparse.ArgumentTypeError(msg)

    cdis = {}
    cdis["old"] = tmp[0]
    cdis["new"] = tmp[1]

    return cdis
