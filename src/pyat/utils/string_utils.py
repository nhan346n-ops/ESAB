#! /usr/bin/env python3
# coding: utf-8
from typing import List


def xstr(s: str) -> str:
    """
    Returns: empty string if null otherwise return the object
    """
    return s or ""


def trim_string_array(string_array: List[str]) -> List[str]:
    """
    remove trailing empty strings in the given array but keep the ones in the middle
    return a shorter list of empty strings without the trailing empty strings
    """
    # find the last non empty name
    last_value_index = None
    for index, name in enumerate(string_array):
        if isinstance(name, str) and name.strip() != "":
            last_value_index = index
    if last_value_index is None:
        return []
    else:
        return string_array[0 : last_value_index + 1]


def upper_camel_case(s: str):
    """
    Returns: string formated with capital for first letter of each word (MY_ENUM -> MyEnum)
    """
    return "".join(word.capitalize() for word in s.split("_"))


def spaced_upper_camel_case(s: str):
    """
    Returns: string formated with capital for first letter of each word (MY_ENUM -> My Enum)
    """
    return " ".join(word.capitalize() for word in s.split("_"))
