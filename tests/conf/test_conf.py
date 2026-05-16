#!/usr/bin/env python
"""
Checks JSON files in conf/:
- For each .json file, extracts the "function" key if present
- Tries to import the function (importlib + getattr)
- Displays a summary of successes/failures
"""
from __future__ import annotations

import json
import os
import sys
from importlib import import_module
from typing import Dict, List, Tuple

import pytest

# Ensures that the repository root is in sys.path to import pyat.*
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CONF_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "src", "gws", "conf"))


def _extract_function_from_json(json_path: str) -> Tuple[str, str]:
    """Reads a JSON file and returns (module_path, function_name) for the "function" key.
    Raises KeyError if the key is missing, ValueError if the format is invalid.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data: Dict = json.load(f)
    if "function" not in data:
        raise KeyError("key 'function' is missing")
    func_value = data["function"]
    if not isinstance(func_value, str) or "." not in func_value:
        raise ValueError("invalid 'function' value (expected 'module.attr' string)")
    module_path, func_name = func_value.rsplit(".", 1)
    return module_path, func_name


def _try_import(module_path: str, func_name: str):
    """Tries to import module_path and access the func_name attribute."""
    mod = import_module(module_path)
    getattr(mod, func_name)  # raises AttributeError if missing


def _iter_json_files(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def _get_indexed_json_files(index_path: str) -> List[str]:
    """
    Recursively finds all JSON file paths from an index file.
    """
    indexed_files = set()
    conf_root = os.path.dirname(index_path)

    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def recurse_find_apps(d):
        if isinstance(d, dict):
            for key, value in d.items():
                if key == "apps" and isinstance(value, list):
                    for app_path in value:
                        full_path = os.path.abspath(os.path.join(conf_root, app_path))
                        indexed_files.add(full_path)
                else:
                    recurse_find_apps(value)
        elif isinstance(d, list):
            for item in d:
                recurse_find_apps(item)

    recurse_find_apps(data)
    return sorted(list(indexed_files))


def test_conf():
    """
    Tests that all functions referenced in the JSON configuration files
    can be imported.
    """
    json_files = _iter_json_files(CONF_ROOT)
    assert json_files, f"No JSON files found under: {CONF_ROOT}"

    import_failures: List[str] = []
    json_errors: List[str] = []

    for path in json_files:
        rel_path = os.path.relpath(path, REPO_ROOT)
        try:
            module_path, func_name = _extract_function_from_json(path)
        except KeyError:
            continue  # 'function' key is optional
        except Exception as e:  # Invalid JSON, ValueError, etc.
            json_errors.append(f"{rel_path}: {e}")
            continue

        try:
            _try_import(module_path, func_name)
        except Exception as e:
            import_failures.append(f"{rel_path} -> {module_path}.{func_name}: {e}")

    # Check that all files in index.json exist
    index_json_path = os.path.join(CONF_ROOT, "index.json")
    indexed_files = _get_indexed_json_files(index_json_path)
    missing_indexed_files = [os.path.relpath(p, REPO_ROOT) for p in indexed_files if not os.path.exists(p)]

    num_ok = len(json_files) - len(import_failures) - len(json_errors)
    print(f"{num_ok} / {len(json_files)} JSON files are OK.")

    error_messages = []
    if import_failures:
        error_messages.append("-- Import failures --")
        error_messages.extend(import_failures)
    if json_errors:
        error_messages.append("-- JSON errors --")
        error_messages.extend(json_errors)
    if missing_indexed_files:
        error_messages.append("-- Files in index.json not found --")
        error_messages.extend(missing_indexed_files)
    if error_messages:
        msg = "Configuration errors found:\n" + "\n".join(error_messages)
        assert not error_messages, msg
