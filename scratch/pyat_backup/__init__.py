import os
import sys

# ── Dynamic DLL and environment variables configuration for Windows ──
if sys.platform == 'win32':
    if getattr(sys, 'frozen', False):
        # Packaged application DLL and environment setup
        # 1. Clean PATH to avoid conflicts with Anaconda
        path_list = os.environ.get("PATH", "").split(os.pathsep)
        clean_path = [p for p in path_list if not ('anaconda' in p.lower() or 'conda' in p.lower())]
        os.environ["PATH"] = os.pathsep.join(clean_path)
        
        # 2. Add DLL search directories
        internal_dir = os.path.dirname(os.path.abspath(__file__))
        _internal = os.path.dirname(internal_dir)
        pyside_dir = os.path.join(_internal, "PySide6")
        if sys.version_info >= (3, 8):
            try:
                os.add_dll_directory(_internal)
                os.add_dll_directory(pyside_dir)
            except Exception:
                pass
                
        # 3. Configure GDAL and PROJ environment variables for packaged app
        gdal_data = os.path.join(_internal, "Library", "share", "gdal")
        if os.path.isdir(gdal_data):
            os.environ["GDAL_DATA"] = gdal_data
            
        proj_lib = os.path.join(_internal, "pyproj", "proj_dir", "share", "proj")
        if os.path.isdir(proj_lib):
            os.environ["PROJ_LIB"] = proj_lib
    else:
        # Development environment: Get the directory of the running python interpreter
        py_bin = os.path.dirname(sys.executable)
        if os.path.basename(py_bin).lower() == "scripts":
            env_root = os.path.dirname(py_bin)
        else:
            env_root = py_bin

        # Library/bin contains gdal, proj, geos etc. DLLs in conda envs
        conda_lib_bin = os.path.join(env_root, "Library", "bin")
        if os.path.isdir(conda_lib_bin):
            # 1. Prepend to system PATH
            if conda_lib_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = conda_lib_bin + os.pathsep + os.environ.get("PATH", "")
            
            # 2. Ensure USE_PATH_FOR_GDAL_PYTHON is NOT set to avoid DLL collisions
            os.environ.pop("USE_PATH_FOR_GDAL_PYTHON", None)

            # 3. For Python 3.8+, explicitly add DLL directory to the search path
            if sys.version_info >= (3, 8):
                try:
                    os.add_dll_directory(conda_lib_bin)
                except Exception:
                    pass
            
            # 4. Configure GDAL plugins path (needed for netCDF driver)
            gdal_plugin = os.path.normpath(os.path.join(conda_lib_bin, "..", "lib", "gdalplugins"))
            if os.path.isdir(gdal_plugin):
                os.environ["GDAL_DRIVER_PATH"] = gdal_plugin
                
            # 5. Configure GDAL data path (needed for coordinate transformations / ESPG database)
            gdal_data = os.path.join(env_root, "Library", "share", "gdal")
            if os.path.isdir(gdal_data):
                os.environ["GDAL_DATA"] = gdal_data
                
            # 6. Configure PROJ library path (needed for pyproj/proj transformations)
            proj_lib = os.path.join(env_root, "Library", "share", "proj")
            if os.path.isdir(proj_lib):
                os.environ["PROJ_LIB"] = proj_lib

# Monkeypatch argument_utils to prevent KeyError: 'help', KeyError: 'long_key', ValueError
try:
    import argparse
    import json
    import pyat.utils.argument_utils
    orig_create_argv_parser = pyat.utils.argument_utils.create_argv_parser

    def patched_create_argv_parser(process_name, json_config_file_path):
        orig_json_load = json.load
        orig_add_argument = argparse.ArgumentParser.add_argument

        def wrapped_json_load(*args, **kwargs):
            conf = orig_json_load(*args, **kwargs)
            if isinstance(conf, dict) and "parameters" in conf:
                for param in conf["parameters"]:
                    if "help" not in param:
                        param["help"] = None
                    if "name" not in param:
                        param["name"] = ""
                    if "long_key" not in param:
                        param["long_key"] = None
                    if "nargs" in param and param["nargs"] not in ["?", "+", "*", None]:
                        nargs_val = str(param["nargs"]).split("|")[0].strip()
                        if nargs_val in ["?", "+", "*"]:
                            param["nargs"] = nargs_val
                        else:
                            try:
                                param["nargs"] = int(nargs_val)
                            except ValueError:
                                param["nargs"] = None
            return conf

        def wrapped_add_argument(self, *args, **kwargs):
            clean_args = [a for a in args if a is not None]
            return orig_add_argument(self, *clean_args, **kwargs)

        json.load = wrapped_json_load
        argparse.ArgumentParser.add_argument = wrapped_add_argument
        try:
            return orig_create_argv_parser(process_name, json_config_file_path)
        finally:
            json.load = orig_json_load
            argparse.ArgumentParser.add_argument = orig_add_argument

    pyat.utils.argument_utils.create_argv_parser = patched_create_argv_parser
except Exception:
    pass
