import os
import sys

# ── Dynamic DLL and environment variables configuration for Windows ──
if sys.platform == 'win32':
    # Get the directory of the running python interpreter
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
