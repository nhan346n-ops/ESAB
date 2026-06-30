import sys
import os
import subprocess

# Setup DLL paths on Windows first
if sys.platform == 'win32':
    py_bin = os.path.dirname(sys.executable)
    env_root = os.path.dirname(py_bin) if os.path.basename(py_bin).lower() == "scripts" else py_bin
    conda_lib_bin = os.path.join(env_root, "Library", "bin")
    if os.path.isdir(conda_lib_bin):
        path_list = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        first_path = os.path.normpath(path_list[0]).lower() if path_list else ""
        conda_lib_bin_norm = os.path.normpath(conda_lib_bin).lower()

        if first_path != conda_lib_bin_norm and not os.environ.get("PYAT_ENV_RESTARTED"):
            os.environ["PATH"] = conda_lib_bin + os.pathsep + os.environ.get("PATH", "")
            os.environ["PYAT_ENV_RESTARTED"] = "1"
            os.environ.pop("USE_PATH_FOR_GDAL_PYTHON", None)
            sys.exit(subprocess.call([sys.executable] + sys.argv, env=os.environ.copy()))

        os.environ.pop("USE_PATH_FOR_GDAL_PYTHON", None)
        if sys.version_info >= (3, 8):
            try:
                os.add_dll_directory(conda_lib_bin)
            except Exception:
                pass

import importlib
import importlib.metadata

dependencies = [
    # Conda packages
    ("conda_pack", "conda-pack"),
    ("osgeo.gdal", "gdal"),
    ("netCDF4", "netcdf4"),
    ("dask", "dask"),
    ("distributed", "distributed"),
    ("cftime", "cftime"),
    ("xarray", "xarray"),
    ("rioxarray", "rioxarray"),
    ("pyfes", "pyfes"),
    ("pyIGRF", "pyigrf"),
    ("skimage", "scikit-image"),
    ("sklearn", "scikit-learn"),
    ("skspatial", "scikit-spatial"),
    ("scipy", "scipy"),
    ("numba", "numba"),
    ("progress", "progress"),
    ("pandas", "pandas"),
    ("geopandas", "geopandas"),
    ("pyproj", "pyproj"),
    ("haversine", "haversine"),
    ("geopy", "geopy"),
    ("seaborn", "seaborn"),
    ("requests", "requests"),
    ("fiona", "fiona"),
    ("pyogrio", "pyogrio"),
    ("dataclasses_json", "dataclasses-json"),
    ("mhkit", "mhkit"),
    ("httpx", "httpx"),
    ("result", "result"),
    ("pydantic", "pydantic"),
    ("cv2", "opencv-python"),
    ("heightmap_interpolation", "heightmap-interpolation"),
    ("rsocket", "rsocket"),
    ("pytide", "pytide"),
    ("sonarnative", "sonarnative"),
    ("pynvi", "pynvi"),
    ("pytechsas", "pytechsas"),
    ("pygws", "pygws"),
    ("sonar_netcdf", "sonar-netcdf")
]

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("-" * 50)
print(f"{'Package (Import)':<30} | {'Status':<10} | {'Installed Version':<20}")
print("-" * 50)

missing = []
for import_name, req_name in dependencies:
    try:
        mod = importlib.import_module(import_name)
        # Try to get version
        version = "unknown"
        try:
            version = importlib.metadata.version(req_name)
        except Exception:
            try:
                version = mod.__version__
            except Exception:
                pass
        print(f"{import_name:<30} | {'OK':<10} | {version:<20}")
    except Exception as e:
        print(f"{import_name:<30} | {'FAILED':<10} | Error: {str(e)}")
        missing.append((import_name, req_name))

print("-" * 50)
if missing:
    print(f"Missing or broken packages ({len(missing)}):")
    for import_name, req_name in missing:
        print(f" - {req_name} (import as {import_name})")
else:
    print("All packages imported successfully!")
