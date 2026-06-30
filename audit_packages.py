import os
import re
import sys
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
gui_dir = os.path.join(script_dir, "gui")
internal_dir = os.path.join(script_dir, "dist", "BS_Processor", "_internal")

# Standard library modules to filter out
stdlib = {
    "os", "sys", "time", "traceback", "datetime", "json", "enum", "typing", "abc", 
    "math", "re", "shutil", "glob", "uuid", "base64", "io", "tempfile", "multiprocessing",
    "inspect", "ctypes", "pathlib", "subprocess", "importlib", "logging", "collections",
    "functools", "warnings", "xml", "hashlib", "socket", "struct", "threading", "queue",
    "select", "signal", "copy", "platform", "sysconfig", "difflib", "sqlite3", "locale",
    "numbers", "weakref", "array", "bisect", "contextlib", "operator", "itertools", "parse"
}

# Scan source code for imports
imported_pkgs = set()
import_pattern = re.compile(r"^\s*(?:import\s+(\w+)|from\s+(\w+)\s+import)", re.MULTILINE)

def scan_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        for m in import_pattern.finditer(content):
                            pkg = m.group(1) or m.group(2)
                            if pkg:
                                imported_pkgs.add(pkg)
                except Exception as e:
                    pass

scan_files(src_dir)
scan_files(gui_dir)

# Filter standard library modules and internal project modules
external_imports = set()
for pkg in imported_pkgs:
    if pkg in stdlib:
        continue
    # Filter local modules of pyat-main
    if pkg in ["pyat", "gui", "src"]:
        continue
    external_imports.add(pkg)

print("=== Third-Party Packages Imported in Code ===")
print(sorted(list(external_imports)))

# Check presence in _internal directory
print("\n=== Auditing Package Presence in dist/BS_Processor/_internal ===")
missing = []
present = []

# List files and folders in _internal
internal_contents = [name.lower() for name in os.listdir(internal_dir)] if os.path.isdir(internal_dir) else []

# Some package names map to different folder/file names, map them
mapping = {
    "cv2": ["cv2"],
    "PIL": ["pil"],
    "OpenGL": ["opengl"],
    "sklearn": ["sklearn"],
    "skimage": ["skimage"],
    "skspatial": ["skspatial"],
    "progress": ["progress"],
    "haversine": ["haversine"],
    "geopy": ["geopy"],
    "seaborn": ["seaborn"],
    "requests": ["requests"],
    "pyogrio": ["pyogrio"],
    "dataclasses_json": ["dataclasses_json"],
    "mhkit": ["mhkit"],
    "httpx": ["httpx"],
    "result": ["result"],
    "pydantic": ["pydantic"],
    "heightmap_interpolation": ["heightmap_interpolation"],
    "rsocket": ["rsocket"],
    "pytide": ["pytide"],
    "sonarnative": ["sonarnative"],
    "pynvi": ["pynvi"],
    "pytechsas": ["pytechsas"],
    "pygws": ["pygws"],
    "sonar_netcdf": ["sonar_netcdf"],
    "laspy": ["laspy"],
    "rioxarray": ["rioxarray"],
    "pyfes": ["pyfes"],
    "pyigrf": ["pyigrf"],
    "numba": ["numba"],
    "netCDF4": ["netcdf4"],
    "pyproj": ["pyproj"],
    "scipy": ["scipy"],
    "pandas": ["pandas"],
    "shapely": ["shapely"],
    "fiona": ["fiona"],
    "rasterio": ["rasterio"],
    "geopandas": ["geopandas"],
    "h5netcdf": ["h5netcdf"],
    "h5py": ["h5py"],
    "dask": ["dask"],
    "distributed": ["distributed"],
    "cftime": ["cftime"],
    "numpy": ["numpy"],
    "matplotlib": ["matplotlib"],
    "PySide6": ["pyside6"],
    "shiboken6": ["shiboken6"],
    "pyqtgraph": ["pyqtgraph"]
}

for pkg in external_imports:
    # Check mapping
    possible_names = mapping.get(pkg, [pkg.lower()])
    is_found = False
    for p_name in possible_names:
        # Check folder presence
        if p_name in internal_contents:
            is_found = True
            break
        # Check file/extension presence (like pynvi.py, sonarnative.pyd)
        if any(f.startswith(p_name + ".") for f in internal_contents):
            is_found = True
            break
            
    if is_found:
        present.append(pkg)
    else:
        # Check if it's imported sub-module (e.g. from osgeo import osr -> osgeo folder present)
        # Try to search for top level folder or package info
        is_sub = False
        for p_name in mapping.values():
            if any(p in internal_contents for p in p_name):
                pass
        if not is_found:
            missing.append(pkg)

print("\n--- Present Packages (Audited OK) ---")
for p in sorted(present):
    print(f"[OK] {p}")

print("\n--- Potentially Missing Packages ---")
if missing:
    for m in sorted(missing):
        print(f"[MISSING/ALERT] {m}")
else:
    print("None! All imported third-party packages are verified present.")

# Check for DLL integrity
print("\n=== Auditing Critical System DLLs ===")
critical_dlls = ["proj_9.dll", "gdal.dll", "lz4.dll", "libssl-3-x64.dll", "libcrypto-3-x64.dll", "netcdf.dll"]
for dll in critical_dlls:
    found = dll.lower() in internal_contents
    status = "[OK]" if found else "[MISSING!]"
    print(f"{status} {dll}")
