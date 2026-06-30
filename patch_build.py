import os
import shutil
import glob

project_dir = os.path.dirname(os.path.abspath(__file__))
internal_dir = os.path.join(project_dir, "dist", "BS_Processor", "_internal")
conda_env = r"C:\Users\GUO\AppData\Local\anaconda3\envs\env_pyat_runtime"

print("1. Copying Conda DLLs...")
dll_src_pattern = os.path.join(conda_env, "Library", "bin", "*.dll")
copied_dlls = 0
for dll_file in glob.glob(dll_src_pattern):
    try:
        shutil.copy2(dll_file, internal_dir)
        copied_dlls += 1
    except Exception as e:
        pass
print(f"Copied {copied_dlls} DLLs from Conda Library/bin.")

print("2. Removing conflicting icu DLLs from Conda...")
removed_icu = 0
for dll_file in glob.glob(os.path.join(internal_dir, "icu*.dll")):
    try:
        os.remove(dll_file)
        removed_icu += 1
    except Exception as e:
        print(f"Failed to remove {dll_file}: {e}")
print(f"Removed {removed_icu} conflicting icu*.dll files.")

print("3. Copying PySide6 and shiboken6 manually...")
for pkg in ["shiboken6", "PySide6"]:
    src = os.path.join(conda_env, "Lib", "site-packages", pkg)
    dst = os.path.join(internal_dir, pkg)
    if os.path.isdir(src):
        print(f"Syncing package {pkg}...")
        try:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"Package {pkg} synced successfully.")
        except Exception as e:
            print(f"Failed to sync {pkg}: {e}")

print("4. Removing duplicate MSVC runtimes from PySide6...")
pyside6_dir = os.path.join(internal_dir, "PySide6")
removed_msvc = 0
if os.path.isdir(pyside6_dir):
    for pattern in ["msvcp140*.dll", "vc*.dll"]:
        for dll_file in glob.glob(os.path.join(pyside6_dir, pattern)):
            try:
                os.remove(dll_file)
                removed_msvc += 1
            except Exception as e:
                print(f"Failed to remove duplicate {dll_file}: {e}")
print(f"Removed {removed_msvc} duplicate MSVC DLLs from PySide6 folder.")

print("5. Copying PyOpenGL...")
opengl_src = os.path.join(project_dir, ".venv", "lib", "site-packages", "OpenGL")
opengl_dst = os.path.join(internal_dir, "OpenGL")
if os.path.isdir(opengl_src):
    try:
        shutil.copytree(opengl_src, opengl_dst, dirs_exist_ok=True)
        print("PyOpenGL copied from .venv.")
    except Exception as e:
        print(f"Failed to copy PyOpenGL: {e}")
else:
    opengl_src_conda = os.path.join(conda_env, "Lib", "site-packages", "OpenGL")
    if os.path.isdir(opengl_src_conda):
        try:
            shutil.copytree(opengl_src_conda, opengl_dst, dirs_exist_ok=True)
            print("PyOpenGL copied from Conda env.")
        except Exception as e:
            print(f"Failed to copy PyOpenGL from Conda: {e}")

print("6. Copying h5netcdf...")
h5_src = os.path.join(conda_env, "Lib", "site-packages", "h5netcdf")
h5_dst = os.path.join(internal_dir, "h5netcdf")
if os.path.isdir(h5_src):
    try:
        shutil.copytree(h5_src, h5_dst, dirs_exist_ok=True)
        print("h5netcdf copied successfully.")
    except Exception as e:
        print(f"Failed to copy h5netcdf: {e}")

print("7. Copying PROJ data files...")
proj_data_src = os.path.join(conda_env, "Library", "share", "proj")
proj_data_dst = os.path.join(internal_dir, "pyproj", "proj_dir", "share", "proj")
if os.path.isdir(proj_data_src):
    os.makedirs(proj_data_dst, exist_ok=True)
    try:
        shutil.copytree(proj_data_src, proj_data_dst, dirs_exist_ok=True)
        print(f"PROJ data copied successfully ({len(os.listdir(proj_data_dst))} files).")
    except Exception as e:
        print(f"Failed to copy PROJ data: {e}")

print("8. Copying GDAL data files...")
gdal_data_src = os.path.join(conda_env, "Library", "share", "gdal")
gdal_data_dst = os.path.join(internal_dir, "Library", "share", "gdal")
if os.path.isdir(gdal_data_src):
    os.makedirs(gdal_data_dst, exist_ok=True)
    try:
        shutil.copytree(gdal_data_src, gdal_data_dst, dirs_exist_ok=True)
        print(f"GDAL data copied successfully ({len(os.listdir(gdal_data_dst))} files).")
    except Exception as e:
        print(f"Failed to copy GDAL data: {e}")

print("9. Copying pyigrf...")
pyigrf_src = os.path.join(conda_env, "Lib", "site-packages", "pyigrf")
pyigrf_dst = os.path.join(internal_dir, "pyigrf")
if os.path.isdir(pyigrf_src):
    try:
        shutil.copytree(pyigrf_src, pyigrf_dst, dirs_exist_ok=True)
        print("pyigrf copied successfully.")
    except Exception as e:
        print(f"Failed to copy pyigrf: {e}")

print("Patch complete successfully!")
