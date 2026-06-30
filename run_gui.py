r"""pyat GUI launcher.

Usage:
    python run_gui.py
"""
import sys, os

# Disable Chromium sandbox to fix map rendering issues (especially with subprocess restarts)
os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"

# Setup DLL paths on Windows first
if sys.platform == 'win32':
    if getattr(sys, 'frozen', False):
        # Packaged/frozen application: Clean PATH to prevent loading conflicting DLLs from Anaconda
        path_list = os.environ.get("PATH", "").split(os.pathsep)
        clean_path = [p for p in path_list if not ('anaconda' in p.lower() or 'conda' in p.lower())]
        os.environ["PATH"] = os.pathsep.join(clean_path)
        
        # Add packaged DLL folders to Windows DLL search paths
        internal_dir = os.path.dirname(os.path.abspath(__file__)) if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "_internal" else os.path.join(os.path.dirname(os.path.abspath(__file__)), "_internal")
        pyside_dir = os.path.join(internal_dir, "PySide6")
        if sys.version_info >= (3, 8):
            try:
                os.add_dll_directory(internal_dir)
                os.add_dll_directory(pyside_dir)
            except Exception:
                pass
    else:
        # Development environment: Setup Anaconda environment DLL paths
        py_bin = os.path.dirname(sys.executable)
        env_root = os.path.dirname(py_bin) if os.path.basename(py_bin).lower() == "scripts" else py_bin
        conda_lib_bin = os.path.join(env_root, "Library", "bin")
        if os.path.isdir(conda_lib_bin):
            path_list = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
            first_path = os.path.normpath(path_list[0]).lower() if path_list else ""
            conda_lib_bin_norm = os.path.normpath(conda_lib_bin).lower()

            if first_path != conda_lib_bin_norm and not os.environ.get("PYAT_ENV_RESTARTED"):
                import subprocess
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

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

# Ensure project root and src are on path
for p in [ROOT, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(ROOT)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    if len(sys.argv) >= 3 and sys.argv[1] == "-m" and sys.argv[2] == "pyat":
        from pyat.__main__ import __launch__
        if len(sys.argv) > 3:
            sys.exit(__launch__(sys.argv[3]))
        else:
            print("Missing configuration file path.")
            sys.exit(1)
    else:
        from gui.app import main
        main()
