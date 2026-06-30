# Force adding the conda environment DLLs to the PATH to avoid missing DLL errors when running in a conda environment.
# Especially important for Windows where DLL loading can be tricky. This is done before importing any module that might load DLLs.
# Solves case of importing _imgaging from PIL when GDAL is already installed in windows
import os

conda_env_path = os.environ.get("CONDA_PREFIX")
if conda_env_path:
    dll_path_conda = os.path.join(conda_env_path, "Library", "bin")
    current_path_env = os.environ.get("PATH", "")
    os.environ["PATH"] = current_path_env + os.path.pathsep + dll_path_conda
