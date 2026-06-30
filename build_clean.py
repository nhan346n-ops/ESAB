import os
import sys
import shutil
import subprocess

conda_env_python = r"C:\Users\GUO\AppData\Local\anaconda3\envs\env_pyat_runtime\python.exe"
conda_env_pip = r"C:\Users\GUO\AppData\Local\anaconda3\envs\env_pyat_runtime\Scripts\pip.exe"
project_dir = os.path.dirname(os.path.abspath(__file__))

print("=== STARTING CLEAN BUILD PROCESS ===")

# Step 1: Clean site-packages pollution
print("\n[Step 1] Uninstalling local pyat package from site-packages...")
subprocess.run([conda_env_pip, "uninstall", "-y", "pyat"], capture_output=True)
print("  Local pyat package uninstalled (pollution cleared).")

# Step 2: Remove old build and dist folders
print("\n[Step 2] Cleaning up previous build/dist folders...")
for folder in ["build", "dist"]:
    path = os.path.join(project_dir, folder)
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
            print(f"  Removed folder: {folder}")
        except Exception as e:
            print(f"  Warning: Could not remove folder {folder}: {e}")

# Step 3: Run PyInstaller
print("\n[Step 3] Running PyInstaller...")
spec_path = os.path.join(project_dir, "BS_Processor.spec")
p_build = subprocess.run([
    conda_env_python,
    "-m",
    "PyInstaller",
    spec_path,
    "--clean",
    "--noconfirm"
], cwd=project_dir)

if p_build.returncode != 0:
    print("  ERROR: PyInstaller build failed!")
    sys.exit(1)
print("  PyInstaller build completed successfully.")

# Step 4: Run patch_build.py
print("\n[Step 4] Running patch_build.py...")
patch_script = os.path.join(project_dir, "patch_build.py")
p_patch = subprocess.run([
    conda_env_python,
    patch_script
], cwd=project_dir)

if p_patch.returncode != 0:
    print("  ERROR: Post-processing patch failed!")
    sys.exit(1)
print("  Post-processing patch completed successfully.")

print("\n=== BUILD PROCESS SUCCESSFUL AND COMPLETE! ===")
print("The fully working executable is at: D:\\Ruanjian\\BS\\pyat-main\\dist\\BS_Processor\\BS_Processor.exe")
