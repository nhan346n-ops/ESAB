import os
import re
import sys
from pathlib import Path

if __name__ == "__main__":
    module_file = Path(os.path.abspath(__file__))
    current_folder = module_file.parent

    # Define pyat root folder (containing app/, pyat/, ...)
    pyat_folder = current_folder.parent
    if not pyat_folder.is_dir():
        print("Pyat folder not found")
        sys.exit(1)

    conf_folder = current_folder / "conf"
    if not conf_folder.is_dir():
        print("Configuration folder not found")
        sys.exit(1)

    index_file = conf_folder / "index.json"
    # Patch configuration file, set conda env
    with open(index_file, "r", encoding="utf8") as f:
        file_content = f.read()
        file_content = re.sub("@PYAT_ROOT@", pyat_folder.as_posix(), file_content)
        file_content = re.sub("@CONF_DIR@", conf_folder.as_posix(), file_content)
        print(file_content)
