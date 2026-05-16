"""tide models files configuration"""

import os
from pathlib import Path

# version release for test path
release = "1.0.0"
destination_dir = os.getenv("TIDE_DATASETS", "C:/tide_datasets")
model_dir = f"{destination_dir}/{release}/tide"


def get_tide_datasets_path() -> Path:
    """
    return TIDE_DATASETS folder if specified in environment variable.
    """
    if os.path.isdir(model_dir):
        print(f"Directory {model_dir} does not exists")

    return Path(model_dir)
