"""Application configuration and constants."""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"

def resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    import sys
    import os
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(str(PROJECT_ROOT), relative_path)

# GUI runtime directories
GUI_DIR = PROJECT_ROOT / "gui"
JOBS_DIR = GUI_DIR / ".pyat_gui" / "jobs"
LOGS_DIR = GUI_DIR / ".pyat_gui" / "logs"

# Backend config paths
GWS_CONF_PATH = SRC_PATH / "gws" / "conf" / "sonar" / "bs"

# Backend entry point
PYAT_ENTRY = ["python", "-m", "pyat"]

# XSF processing status attribute
ATT_PROCESSING_STATUS_BACKSCATTER_CORRECTION = "backscatterCorrection"

# Supported sounder types (from avg_backscatter_model.json choices)
SOUNDER_TYPES = [
    "AUTO", "EM1002_ALL", "EM2040_ALL", "EM120_ALL", "EM122_ALL",
    "EM302_ALL", "EM710_ALL", "ME70_ALL", "EM2040_KMALL", "EM124_KMALL",
    "EM304_KMALL", "EM712_KMALL", "7150_S7K", "7125_S7K", "7111_S7K"
]

# Map projections
PROJECTIONS = ["自动检测", "通用横轴墨卡托 (UTM)", "墨卡托 (Mercator)", "自定义 EPSG"]

# Grid resolutions (meters)
RESOLUTIONS = ["0.5", "1.0", "2.0", "5.0", "10.0", "自定义"]

# Gap filling methods
GAP_FILL_METHODS = ["None", "双线性 (Bilinear)", "反距离权重 (IDW)"]

# Integration methods (from configuration.py)
INTEGRATION_METHODS = ["MEAN", "MEDIAN"]

# Linear scale types (from configuration.py)
LINEAR_SCALES = ["AMPLITUDE", "ENERGY"]


def ensure_dirs() -> None:
    """Create runtime directories if they don't exist."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """Return ISO-like timestamp for file naming."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")
