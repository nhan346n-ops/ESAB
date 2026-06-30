from typing import Dict, Optional

from pyat.sonarscope.bs_correction.file_data import FileDataStore
from pyat.sonarscope.bs_correction.mean_bs_model import MeanBSModel
from pyat.sonarscope.model.sounder_mode.sounder_modes import KeyMode


class GlobalDataModel:
    """A global container which store per data used by bs_correction"""

    def __init__(self, file_data: Dict[str, FileDataStore] | None = None):
        if file_data is None:
            file_data = {}
        self.file_data = file_data
        # Netcdf incidence curve file path
        self.incidence_curve_file: Optional[str] = None
        # Netcdf transmission curve file path
        self.transmission_curve_file: Optional[str] = None
        # Mean bs model
        self.model_file : Optional[MeanBSModel] = None
        # Keymode dictionary
        self.keymode_dict : Dict[KeyMode, int] = {}
