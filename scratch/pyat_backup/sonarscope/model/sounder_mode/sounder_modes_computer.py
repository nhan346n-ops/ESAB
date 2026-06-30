from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import numpy as np

from pyat.sonarscope.model.sounder_mode.sounder_modes import KeyMode
from pyat.utils import pyat_logger
from pyat.xsf.xsf_driver import XsfDriver


class ModeComputer(ABC):
    """Abstract mode computer"""

    def __init__(self):
        self.logger = pyat_logger.logging.getLogger(__name__)

    @abstractmethod
    def compute(self, input_files: List[str]) -> Tuple[Dict[KeyMode, int], Dict[str, np.ndarray]]:
        """
        Compute the list of available mode for a set of file
        return a list of mode for the set of file and a set of 1D data per file containing the id of the modes
        """

    @abstractmethod
    def compute_xsf(self, xsf: XsfDriver, global_keys: Dict[KeyMode, int]) -> Tuple[Dict[KeyMode, int], np.ndarray]:
        """Compute mode on the xsf file passed as parameter, allow to work on already opened file"""


def remove_invalid_key(key_dict: Dict[KeyMode, int]):
    return {k: v for k, v in key_dict.items() if k.is_valid()}


def get_valid_key_indices(key_dict: Dict[KeyMode, int]) -> np.ndarray:
    """
    retrieve invalid mode key index from dictionary got by ModeComputer.compute
    """
    return np.asarray([v for k, v in key_dict.items() if k.is_valid()])


def get_invalid_key_indices(key_dict: Dict[KeyMode, int]) -> np.ndarray:
    """
    retrieve invalid mode key index from dictionary got by ModeComputer.compute
    """
    return np.asarray([v for k, v in key_dict.items() if not k.is_valid()])
