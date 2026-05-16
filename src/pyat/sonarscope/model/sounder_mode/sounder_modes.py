"""
For a sounder type compute a mode information object corresponding to a combination of several sounder mode
"""
from abc import ABC, abstractmethod
from enum import Enum

class KeyMode(ABC):
    """Abstract class of a mode combination key, the way the key is constructed depends on the sounder"""

    @abstractmethod
    def is_valid(self) -> bool:
        """indicate if the keyMode match valid data"""

    @abstractmethod
    def mode_to_json(self) -> str:
        """Serialize to json"""

    @abstractmethod
    def short_name(self) -> str:
        """
        Return a descriptive name of the mode
        Args:
           ignore_mutiping: if True, return the same name in case of multiswath
        """

    @classmethod
    @abstractmethod
    def mode_from_json(cls, text):
        """deserialize from json"""

    @abstractmethod
    def get_tx_beam_count(self) -> int:
        """return the number of tx beam for this mode, this is used to have separate statistics
        for each tx sector for kongsberg sounders
        """



class KongsbergKey(KeyMode):
    pass

class ResonKey(KeyMode):
    pass


class KmPulseForm(Enum):
    """Enum for .all and .kmall pulse form"""
    UNKNOWN = -1
    CW = 0
    MIXED = 1
    FM = 2
    
class ResonPulseForm(Enum):
    """Enum for .s7k pulse form"""
    UNKNOWN = -1
    CW = 0
    FM = 1
    