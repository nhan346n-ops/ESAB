"""
For a sounder type compute a mode information object corresponding to a combination of several sounder mode
"""

from dataclasses import dataclass

from dataclasses_json import dataclass_json

from pyat.sonarscope.model.sounder_mode.sounder_modes import KeyMode


@dataclass_json
@dataclass(frozen=False, init=False, unsafe_hash=True)
class KeyModeCalibrated(KeyMode):
    frequency: float | None = None

    def __init__(
        self,
        frequency: float | None = None,
    ):
        self.frequency = float(frequency) if frequency is not None else None

    def short_name(self) -> str:
        short_name = "Calibrated"
        if self.frequency is not None and self.frequency > 0:
            short_name += f"_{float(self.frequency/1000)}kHz"

        return short_name

    def is_valid(self):
        return True

    # pylint: disable=no-member
    @classmethod
    def mode_from_json(cls, text: str):
        mode = cls.from_json(text)
        return mode

    # pylint: disable=no-member
    def mode_to_json(self):
        return self.to_json()

    def get_tx_beam_count(self):
        return 1

    def get_center_frequency(self) -> tuple | None:
        return (self.frequency,) if self.frequency is not None else None

    def __str__(self):
        return self.short_name()

    # override __eq__ to compare selection of attributes
    def __eq__(self, other):
        if isinstance(other, KeyModeCalibrated):
            return self.frequency == other.frequency
        return False
