"""
For a sounder type compute a mode information object corresponding to a combination of several sounder mode
"""

from dataclasses import dataclass

from dataclasses_json import dataclass_json

from pyat.sonarscope.model.sounder_mode.sounder_modes import KeyMode


@dataclass_json
@dataclass(frozen=False, init=False, unsafe_hash=True)
class KeyModeCommon(KeyMode):
    def short_name(self) -> str:
        return "Common"

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
        return None

    def __str__(self):
        return self.short_name()
