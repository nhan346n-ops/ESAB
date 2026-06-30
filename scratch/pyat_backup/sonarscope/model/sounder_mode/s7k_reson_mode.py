"""
For a sounder type compute a mode information object corresponding to a combination of several sounder mode
"""

import dataclasses
from dataclasses import dataclass

from dataclasses_json import dataclass_json

from pyat.sonarscope.model.sounder_mode.sounder_modes import ResonKey, ResonPulseForm
from pyat.utils.string_utils import upper_camel_case


@dataclass_json
@dataclass(frozen=False, init=False, unsafe_hash=True)
class KeyModeResonGeneric(ResonKey):
    frequency_install: float | None
    frequency_mode: float | None
    pulse_form: int | None
    swath_count: int | None
    swath_index: int | None

    def __init__(
        self,
        frequency_install: float | None = None,
        frequency_mode: float | None = None,
        pulse_form: int | None = None,
        swath_count: int | None = None,
        swath_index: int | None = None,
    ):
        self.frequency_install = float(frequency_install) if frequency_install is not None else None
        self.frequency_mode = float(frequency_mode) if frequency_mode is not None else None
        self.pulse_form = int(pulse_form) if pulse_form is not None else None
        self.swath_count = int(swath_count) if swath_count is not None else None
        self.swath_index = int(swath_index) if swath_index is not None else None

    def short_name(self) -> str:
        short_name = "Reson"

        if self.frequency_install is not None:
            short_name += f"_{float(self.frequency_install/1000)}kHz"
        if self.pulse_form in [mode.value for mode in ResonPulseForm]:
            short_name += f"_{upper_camel_case(ResonPulseForm(self.pulse_form).name)}"
        if self.swath_count is not None:
            short_name += f"_{self.swath_count}Swath"
        if self.frequency_mode is not None and self.swath_index is not None:
            short_name += f"({float(self.frequency_mode/1000)}kHz_{self.swath_index})"

        return short_name

    def is_valid(self):
        """indicate if the mode is a valid mode, ie has all values set"""
        valid = True
        for value in dataclasses.asdict(self).values():
            if value is None:
                valid = False
        return valid

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

    def __str__(self):
        return self.short_name()
