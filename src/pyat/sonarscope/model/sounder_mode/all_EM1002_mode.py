"""
For a sounder type compute a mode information object corresponding to a combination of several sounder modes
"""

import dataclasses
from dataclasses import dataclass

import sonar_netcdf.vendor_types as km_types
from dataclasses_json import dataclass_json

from pyat.sonarscope.model.sounder_mode.sounder_modes import KongsbergKey
from pyat.utils.string_utils import upper_camel_case


@dataclass_json
@dataclass(frozen=True)
class KeyModeAllEM1002(KongsbergKey):
    ping_mode: int | None = None
    center_frequency: tuple | None = None  # only informative for this mode

    def short_name(self) -> str:
        short_name = "All1002"
        if self.ping_mode in [mode.value for mode in km_types.KmPingMode]:
            short_name += f"_{upper_camel_case(km_types.KmPingMode(self.ping_mode).name)}"
        if self.center_frequency is not None:
            short_name += f"_{float(self.center_frequency[0]/1000)}kHz"

        return short_name

    def is_valid(self):
        """indicate if the mode is a valid mode, ie has all values set"""
        return all(value is not None for value in dataclasses.asdict(self).values())

    # pylint: disable=no-member
    @classmethod
    def mode_from_json(cls, text: str):
        mode = cls.from_json(text)
        return mode

    # pylint: disable=no-member
    def mode_to_json(self) -> str:
        return self.to_json()

    def get_tx_beam_count(self):
        return 1

    def get_center_frequency(self) -> tuple | None:
        return self.center_frequency

    def __str__(self):
        return self.short_name()

    def __eq__(self, other):
        if isinstance(other, KeyModeAllEM1002):
            return self.ping_mode == other.ping_mode
        return False

    # override __hash__ to hash the selection of attributes
    def __hash__(self):
        return hash((self.ping_mode,))
