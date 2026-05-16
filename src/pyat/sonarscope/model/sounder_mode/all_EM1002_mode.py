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
@dataclass(frozen=False, init=False, unsafe_hash=True)
class KeyModeAllEM1002(KongsbergKey):
    ping_mode: int | None = None

    def __init__(
        self,
        ping_mode: int | None = None,
    ):
        self.ping_mode = int(ping_mode) if ping_mode is not None else None

    def short_name(self) -> str:
        short_name = "All1002"
        if self.ping_mode in [mode.value for mode in km_types.KmPingMode]:
            short_name += f"_{upper_camel_case(km_types.KmPingMode(self.ping_mode).name)}"

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

    def __str__(self):
        return self.short_name()
