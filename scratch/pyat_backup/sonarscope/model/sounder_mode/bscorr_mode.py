"""
For a sounder type compute a mode information object corresponding to a combination of several sounder mode
"""

import dataclasses
from dataclasses import dataclass

import sonar_netcdf.vendor_types as km_types
from dataclasses_json import dataclass_json

from pyat.sonarscope.model.sounder_mode.sounder_modes import KongsbergKey
from pyat.utils.string_utils import spaced_upper_camel_case


@dataclass_json
@dataclass(frozen=False, init=False, unsafe_hash=False)
class KeyModeBscorr(KongsbergKey):
    mode_id: int | None = None
    ping_mode: int | None = None  #
    swath_index: int | None = None  # 0: Single Swath, 1: Dual Swath 1, 2: Dual Swath 2
    sector_count: int | None = None

    def __init__(
        self,
        mode_id: int | None = None,
        ping_mode: int | None = None,
        swath_index: int | None = None,
        sector_count: int | None = None,
    ):
        self.mode_id = int(mode_id) if mode_id is not None else None
        self.ping_mode = int(ping_mode) if ping_mode is not None else None
        self.swath_index = int(swath_index) if swath_index is not None else None
        self.sector_count = int(sector_count) if sector_count is not None else None

    def short_name(self) -> str:
        short_name = ""
        if self.ping_mode in [mode.value for mode in km_types.KmPingMode]:
            short_name += f"{spaced_upper_camel_case(km_types.KmPingMode(self.ping_mode).name)}"
        if self.swath_index is not None:
            if self.swath_index == 0:
                short_name += " - SingleSwath"
            elif self.swath_index > 0:
                short_name += " - DualSwath"
                if self.swath_index is not None:
                    short_name += f" {self.swath_index}"
        if self.sector_count is not None:
            short_name += f" - {self.sector_count} sectors"
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
        return self.sector_count or 1

    def __str__(self):
        return self.short_name()

    # override __eq__ to compare selection of attributes
    def __eq__(self, other):
        if isinstance(other, KeyModeBscorr):
            return (
                self.ping_mode == other.ping_mode
                and self.swath_index == other.swath_index
                and self.sector_count == other.sector_count
            )
        return False

    # override __hash__ to hash the selection of attributes
    def __hash__(self):
        return hash(
            (
                self.ping_mode,
                self.swath_index,
                self.sector_count,
            )
        )
