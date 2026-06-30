"""
For a sounder type compute a mode information object corresponding to a combination of several sounder mode


"""

import dataclasses
from dataclasses import dataclass

import sonar_netcdf.vendor_types as km_types
from dataclasses_json import dataclass_json

from pyat.sonarscope.model.sounder_mode.sounder_modes import KmPulseForm, KongsbergKey
from pyat.utils.string_utils import upper_camel_case


@dataclass_json
@dataclass(frozen=True, unsafe_hash=False)
class KeyModeAllGeneric(KongsbergKey):
    ping_mode: int | None = None
    pulse_form: int | None = None
    swath_mode: int | None = None
    swath_index: int | None = None
    sector_count: int | None = None
    center_frequency: tuple | None = None

    def short_name(self) -> str:
        short_name = "All"
        if self.ping_mode in [mode.value for mode in km_types.KmPingMode]:
            short_name += f"_{upper_camel_case(km_types.KmPingMode(self.ping_mode).name)}"
        if self.pulse_form in [mode.value for mode in KmPulseForm]:
            short_name += f"_{upper_camel_case(KmPulseForm(self.pulse_form).name)}"
        if self.sector_count is not None:
            short_name += f"_{self.sector_count}Sector"
        if self.swath_mode is not None:
            if self.swath_mode == 0:
                short_name += "_SingleSwath"
            elif self.swath_mode > 0:
                short_name += "_DualSwath"
                if self.swath_index is not None:
                    short_name += f"({self.swath_index})"

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

    def get_center_frequency(self) -> tuple | None:
        return self.center_frequency

    def __str__(self):
        return self.short_name()

    # override __eq__ to compare selection of attributes
    def __eq__(self, other):
        if isinstance(other, KeyModeAllGeneric):
            if self.center_frequency is None or other.center_frequency is None:
                return (
                    self.ping_mode == other.ping_mode
                    and self.pulse_form == other.pulse_form
                    and self.swath_mode == other.swath_mode
                    and self.swath_index == other.swath_index
                    and self.sector_count == other.sector_count
                )
            else:
                return (
                    self.ping_mode == other.ping_mode
                    and self.pulse_form == other.pulse_form
                    and self.swath_mode == other.swath_mode
                    and self.sector_count == other.sector_count
                    and self.center_frequency == other.center_frequency
                )
        return False

    # override __hash__ to hash the selection of attributes
    def __hash__(self):
        if self.center_frequency is None:
            return hash(
                (
                    self.ping_mode,
                    self.pulse_form,
                    self.swath_mode,
                    self.swath_index,
                    self.sector_count,
                )
            )
        else:
            return hash(
                (
                    self.ping_mode,
                    self.pulse_form,
                    self.swath_mode,
                    self.sector_count,
                    self.center_frequency,
                )
            )
