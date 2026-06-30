"""
For a sounder type compute a mode information object corresponding to a combination of several sounder mode


"""

import dataclasses
from dataclasses import dataclass

import sonar_netcdf.vendor_types as km_types
from dataclasses_json import dataclass_json

from pyat.sonarscope.model.sounder_mode.sounder_modes import KongsbergKey
from pyat.utils.string_utils import upper_camel_case


@dataclass_json
@dataclass(frozen=False, init=False, unsafe_hash=False)
class KeyModeAllEM2040(KongsbergKey):
    frequency_mode: float | None
    swath_mode: int | None
    swath_index: int | None
    sector_count: int | None
    center_frequency: tuple | None = None
    scanning_mode: bool | None
    pulse_length_mode: int | None
    pulse_form: int | None

    def __init__(
        self,
        frequency_mode: float | None = None,
        swath_mode: int | None = None,
        swath_index: int | None = None,
        sector_count: int | None = None,
        center_frequency: tuple | None = None,
        scanning_mode: bool | None = None,
        pulse_length_mode: int | None = None,
        pulse_form: int | None = None,
    ):
        self.frequency_mode = float(frequency_mode) if frequency_mode is not None else None
        self.swath_mode = int(swath_mode) if swath_mode is not None else None
        self.swath_index = int(swath_index) if swath_index is not None else None
        self.sector_count = int(sector_count) if sector_count is not None else None
        self.center_frequency = tuple(center_frequency) if center_frequency is not None else None
        self.scanning_mode = bool(scanning_mode) if scanning_mode is not None else None
        self.pulse_length_mode = int(pulse_length_mode) if pulse_length_mode is not None else None
        self.pulse_form = int(pulse_form) if pulse_form is not None else None

    def short_name(self):
        short_name = "All2040"
        if self.frequency_mode is not None and self.frequency_mode > 0:
            short_name += f"_{float(self.frequency_mode/1000)}kHz"
        if self.pulse_length_mode in [mode.value for mode in km_types.KmPulseLengthMode]:
            short_name += f"_{upper_camel_case(km_types.KmPulseLengthMode(self.pulse_length_mode).name)}"
        if self.sector_count is not None:
            short_name += f"_{self.sector_count}Sector"
        if self.scanning_mode is True:
            short_name += "_Scan"
        elif self.swath_mode is not None:
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
    def mode_to_json(self):
        return self.to_json()

    def get_tx_beam_count(self):
        return self.sector_count or 1

    def __str__(self):
        return self.short_name()

    def __eq__(self, other):
        if isinstance(other, KeyModeAllEM2040):
            if self.center_frequency is None or other.center_frequency is None:
                return (
                    self.frequency_mode == other.frequency_mode
                    and self.swath_mode == other.swath_mode
                    and self.swath_index == other.swath_index
                    and self.sector_count == other.sector_count
                    and self.scanning_mode == other.scanning_mode
                    and self.pulse_length_mode == other.pulse_length_mode
                    and self.pulse_form == other.pulse_form
                )
            else:
                return (
                    self.frequency_mode == other.frequency_mode
                    and self.swath_mode == other.swath_mode
                    and self.sector_count == other.sector_count
                    and self.scanning_mode == other.scanning_mode
                    and self.pulse_length_mode == other.pulse_length_mode
                    and self.pulse_form == other.pulse_form
                    and self.center_frequency == other.center_frequency
                )
        return False

    # override __hash__ to hash the selection of attributes
    def __hash__(self):
        if self.center_frequency is None:
            return hash(
                (
                    self.frequency_mode,
                    self.swath_mode,
                    self.swath_index,
                    self.sector_count,
                    self.scanning_mode,
                    self.pulse_length_mode,
                    self.pulse_form,
                )
            )
        else:
            return hash(
                (
                    self.frequency_mode,
                    self.swath_mode,
                    self.sector_count,
                    self.scanning_mode,
                    self.pulse_length_mode,
                    self.pulse_form,
                    self.center_frequency,
                )
            )
