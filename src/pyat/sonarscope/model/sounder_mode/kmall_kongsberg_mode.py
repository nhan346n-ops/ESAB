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
@dataclass(frozen=False, init=False, unsafe_hash=True)
class KeyModeKmallGeneric(KongsbergKey):
    frequency_mode: float | None
    depth_mode: int | None
    pulse_form: int | None
    sector_count: int | None
    swath_count: int | None
    swath_index: int | None

    def __init__(
        self,
        frequency_mode: float | None = None,
        depth_mode: int | None = None,
        pulse_form: int | None = None,
        sector_count: int | None = None,
        swath_count: int | None = None,
        swath_index: int | None = None,
    ):
        self.frequency_mode = float(frequency_mode) if frequency_mode is not None else None
        self.depth_mode = int(depth_mode) if depth_mode is not None else None
        self.pulse_form = int(pulse_form) if pulse_form is not None else None
        self.sector_count = int(sector_count) if sector_count is not None else None
        self.swath_count = int(swath_count) if swath_count is not None else None
        self.swath_index = int(swath_index) if swath_index is not None else None

    def short_name(self) -> str:
        short_name = "Kmall"

        # EMdgmMRZ_pingInfo_def Struct Reference : frequencyMode_Hz
        if self.frequency_mode is not None:
            if self.frequency_mode > 100:
                short_name += f"_{int(self.frequency_mode/1000)}kHz"
            elif self.frequency_mode == 0:
                short_name += "_40-100kHz"
            elif self.frequency_mode == 1:
                short_name += "_50-100kHz"
            elif self.frequency_mode == 2:
                short_name += "_70-100kHz"
            elif self.frequency_mode == 3:
                short_name += "_50kHz"
            elif self.frequency_mode == 4:
                short_name += "_40kHz"

        if self.depth_mode in [mode.value for mode in km_types.KmPingMode]:
            short_name += f"_{upper_camel_case(km_types.KmPingMode(self.depth_mode).name)}"
        if self.pulse_form in [mode.value for mode in KmPulseForm]:
            short_name += f"_{upper_camel_case(KmPulseForm(self.pulse_form).name)}"
        if self.sector_count is not None:
            short_name += f"_{self.sector_count}Sector"
        if self.swath_count is not None:
            short_name += f"_{self.swath_count}Swath"
            if self.swath_count > 1 and self.swath_index is not None:
                short_name += f"({self.swath_index})"

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
        return self.sector_count or 1

    def __str__(self):
        return self.short_name()
