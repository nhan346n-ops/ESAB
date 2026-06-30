from pyat.sonarscope.model.sounder_mode.all_EM1002_mode import KeyModeAllEM1002
from pyat.sonarscope.model.sounder_mode.all_EM1002_modes_computer import ModeComputerAllEM1002
from pyat.sonarscope.model.sounder_mode.all_EM2040_mode import KeyModeAllEM2040
from pyat.sonarscope.model.sounder_mode.all_EM2040_modes_computer import ModeComputerAllEM2040
from pyat.sonarscope.model.sounder_mode.all_kongsberg_mode import KeyModeAllGeneric
from pyat.sonarscope.model.sounder_mode.all_kongsberg_modes_computer import ModeComputerAllGeneric
from pyat.sonarscope.model.sounder_mode.calibrated_mode import KeyModeCalibrated
from pyat.sonarscope.model.sounder_mode.common_mode import KeyModeCommon
from pyat.sonarscope.model.sounder_mode.kmall_kongsberg_mode import KeyModeKmallGeneric
from pyat.sonarscope.model.sounder_mode.kmall_kongsberg_modes_computer import ModeComputerKmallGeneric
from pyat.sonarscope.model.sounder_lib import SounderType
from pyat.sonarscope.model.sounder_mode.s7k_reson_mode import KeyModeResonGeneric
from pyat.sonarscope.model.sounder_mode.s7k_reson_modes_computer import ModeComputerResonGeneric
from pyat.sonarscope.model.sounder_mode.sounder_modes import KeyMode
from pyat.sonarscope.model.sounder_mode.sounder_modes_computer import ModeComputer


class ModeComputerFactory:
    """Factory returning a mode computer for the given sounder_type (given as string)"""

    @staticmethod
    def create_mode_computer(sounder_type: str) -> ModeComputer:
        # create a mode computer for the given sounder type
        sounder_type = SounderType.from_type(sounder_type)
        if SounderType.EM1002_ALL == sounder_type:
            mode_computer = ModeComputerAllEM1002()
        elif SounderType.EM3002_ALL == sounder_type:
            mode_computer = ModeComputerAllGeneric()
        elif SounderType.EM2040_ALL == sounder_type:
            mode_computer = ModeComputerAllEM2040()
        elif SounderType.EM120_ALL == sounder_type:
            mode_computer = ModeComputerAllGeneric()
        elif SounderType.EM122_ALL == sounder_type:
            mode_computer = ModeComputerAllGeneric()
        elif SounderType.EM302_ALL == sounder_type:
            mode_computer = ModeComputerAllGeneric()
        elif SounderType.EM710_ALL == sounder_type:
            mode_computer = ModeComputerAllGeneric()
        elif SounderType.ME70_ALL == sounder_type:
            mode_computer = ModeComputerAllGeneric()
        elif SounderType.EM2040_KMALL == sounder_type:
            mode_computer = ModeComputerKmallGeneric()
        elif SounderType.EM124_KMALL == sounder_type:
            mode_computer = ModeComputerKmallGeneric()
        elif SounderType.EM304_KMALL == sounder_type:
            mode_computer = ModeComputerKmallGeneric()
        elif SounderType.EM712_KMALL == sounder_type:
            mode_computer = ModeComputerKmallGeneric()
        elif SounderType.SEABAT7150_S7K == sounder_type:
            mode_computer = ModeComputerResonGeneric()
        elif SounderType.SEABAT7125_S7K == sounder_type:
            mode_computer = ModeComputerResonGeneric()
        elif SounderType.SEABAT7111_S7K == sounder_type:
            mode_computer = ModeComputerResonGeneric()
        else:
            raise NotImplementedError(
                f"Sounder {sounder_type} not supported yet, coding error in {ModeComputerFactory.__name__}"
            )
        return mode_computer

    @staticmethod
    def key_mode_from_json(sounder_type: str, json_text: str) -> KeyMode:
        sounder_type = SounderType.from_type(sounder_type)
        if json_text == "{}":
            key_mode = KeyModeCommon()
        elif SounderType.CALIBRATED == sounder_type:
            key_mode = KeyModeCalibrated.mode_from_json(json_text)
        elif SounderType.EM1002_ALL == sounder_type:
            key_mode = KeyModeAllEM1002.mode_from_json(json_text)
        elif SounderType.EM3002_ALL == sounder_type:
            key_mode = KeyModeAllGeneric.mode_from_json(json_text)
        elif SounderType.EM2040_ALL == sounder_type:
            key_mode = KeyModeAllEM2040.mode_from_json(json_text)
        elif SounderType.EM120_ALL == sounder_type:
            key_mode = KeyModeAllGeneric.mode_from_json(json_text)
        elif SounderType.EM122_ALL == sounder_type:
            key_mode = KeyModeAllGeneric.mode_from_json(json_text)
        elif SounderType.EM302_ALL == sounder_type:
            key_mode = KeyModeAllGeneric.mode_from_json(json_text)
        elif SounderType.EM710_ALL == sounder_type:
            key_mode = KeyModeAllGeneric.mode_from_json(json_text)
        elif SounderType.ME70_ALL == sounder_type:
            key_mode = KeyModeAllGeneric.mode_from_json(json_text)
        elif SounderType.EM2040_KMALL == sounder_type:
            key_mode = KeyModeKmallGeneric.mode_from_json(json_text)
        elif SounderType.EM124_KMALL == sounder_type:
            key_mode = KeyModeKmallGeneric.mode_from_json(json_text)
        elif SounderType.EM304_KMALL == sounder_type:
            key_mode = KeyModeKmallGeneric.mode_from_json(json_text)
        elif SounderType.EM712_KMALL == sounder_type:
            key_mode = KeyModeKmallGeneric.mode_from_json(json_text)
        elif SounderType.SEABAT7150_S7K == sounder_type:
            key_mode = KeyModeResonGeneric.mode_from_json(json_text)
        elif SounderType.SEABAT7125_S7K == sounder_type:
            key_mode = KeyModeResonGeneric.mode_from_json(json_text)
        elif SounderType.SEABAT7111_S7K == sounder_type:
            key_mode = KeyModeResonGeneric.mode_from_json(json_text)
        else:
            raise NotImplementedError(
                f"Sounder {sounder_type} not supported yet, coding error in {ModeComputerFactory.__name__}"
            )
        return key_mode
