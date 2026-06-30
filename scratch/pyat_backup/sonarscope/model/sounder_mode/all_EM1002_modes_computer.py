from typing import Dict, List, Tuple

import numpy as np

from pyat.sonarscope.model.constants import VariableKeys as Key
from pyat.sonarscope.model.signal.ping_signal import PingSignal
from pyat.sonarscope.model.sounder_mode.all_EM1002_mode import KeyModeAllEM1002
from pyat.sonarscope.model.sounder_mode.sounder_modes import KeyMode
from pyat.sonarscope.model.sounder_mode.sounder_modes_computer import ModeComputer
from pyat.xsf import xsf_driver
from pyat.xsf.xsf_driver import XsfDriver


class ModeComputerAllEM1002(ModeComputer):
    """mode computer for EM1002 kongsberg sounder from all"""

    def compute(self, input_files: List[str]) -> Tuple[Dict[KeyMode, int], Dict[str, np.ndarray]]:
        total_keys = {}  # dictionary containing KeyMode and their id
        values_dict = {}  # dictionary containing values for each file
        for f in input_files:
            with xsf_driver.open_xsf(file_path=f) as xsf_file:
                total_keys, values = self.compute_xsf(xsf=xsf_file, global_keys=total_keys)
                values_dict[f] = values

        return total_keys, values_dict

    def compute_keys_values(
        self,
        ping_mode: np.ndarray,
        global_keys: Dict[KeyMode, int],
    ) -> Tuple[Dict[KeyMode, int], np.ndarray]:
        """
        Parse all the parameter arrays and retrieve a set of exclusive KeyModeEM2040 of all combination seen in the file
        Returns : a tuple containing a dictionary of the KeyMode values and their id, and a 1D array of modes

        """
        reference_shape = ping_mode.shape
        values = np.full(shape=reference_shape, fill_value=-1)
        # iterate over all frequency, mode, etc arrays
        i = 0
        for pmode in np.nditer((ping_mode)):
            # For each combination create a key
            pmode = None if np.isnan(pmode) else int(pmode)

            key = KeyModeAllEM1002(ping_mode=pmode)
            # the use of a set will retain unique keys

            if key not in global_keys:
                next_index = len(global_keys)
                global_keys[key] = next_index

            values[i] = global_keys[key]
            i = i + 1
        return global_keys, values

    def compute_xsf(self, xsf: XsfDriver, global_keys: Dict[KeyMode, int]) -> Tuple[Dict[KeyMode, int], np.ndarray]:
        """Compute the list of available mode for a given file"""

        xsf.open()

        # Read data as 1D values
        model = PingSignal(xsf_dataset=xsf)
        model.read([Key.PING_MODE])

        # retrieve values
        ping_mode = model.xr_dataset[Key.PING_MODE].to_numpy()

        # compute signal modes
        return self.compute_keys_values(
            ping_mode=ping_mode,
            global_keys=global_keys,
        )
