from typing import Dict, List, Tuple

import numpy as np

from pyat.sonarscope.model.constants import VariableKeys as Key
from pyat.sonarscope.model.signal.ping_signal import PingSignal
from pyat.sonarscope.model.sounder_mode.all_kongsberg_mode import KeyModeAllGeneric
from pyat.sonarscope.model.sounder_mode.sounder_modes import KeyMode
from pyat.sonarscope.model.sounder_mode.sounder_modes_computer import ModeComputer
from pyat.utils.exceptions.exception_list import BadParameter
from pyat.xsf import xsf_driver
from pyat.xsf.xsf_driver import XsfDriver


class ModeComputerAllGeneric(ModeComputer):
    """mode computer for kongsberg sounder from all"""

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
        pulse_form: np.ndarray,
        swath_mode: np.ndarray,
        swath_index: np.ndarray,
        sector_count: np.ndarray,
        center_frequency: np.ndarray,
        global_keys: Dict[KeyMode, int],
    ) -> Tuple[Dict[KeyMode, int], np.ndarray]:
        """
        Parse all the parameter arrays and retrieve a set of exclusive KeyModeAllGeneric of all combination seen in the file
        Returns : a tuple containing a dictionary of the KeyMode values and their id, and a 1D array of modes

        """
        reference_shape = ping_mode.shape
        if (
            reference_shape != pulse_form.shape
            or reference_shape != swath_mode.shape
            or reference_shape != swath_index.shape
            or reference_shape != sector_count.shape
            or reference_shape[0] != center_frequency.shape[0]
        ):
            raise BadParameter(
                f"Compute backscatter key mode function does not support arrays with different shape, coding error in {KeyModeAllGeneric.__name__}"
            )
        values = np.full(shape=reference_shape, fill_value=-1)
        # iterate over all frequency, mode, etc arrays
        for i in range(reference_shape[0]):
            pmode = ping_mode[i]
            pform = pulse_form[i]
            smode = swath_mode[i]
            sindex = swath_index[i]
            tx_count = sector_count[i]
            cfreq = center_frequency[i]
            # For each combination create a key
            pmode = None if np.isnan(pmode) else int(pmode)
            pform = None if np.isnan(pform) else int(pform)
            if np.isnan(smode):
                smode = None
            elif smode > 0:
                # merge dual swath mode fix and dynamic into a unique mode
                smode = 1
            else:
                smode = int(smode)
            sindex = None if np.isnan(sindex) else int(sindex)
            tx_count = None if np.isnan(tx_count) else int(tx_count)
            # ensure converting to immutable tuple of int
            if tx_count is not None and tx_count > 0:
                cfreq = tuple(int(x) for x in cfreq[:tx_count])
            else:
                cfreq = None

            key = KeyModeAllGeneric(
                ping_mode=pmode,
                pulse_form=pform,
                swath_mode=smode,
                swath_index=sindex,
                sector_count=tx_count,
                center_frequency=cfreq,
            )
            # the use of a set will retain unique keys

            if key not in global_keys:
                next_index = len(global_keys)
                global_keys[key] = next_index

            values[i] = global_keys[key]
        return global_keys, values

    def compute_xsf(self, xsf: XsfDriver, global_keys: Dict[KeyMode, int]) -> Tuple[Dict[KeyMode, int], np.ndarray]:
        """Compute the list of available mode for a given file"""

        xsf.open()

        # Read data as 1D values
        model = PingSignal(xsf_dataset=xsf)
        model.read(
            [
                Key.PING_MODE,
                Key.PULSE_FORM,
                Key.TX_SECTOR_COUNT,
                Key.DUAL_SWATH_MODE,
                Key.MULTIPING_SEQUENCE,
            ]
        )

        # retrieve values
        ping_mode = model.xr_dataset[Key.PING_MODE].to_numpy()
        pulse_form = model.xr_dataset[Key.PULSE_FORM].to_numpy()
        tx_sector_count = model.xr_dataset[Key.TX_SECTOR_COUNT].to_numpy()
        dual_mode = model.xr_dataset[Key.DUAL_SWATH_MODE].to_numpy()
        multiping_sequence = model.xr_dataset[Key.MULTIPING_SEQUENCE].to_numpy()
        center_frequency = xsf.read_multiping_center_frequency()

        # remove data when ping_mode or dual_mode changes
        # dual mode change could append one ping later
        # ping_mode = np.where(ping_mode == 0, np.nan, ping_mode)
        # dual_mode = np.where(dual_mode == 0, np.nan, dual_mode)
        dual_mode_diff = np.diff(dual_mode, prepend=dual_mode[0])
        dual_mode = np.where(dual_mode_diff != 0, np.nan, dual_mode)

        # compute signal modes
        return self.compute_keys_values(
            ping_mode=ping_mode,
            pulse_form=pulse_form,
            swath_mode=dual_mode,
            swath_index=multiping_sequence,
            sector_count=tx_sector_count,
            center_frequency=center_frequency,
            global_keys=global_keys,
        )
