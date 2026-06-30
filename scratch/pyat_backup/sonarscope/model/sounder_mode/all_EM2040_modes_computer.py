from typing import List, Tuple, Dict

import numpy as np

from pyat.sonarscope.model.constants import VariableKeys as Key
from pyat.sonarscope.model.signal.ping_signal import PingSignal
from pyat.sonarscope.model.sounder_mode.all_EM2040_mode import KeyModeAllEM2040
from pyat.sonarscope.model.sounder_mode.sounder_modes import KeyMode
from pyat.sonarscope.model.sounder_mode.sounder_modes_computer import ModeComputer
from pyat.utils.exceptions.exception_list import BadParameter
from pyat.xsf import xsf_driver
from pyat.xsf.xsf_driver import XsfDriver


class ModeComputerAllEM2040(ModeComputer):
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
        frequency_array: np.ndarray,
        swath_mode: np.ndarray,
        swath_index: np.ndarray,
        sector_count: np.ndarray,
        center_frequency: np.ndarray,
        scanning_mode: np.ndarray,
        pulse_length_mode: np.ndarray,
        pulse_form: np.ndarray,
        global_keys: Dict[KeyMode, int],
    ) -> Tuple[Dict[KeyMode, int], np.ndarray]:
        """
        Parse all the parameter arrays and retrieve a set of exclusive KeyModeEM2040 of all combination seen in the file
        Returns : a tuple containing a dictionary of the KeyMode values and their id, and a 1D array of modes

        """
        # pylint: disable=too-many-boolean-expressions
        reference_shape = frequency_array.shape
        if (
            reference_shape != swath_mode.shape
            or reference_shape != swath_index.shape
            or reference_shape != sector_count.shape
            or reference_shape != scanning_mode.shape
            or reference_shape != pulse_length_mode.shape
            or reference_shape != pulse_form.shape
            or reference_shape[0] != center_frequency.shape[0]
        ):
            raise BadParameter(
                f"Compute backscatter key mode function does not support arrays with different shape, coding error in {KeyModeAllEM2040.__name__}"
            )
        values = np.full(shape=reference_shape, fill_value=-1)
        # iterate over all frequency, mode, etc arrays
        for i in range(reference_shape[0]):
            fq = frequency_array[i]
            smode = swath_mode[i]
            sindex = swath_index[i]
            tx_count = sector_count[i]
            cfreq = center_frequency[i]

            pform = pulse_form[i]
            scan = scanning_mode[i]
            pl = pulse_length_mode[i]
            # For each combination create a key
            fq = None if np.isnan(fq) else float(fq)
            # if frequency mode is 400kHz, the swath index is inverted (multiping_sequence assume center_frequancies are in ascending order)

            if np.isnan(sindex):
                sindex = None
            elif fq == 400000:
                sindex = int(2 - sindex)
            else:
                sindex = int(sindex)

            if np.isnan(smode):
                smode = None
            elif smode > 0:
                # merge dual swath mode fix and dynamic into a unique mode
                smode = 1
            else:
                smode = int(smode)
            tx_count = None if np.isnan(tx_count) else int(tx_count)
            scan = bool(scan)
            pl = None if np.isnan(pl) else int(pl)
            pform = None if np.isnan(pform) else int(pform)
            if tx_count is not None and tx_count > 0:
                cfreq = tuple(int(x) for x in cfreq[:tx_count])
            else:
                cfreq = None
            key = KeyModeAllEM2040(
                frequency_mode=fq,
                swath_mode=smode,
                swath_index=sindex,
                sector_count=tx_count,
                center_frequency=cfreq,
                scanning_mode=scan,
                pulse_length_mode=pl,
                pulse_form=pform,
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
                Key.FREQUENCY_MODE,
                Key.TX_SECTOR_COUNT,
                Key.PULSE_LENGTH_MODE,
                Key.PULSE_FORM,
                Key.DUAL_SWATH_MODE,
                Key.MULTIPING_SEQUENCE,
            ]
        )

        # retrieve values
        fq_mode = model.xr_dataset[Key.FREQUENCY_MODE].to_numpy()
        tx_sector_count = model.xr_dataset[Key.TX_SECTOR_COUNT].to_numpy()
        pulse_len_mode = model.xr_dataset[Key.PULSE_LENGTH_MODE].to_numpy()
        pulse_form = model.xr_dataset[Key.PULSE_FORM].to_numpy()
        dual_mode = model.xr_dataset[Key.DUAL_SWATH_MODE].to_numpy()
        multiping_sequence = model.xr_dataset[Key.MULTIPING_SEQUENCE].to_numpy()
        scanning_mode = np.full(shape=dual_mode.shape, fill_value=False)
        center_frequency = xsf.read_multiping_center_frequency()

        # compute signal modes
        return self.compute_keys_values(
            frequency_array=fq_mode,
            swath_mode=dual_mode,
            swath_index=multiping_sequence,
            sector_count=tx_sector_count,
            center_frequency=center_frequency,
            scanning_mode=scanning_mode,
            pulse_length_mode=pulse_len_mode,
            pulse_form=pulse_form,
            global_keys=global_keys,
        )
