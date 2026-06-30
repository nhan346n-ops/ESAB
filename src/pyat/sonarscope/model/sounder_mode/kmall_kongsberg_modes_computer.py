from typing import List, Tuple, Dict

import numpy as np

from pyat.sonarscope.model.sounder_mode.kmall_kongsberg_mode import KeyModeKmallGeneric
from pyat.sonarscope.model.constants import VariableKeys as Key
from pyat.sonarscope.model.signal.ping_signal import PingSignal
from pyat.sonarscope.model.sounder_mode.sounder_modes import KeyMode
from pyat.sonarscope.model.sounder_mode.sounder_modes_computer import ModeComputer
from pyat.utils.exceptions.exception_list import BadParameter
from pyat.xsf import xsf_driver
from pyat.xsf.xsf_driver import XsfDriver


class ModeComputerKmallGeneric(ModeComputer):
    """mode computer for generic sounder from kmall"""

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
        depth_mode: np.ndarray,
        pulse_form: np.ndarray,
        sector_count: np.ndarray,
        swath_count: np.ndarray,
        swath_index: np.ndarray,
        center_frequency: np.ndarray,
        global_keys: Dict[KeyMode, int],
    ) -> Tuple[Dict[KeyMode, int], np.ndarray]:
        """
        Parse all the parameter arrays and retrieve a set of exclusive KeyMode of all combination seen in the file
        Returns : a tuple containing a dictionary of the KeyMode values and their id, and a 1D array of modes

        """
        reference_shape = frequency_array.shape
        # ensure all per-ping arrays share the same shape and center_frequency matches in first dim
        if (
            not (
                reference_shape
                == depth_mode.shape
                == pulse_form.shape
                == sector_count.shape
                == swath_count.shape
                == swath_index.shape
            )
            or reference_shape[0] != center_frequency.shape[0]
        ):
            raise BadParameter(
                f"Compute backscatter key mode function does not support arrays with different shape, coding error in {KeyModeKmallGeneric.__name__}"
            )
        values = np.full(shape=reference_shape, fill_value=-1)
        # iterate over all frequency, mode, etc arrays
        for i in range(reference_shape[0]):
            fq = None if np.isnan(frequency_array[i]) else float(frequency_array[i])
            d_mode = None if np.isnan(depth_mode[i]) else int(depth_mode[i])
            p_form = None if np.isnan(pulse_form[i]) else int(pulse_form[i])
            tx_count = None if np.isnan(sector_count[i]) else int(sector_count[i])
            s_count = None if np.isnan(swath_count[i]) else int(swath_count[i])
            s_index = None if np.isnan(swath_index[i]) else int(swath_index[i])
            c_freq = center_frequency[i]
            # ensure converting to immutable tuple of int
            if tx_count is not None and tx_count > 0:
                c_freq = tuple(int(x) for x in c_freq[:tx_count])
            else:
                c_freq = None

            key = KeyModeKmallGeneric(
                frequency_mode=fq,
                depth_mode=d_mode,
                pulse_form=p_form,
                sector_count=tx_count,
                swath_count=s_count,
                swath_index=s_index,
                center_frequency=c_freq,
            )
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
        model.read(
            [
                Key.FREQUENCY_MODE,
                Key.PING_MODE,
                Key.PULSE_FORM,
                Key.TX_SECTOR_COUNT,
                Key.SWATH_PER_PING,
                Key.MULTIPING_SEQUENCE,
            ]
        )

        # retrieve values
        fq_mode = model.xr_dataset[Key.FREQUENCY_MODE].to_numpy()
        ping_mode = model.xr_dataset[Key.PING_MODE].to_numpy()
        pulse_form = model.xr_dataset[Key.PULSE_FORM].to_numpy()
        tx_sector_count = model.xr_dataset[Key.TX_SECTOR_COUNT].to_numpy()
        swath_per_ping = model.xr_dataset[Key.SWATH_PER_PING].to_numpy()
        multiping_sequence = model.xr_dataset[Key.MULTIPING_SEQUENCE].to_numpy()
        center_frequency = xsf.read_multiping_center_frequency()

        # compute signal modes
        return self.compute_keys_values(
            frequency_array=fq_mode,
            depth_mode=ping_mode,
            pulse_form=pulse_form,
            sector_count=tx_sector_count,
            swath_count=swath_per_ping,
            swath_index=multiping_sequence,
            center_frequency=center_frequency,
            global_keys=global_keys,
        )
