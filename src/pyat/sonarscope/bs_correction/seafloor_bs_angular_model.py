"""
Model for seafloor angular offset to apply, used to compensate for angular dependency variation of backscatter
"""

from typing import Tuple

import numpy as np

from pyat.sonarscope.bs_correction.mean_bs_model import (
    BackscatterCurveByIncidenceByPing,
    SlidingMeanBSModel,
)
from pyat.sonarscope.bs_correction.stats_computer import BackscatterCurve, MeanBSModel
from pyat.sonarscope.common.configuration import default_config
from pyat.sonarscope.model.sounder_mode.sounder_modes import KeyMode


class ConstantModel:
    """A constant seafloor model, expected response value is a constant value (thus mode independent)"""

    def __init__(self, mean_bs: MeanBSModel, bs_value=-20):
        self.value = bs_value
        # retain avg look up table per mode
        self.avg_incidence_lut = {}
        self.avg_incidence_angles = None

        self.avg_transmission_lut = {}
        self.avg_residual_transmission_lut = {}
        self.avg_transmission_angles = None

        # get an angular independent response model for the surveyed area
        for mode, (curve_by_incidence, curve_by_transmission) in mean_bs.model.items():
            avg_incidence_offset, incidence_angles = self.__get_avg_table(curve_by_incidence)
            avg_residual_transmission_offset, transmission_angles = self.__get_avg_residual_table(curve_by_transmission)

            # adapt incidence curve shape on transmission curve shape if needed (old format compatibility)
            if avg_incidence_offset.shape[0] == 1 and avg_residual_transmission_offset.shape[0] > 1:
                avg_incidence_offset = np.repeat(
                    avg_incidence_offset, avg_residual_transmission_offset.shape[0], axis=0
                )
            if avg_incidence_offset.shape[1] == 1 and avg_residual_transmission_offset.shape[1] > 1:
                avg_incidence_offset = np.repeat(
                    avg_incidence_offset, avg_residual_transmission_offset.shape[1], axis=1
                )

            self.avg_incidence_lut[mode] = avg_incidence_offset
            self.avg_incidence_angles = incidence_angles

            self.avg_residual_transmission_lut[mode] = avg_residual_transmission_offset
            self.avg_transmission_angles = transmission_angles

    def get_reference_level(self):
        return self.value

    def __get_avg_table(self, bs_values: BackscatterCurve) -> Tuple[np.ndarray, np.ndarray]:
        """
        return the value to add to bs (per angle) to get normalized value
        """
        return (
            self.get_reference_level() - bs_values.ds[BackscatterCurve.MEAN_BS].data,
            bs_values.ds[BackscatterCurve.ANGLE].data,
        )

    def __get_avg_residual_table(self, bs_values: BackscatterCurve) -> Tuple[np.ndarray, np.ndarray]:
        """
        return the value to add to bs corrected from incidence (per transmission angle) to get normalized value
        """
        return (
            -bs_values.ds[BackscatterCurve.MEAN_RESIDUAL_BS].data,
            bs_values.ds[BackscatterCurve.ANGLE].data,
        )

    def find_mode(self, mode: KeyMode) -> KeyMode | None:
        """
        Return the closest mode to the requested one
        Directly from avg_incidence_lut keys or an equal key
        """
        if mode in self.avg_incidence_lut:
            return mode
        else:
            return next((m for m in self.avg_incidence_lut.keys() if mode == m), None)

    def get_avg_incidence_lut(self, mode: KeyMode) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """Return the look up table array and angle definition array for the requested mode"""
        lut_mode = self.find_mode(mode)
        if lut_mode is None:
            default_config.logger.error(f"Mode {mode} is not define in mean bs model, skipping these values")
            return None, None
        return self.avg_incidence_lut[lut_mode], self.avg_incidence_angles

    def get_avg_residual_transmission_lut(self, mode: KeyMode) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """Return the look up table array and angle definition array for the requested mode"""
        lut_mode = self.find_mode(mode)
        if lut_mode is None:
            default_config.logger.error(f"Mode {mode} is not define in mean bs model, skipping these values")
            return None, None
        return self.avg_residual_transmission_lut[lut_mode], self.avg_transmission_angles


class SlidingModel:
    """A sliding seafloor model, expected response value is given by a specific angle range (default : 45 degrees)"""

    def __init__(
        self,
        mean_bs: SlidingMeanBSModel,
        ref_angles=(43, 47),
        sliding_window_min=1,
    ):
        self.sliding_short_window = np.timedelta64(sliding_window_min, "m").astype("timedelta64[ns]") / 10
        self.ref_angles = ref_angles

        incidence_curve = mean_bs.incidence_model
        transmission_curve = mean_bs.transmission_model

        # Sort along ping_time coordinate to make it monotonically increasing (required for interpolate_na)
        if BackscatterCurve.PING_TIME in incidence_curve.ds.dims:
            incidence_curve.ds = incidence_curve.ds.sortby(BackscatterCurve.PING_TIME)
        if BackscatterCurve.PING_TIME in transmission_curve.ds.dims:
            transmission_curve.ds = transmission_curve.ds.sortby(BackscatterCurve.PING_TIME)

        # estimate reference level by ping
        self.ref_level = (
            incidence_curve.ds[BackscatterCurve.MEAN_BS]
            .sel(angle=slice(ref_angles[0], ref_angles[1]))
            .mean(dim=BackscatterCurve.ANGLE, skipna=True)
            .interpolate_na(dim=BackscatterCurve.PING_TIME, method="linear")
            .interpolate_na(
                dim=BackscatterCurve.PING_TIME, method="nearest", fill_value="extrapolate"
            )  # extrapolate before and after
        )

        # estimate current level by ping
        corrected_level = (
            transmission_curve.ds[BackscatterCurve.MEAN_BS] - transmission_curve.ds[BackscatterCurve.MEAN_RESIDUAL_BS]
        )

        self.current_level = (
            corrected_level.where(
                (abs(corrected_level.angle) >= ref_angles[0]) & (abs(corrected_level.angle) <= ref_angles[1])
            )
            .mean(dim=[BackscatterCurve.RX_ANTENNA, BackscatterCurve.TX_BEAM, BackscatterCurve.ANGLE], skipna=True)
            .interpolate_na(dim=BackscatterCurve.PING_TIME, method="linear")
            .interpolate_na(
                dim=BackscatterCurve.PING_TIME, method="nearest", fill_value="extrapolate"
            )  # extrapolate before and after
        )

        # estimate inter modes offsets
        self.mode_offset = self._compute_inter_mode_offsets(incidence_curve)

        # get an angular independent response model for the surveyed area
        self.avg_incidence_lut = self.ref_level - self.mode_offset - incidence_curve.ds[BackscatterCurve.MEAN_BS]
        self.avg_residual_transmission_lut = -transmission_curve.ds[BackscatterCurve.MEAN_RESIDUAL_BS]

    def _compute_inter_mode_offsets(
        self,
        incidence_curve: BackscatterCurveByIncidenceByPing,
    ):
        """
        Return the offset array to apply to compensate levels between modes
        These offset are obtained by comparing backscatter mean level on pings around a change of mode
        The most present mode is considered to have no offset
        Modes having a transition with the main mode are directly computed.
        Modes without a transition are estimated by propagation of offsets already computed.
        """

        # Get mode indices array and identify invalid modes (-1)
        mode_idx_array = np.copy(incidence_curve.ds[BackscatterCurve.MODE].values)
        mode_idx_array[np.isnan(mode_idx_array)] = -1
        mode_idx_array = mode_idx_array.astype(int)

        # Sort modes by occurrence : most present first
        mode, mode_count = np.unique(mode_idx_array[mode_idx_array >= 0], return_counts=True)
        sorted_mode = mode[np.argsort(mode_count)][::-1]
        # Create 2 transition matrices between mode : diff value and count representing the reference level offsets
        # between each mode and the number of occurrences of these transitions.

        num_mode = int(max(sorted_mode) + 1)
        if num_mode > 1:
            mode_diff_values = np.full(shape=(num_mode, num_mode), fill_value=np.nan, dtype=np.float32)
            mode_diff_count = np.full(shape=(num_mode, num_mode), fill_value=0.0, dtype=np.float32)
            # Retrieve change mode ping indices
            mode_diff_array = incidence_curve.ds[BackscatterCurve.MODE].diff(BackscatterCurve.PING_TIME).values
            mode_diff_array[np.isnan(mode_diff_array)] = 0
            change_mode_idx = mode_diff_array.nonzero()[0]

            # fill transition matrix containing mean of ref level differences between modes
            for prev_idx, next_idx in zip(change_mode_idx, change_mode_idx + 1):
                # remove transition with pings further than rolling window
                prev_time = self.current_level[BackscatterCurve.PING_TIME][prev_idx]
                next_time = self.current_level[BackscatterCurve.PING_TIME][next_idx]
                if next_time - prev_time > self.sliding_short_window / 2:
                    continue

                # fill transition matrix
                pmode = mode_idx_array.data[prev_idx]
                nmode = mode_idx_array.data[next_idx]
                pdiff = mode_diff_values[pmode][nmode]
                pcount = mode_diff_count[pmode][nmode]
                ndiff = self.current_level[next_idx] - self.current_level[prev_idx]
                # forward
                if np.isnan(pdiff):
                    mode_diff_values[pmode][nmode] = ndiff
                    mode_diff_count[pmode][nmode] = 1
                else:
                    mode_diff_values[pmode][nmode] = (pdiff * pcount + ndiff) / (pcount + 1)
                    mode_diff_count[pmode][nmode] = pcount + 1
                # backward (offset from a mode A to B is the opposite of offset from B to A)
                mode_diff_values[nmode][pmode] = -mode_diff_values[pmode][nmode]
                mode_diff_count[nmode][pmode] = mode_diff_count[pmode][nmode]

            # Complete iteratively transition matrix using propagation of offset from most present mode to least
            mode_offset = np.full(shape=num_mode, fill_value=np.nan, dtype=float)
            for smode in sorted_mode:
                if np.isnan(mode_offset[smode]):
                    mode_offset[smode] = 0
                    self.__recursive_fill_mode(
                        mode_values_matrix=mode_diff_values,
                        mode_counts_matrix=mode_diff_count,
                        main_mode=smode,
                        current_mode=smode,
                    )
                    for imode in range(num_mode):
                        if np.isnan(mode_offset[imode]):
                            mode_offset[imode] = mode_offset[smode] + mode_diff_values[smode, imode]
        else:
            mode_offset = np.zeros(shape=num_mode, dtype=float)

        default_config.logger.info(f"mode offsets : {mode_offset} ")
        return np.where(mode_idx_array >= 0, mode_offset[mode_idx_array], 0)

    def __recursive_fill_mode(
        self, mode_values_matrix: np.ndarray, mode_counts_matrix: np.ndarray, main_mode: int, current_mode: int
    ):
        """
        Recursively fill transition matrix of offsets between modes.
        Transitions the most represented have more weight than others.
        The following algorithm is based on computation of equivalent strings strength to estimate indirect relationship between modes.

        @param: mode_values_matrix : intput/output offset values
        @param: mode_counts_matrix : intput/output transition counts
        @param: main_mode : first mode of the graph (offset=0)
        @param: current_mode : parent mode of this iteration

        """
        # Fill all modes connected to the current mode (having main_mode as root)
        mode_values_matrix[current_mode, current_mode] = 0
        relative_modes = mode_counts_matrix[current_mode, :].nonzero()[0]
        parent_modes_idx = mode_counts_matrix[main_mode, relative_modes].nonzero()[0]
        # parent mode are connected modes with offset from main_mode already filled
        parent_modes = relative_modes[parent_modes_idx] if main_mode != current_mode else []
        # child mode are connected modes with offset from main_mode to be filled
        child_modes = np.setdiff1d(relative_modes, parent_modes)
        # remove main_mode from child_modes
        child_modes = np.setdiff1d(child_modes, main_mode)

        # compute offset from main_mode to current_mode using parent_modes
        if mode_counts_matrix[main_mode, current_mode] == 0:
            total_count = 0
            total_diff = 0
            for parent in parent_modes:
                # equivalent weight calculated for each parent (as spring constant in series)
                new_count = (
                    mode_counts_matrix[main_mode, parent]
                    * mode_counts_matrix[parent, current_mode]
                    / (mode_counts_matrix[main_mode, parent] + mode_counts_matrix[parent, current_mode])
                )
                # offset calculated from main_mode to current_mode for each parent
                new_diff = mode_values_matrix[main_mode, parent] + mode_values_matrix[parent, current_mode]

                # compute weighted mean of parent offsets
                total_diff = (total_diff * total_count + new_diff * new_count) / (total_count + new_count)
                # parents weights are sumed (as spring constant in parallel)
                total_count += new_count
            mode_values_matrix[main_mode, current_mode] = total_diff
            mode_counts_matrix[main_mode, current_mode] = total_count

        # recurse to fill child modes order by decreasing weight
        for child in sorted(child_modes, key=lambda x: mode_counts_matrix[current_mode, x], reverse=True):
            self.__recursive_fill_mode(
                mode_values_matrix=mode_values_matrix,
                mode_counts_matrix=mode_counts_matrix,
                main_mode=main_mode,
                current_mode=child,
            )

    def get_reference_level(self):
        return self.ref_level

    def get_avg_incidence_lut(self, ping_time: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """Return the look up table array and angle definition array for the requested mode"""
        if ping_time is not None:
            avg_lut = self.avg_incidence_lut.sel(ping_time=ping_time).values
            return avg_lut, self.avg_incidence_lut[BackscatterCurve.ANGLE].values
        return (
            self.avg_incidence_lut[BackscatterCurve.MEAN_BS].values,
            self.avg_incidence_lut[BackscatterCurve.ANGLE].values,
        )

    def get_avg_residual_transmission_lut(
        self, ping_time: np.ndarray | None = None
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """Return the look up table array and angle definition array for the requested mode"""
        if ping_time is not None:
            avg_lut = self.avg_residual_transmission_lut.sel(ping_time=ping_time).values
            return avg_lut, self.avg_residual_transmission_lut[BackscatterCurve.ANGLE].values
        return (
            self.avg_residual_transmission_lut[BackscatterCurve.MEAN_RESIDUAL_BS].values,
            self.avg_residual_transmission_lut[BackscatterCurve.ANGLE].values,
        )
