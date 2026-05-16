"""
Model for backscatter angular response statistics according incidence and transmission angles
"""

from typing import Dict, Optional

import numpy as np
import sonar_netcdf.vendor_types as km_types
import xarray as xr

from pyat.sonarscope.model.sounder_lib import SounderType
from pyat.sonarscope.model.sounder_mode.bscorr_mode import KeyModeBscorr
from pyat.utils.exceptions.exception_list import UnexpectedError

from ..common.configuration import default_config


class BSCorrCurve:
    """Backscatter correction values model referred to angle values"""

    BS_CORR = "bs_correction"
    ANGLE = "angle"
    TX_BEAM = "tx_beam"

    """a compensation curve, stored as a xarray in memory
    The first dimension is the tx beam index, the second dimension is the angle.
    The values are the backscatter correction values.
    The angles are the incidence angles of the center of the bin.
    The tx beam index is the index of the tx sector in the sonar system.
    The angles are in degrees.
    The values are the backscatter correction values in dB defined as the absolute applied source level for .all
    or relative to nominal level for .kmall
    """

    def __init__(self, xr_dataset: xr.Dataset, origin: Optional[str] = None):
        self.ds = xr_dataset
        self.origin = origin

    def __eq__(self, other):
        if self.ds is None and other.ds is None:
            return True
        if self.ds is None or other.ds is None:
            return False
        return self.ds.equals(other.ds)

    @classmethod
    def build(cls, tx_beam_count, bscorr_values: np.ndarray, angles: np.ndarray):
        if bscorr_values.shape[0] != tx_beam_count:
            raise UnexpectedError(
                f"{BSCorrCurve.__name__} expect 2D array indexed per tx sector, the first dimension does not match tx_beam_count = {tx_beam_count}"
            )

        ds = xr.Dataset(
            data_vars={
                BSCorrCurve.BS_CORR: (
                    [BSCorrCurve.TX_BEAM, BSCorrCurve.ANGLE],
                    bscorr_values.astype(np.float64),
                    {
                        "long_name": "bs correction (applied as absolute source level(.all) or relative to nominal level(.kmall))"
                    },
                ),
            },
            coords={
                BSCorrCurve.TX_BEAM: (
                    [BSCorrCurve.TX_BEAM],
                    np.arange(0, tx_beam_count, dtype=np.int32),
                    {"long_name": "tx beam index"},
                ),
                BSCorrCurve.ANGLE: (
                    [BSCorrCurve.ANGLE],
                    angles.astype(np.float64),
                    {"long_name": "incidence angle of the center of the bin"},
                ),
            },
        )
        return cls(xr_dataset=ds)


class BSCorrModel:
    """
    Class for kongsberg bscorr model
    """

    def __init__(
        self,
        sounder_type: Optional[str],
        mode_curves: Dict[KeyModeBscorr, BSCorrCurve],
    ):
        self.model = mode_curves
        self.sounder_type = sounder_type

    def get_curve(self, mode: KeyModeBscorr) -> Optional[BSCorrCurve]:
        """retrieve curve by incidence"""
        return self.model[mode] if mode in self.model.keys() else None

    def set_curve(self, mode: KeyModeBscorr, curve: BSCorrCurve):
        """Set curve for a given mode"""
        if not isinstance(curve, BSCorrCurve):
            raise TypeError(f"Expected BSCorrCurve, got {type(curve)}")
        self.model[mode] = curve

    EM71X_MODE_IDS = {
        1: km_types.KmPingMode.VERY_SHALLOW,
        2: km_types.KmPingMode.SHALLOW,
        3: km_types.KmPingMode.MEDIUM,
        4: km_types.KmPingMode.DEEP,
        5: km_types.KmPingMode.VERY_DEEP,
        6: km_types.KmPingMode.EXTRA_DEEP,
    }
    GENERIC_MODE_IDS = {
        1: km_types.KmPingMode.VERY_SHALLOW,
        2: km_types.KmPingMode.SHALLOW,
        3: km_types.KmPingMode.MEDIUM,
        4: km_types.KmPingMode.DEEP,
        5: km_types.KmPingMode.DEEPER,
        6: km_types.KmPingMode.VERY_DEEP,
        7: km_types.KmPingMode.EXTRA_DEEP,
        8: km_types.KmPingMode.EXTREME_DEEP,
    }

    @staticmethod
    def import_from_txt(input_file: str, sounder_type: str):
        use_source_level = False
        match sounder_type:
            case SounderType.EM710_ALL:
                use_source_level = True
                mode_ids_dict = BSCorrModel.EM71X_MODE_IDS
            case SounderType.EM712_KMALL:
                use_source_level = False
                mode_ids_dict = BSCorrModel.EM71X_MODE_IDS
            case SounderType.EM124_KMALL:
                use_source_level = False
                mode_ids_dict = BSCorrModel.GENERIC_MODE_IDS
            case SounderType.EM304_KMALL:
                use_source_level = False
                mode_ids_dict = BSCorrModel.GENERIC_MODE_IDS
            case _:
                use_source_level = True
                mode_ids_dict = BSCorrModel.GENERIC_MODE_IDS
                default_config.logger.error(f"Unknown or not supported sounder type {sounder_type}")
        mode_curves = {}
        lines = []
        with open(file=input_file, mode="r", encoding="utf_8") as f:
            for line in f:
                stripped_line = line.strip()  # Remove leading/trailing whitespace, including newlines
                # Check if the line is not empty and does not start with '#'
                if stripped_line and not stripped_line.startswith("#"):
                    lines.append(stripped_line)  # Append the cleaned line
        try:
            i = 0
            while i < len(lines):
                parts = lines[i].split()
                i += 1
                # read depth_mode/swath_index/sector_number
                if len(parts) != 3:
                    raise ValueError(f"Invalid line format: {i}:{lines[i]}")
                mode_id = int(parts[0])
                ping_mode = mode_ids_dict.get(mode_id, km_types.KmPingMode.UNKNOW).value
                swath_index = int(parts[1])
                sector_count = int(parts[2])
                mode = KeyModeBscorr(
                    mode_id=mode_id, ping_mode=ping_mode, swath_index=swath_index, sector_count=sector_count
                )

                angles_per_sector = []
                values_per_sector = []
                for _ in range(sector_count):
                    # read source level from .all bscorr file
                    source_level = 0.0
                    if use_source_level:
                        source_level = float(lines[i])
                        i += 1
                    # read number of pairs
                    num_pairs = int(lines[i])
                    i += 1
                    # read angle/value pairs
                    angles = []
                    values = []
                    for _ in range(num_pairs):
                        parts = lines[i].split()
                        i += 1
                        if len(parts) != 2:
                            raise ValueError(f"Invalid line format: {i}:{lines[i]}")
                        angle = float(parts[0])
                        if use_source_level:
                            # for all bscorr files compute absolute value
                            value = float(parts[1]) + source_level
                        else:
                            # for kmall bscorr files, the value is the correction relative to source level
                            value = -float(parts[1])
                        angles.append(angle)
                        values.append(value)
                    angles_per_sector.append(angles)
                    values_per_sector.append(values)

                # merge all angles
                angles_all_sectors = np.unique(np.concatenate(angles_per_sector))

                # aligned all values to the same angle
                for sector_index in range(sector_count):
                    # map values to common angle array using indices
                    idx = np.searchsorted(angles_all_sectors, angles_per_sector[sector_index])
                    new_values = np.full_like(angles_all_sectors, np.nan, dtype=np.float64)
                    new_values[idx] = values_per_sector[sector_index]
                    values_per_sector[sector_index] = new_values

                # create a 2D array of values as ndarray
                values_per_sector = np.array(values_per_sector, dtype=np.float64)

                mode_curves[mode] = BSCorrCurve.build(
                    tx_beam_count=sector_count,
                    bscorr_values=values_per_sector,
                    angles=angles_all_sectors,
                )
        except ValueError as e:
            default_config.logger.warning(f"Invalid bscorr format: {e}")

        return BSCorrModel(mode_curves=mode_curves, sounder_type=sounder_type)

    def export_to_txt(self, output_file: str):
        match self.sounder_type:
            case SounderType.EM710_ALL:
                self.export_for_all(output_file)
            case SounderType.EM124_KMALL | SounderType.EM304_KMALL | SounderType.EM712_KMALL:
                self.export_for_kmall(output_file)
            case _:
                default_config.logger.error(f"Unknown or not supported sounder type {self.sounder_type}")

    def export_for_all(self, output_file: str):
        with open(file=output_file, mode="w", encoding="UTF-8") as f:
            f.write("# Kongsberg bscorr file modified with pyAT\n")
            f.write(f"# {self.sounder_type}\n")
            for mode, curve in self.model.items():
                f.write(f"# {mode.short_name()}\n")
                f.write(f"{mode.mode_id}\t{mode.swath_index}\t{mode.sector_count}\n")
                for sector_index in range(mode.sector_count):
                    f.write(f"# TX Sector {sector_index + 1}:\n")
                    # write source level to .all bscorr file
                    source_level = np.nanmax(curve.ds[BSCorrCurve.BS_CORR][sector_index, :].values)
                    f.write(f"{source_level:.1f}\n")
                    # write number of pairs
                    mask = np.isfinite(curve.ds[BSCorrCurve.BS_CORR][sector_index, :].values)
                    num_pairs = len(curve.ds[BSCorrCurve.BS_CORR][sector_index, mask].values)
                    f.write(f"{num_pairs}\n")
                    # write angle/value pairs in reverse order
                    angles = curve.ds[BSCorrCurve.ANGLE][mask].values[::-1]
                    values = curve.ds[BSCorrCurve.BS_CORR][sector_index, mask].values[::-1]
                    for angle, value in zip(angles, values):
                        f.write(f"{angle:.1f}\t{(value - source_level):.1f}\n")

    def export_for_kmall(self, output_file: str):
        with open(file=output_file, mode="w", encoding="UTF-8") as f:
            f.write("# Kongsberg bscorr file modified with pyAT\n")
            f.write(f"# {self.sounder_type}\n")
            for mode, curve in self.model.items():
                f.write(f"# {mode.short_name()}\n")
                f.write(f"{mode.mode_id}   {mode.swath_index}   {mode.sector_count}\n")
                for sector_index in range(mode.sector_count):
                    f.write(f"# TX Sector {sector_index + 1}:\n")
                    # write number of pairs
                    mask = np.isfinite(curve.ds[BSCorrCurve.BS_CORR][sector_index, :].values)
                    num_pairs = len(curve.ds[BSCorrCurve.BS_CORR][sector_index, mask].values)
                    f.write(f"{num_pairs}\n")
                    # write angle/value pairs in reverse order
                    angles = curve.ds[BSCorrCurve.ANGLE][mask].values[::-1]
                    values = curve.ds[BSCorrCurve.BS_CORR][sector_index, mask].values[::-1]
                    for angle, value in zip(angles, values):
                        # for kmall bscorr files, the value is the correction removed to source level
                        # so we write the negative value
                        f.write(f"{angle:g} {-value:.2f}\n")

    def __eq__(self, other):
        if not isinstance(other, BSCorrModel):
            return False
        if self.sounder_type != other.sounder_type:
            return False
        if len(self.model) != len(other.model):
            return False
        for sk, so in zip(self.model.keys(), other.model.keys()):
            if sk != so:
                return False
            if self.model[sk] != other.model[so]:
                return False
        return True
