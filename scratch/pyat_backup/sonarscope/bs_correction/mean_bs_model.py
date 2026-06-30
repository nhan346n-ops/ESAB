"""
Model for backscatter angular response statistics according incidence and transmission angles
"""

from __future__ import annotations

import os.path
import re
from typing import Dict, Iterable, Optional, TypeAlias

import numpy as np
import pandas as pd
import xarray as xr

from pyat.sonarscope.model.sounder_lib import SounderType
from pyat.sonarscope.model.sounder_mode.bscorr_mode import KeyModeBscorr
from pyat.sonarscope.model.sounder_mode.all_kongsberg_mode import KeyModeAllGeneric
from pyat.sonarscope.model.sounder_mode.common_mode import KeyModeCommon
from pyat.sonarscope.model.sounder_mode.kmall_kongsberg_mode import KeyModeKmallGeneric
from pyat.utils.nc_encoding import open_nc_file

from ...utils.exceptions.exception_list import UnexpectedError
from ...utils.netcdf_utils import DEFAULT_COMPRESSION_LEVEL
from ..common.configuration import default_config
from ..model.sonar_factories import ModeComputerFactory
from ..model.sounder_mode.sounder_modes import KeyMode


class BackscatterCurve:
    """Backscatter mean values model referred to angle values"""

    MEAN_BS = "mean_bs"
    MEAN_RESIDUAL_BS = "mean_residual_bs"
    RAW_MEAN_BS = "raw_mean_bs"
    RAW_MEAN_RESIDUAL_BS = "raw_mean_residual_bs"
    VALUE_COUNT = "value_count"
    ANGLE = "angle"
    TX_BEAM = "tx_beam"
    RX_ANTENNA = "rx_antenna"
    PING_TIME = "ping_time"
    MODE = "mode"

    """a compensation curve, stored as a xarray in memory"""

    def __init__(self, xr_dataset: xr.Dataset, origin: Optional[str] = None):
        self.ds = xr_dataset
        self.origin = origin

    @classmethod
    def from_netcdf(cls, filepath, group_name: Optional[str] = None) -> BackscatterCurve:
        ds = xr.open_dataset(filepath, engine="netcdf4", group=group_name)
        return BackscatterCurve(xr_dataset=ds, origin=group_name)

    def to_netcdf(self, filepath, group_name: Optional[str] = None, mode="a"):
        self.ds.to_netcdf(filepath, mode=mode, engine="netcdf4", group=group_name)

    def to_csv(self, filepath, mode="a"):
        self.ds[BackscatterCurve.MEAN_BS].to_pandas().to_csv(filepath, sep=";", header=False, mode=mode)

    def __eq__(self, other):
        if self.ds is None and other.ds is None:
            return True
        if self.ds is None or other.ds is None:
            return False
        return self.ds.equals(other.ds)


class BackscatterCurveByIncidence(BackscatterCurve):
    @classmethod
    def build(
        cls,
        mean_values: np.ndarray,
        count: np.ndarray,
        bin_centers: np.ndarray,
        raw_mean_values: Optional[np.ndarray] = None,
        origin: Optional[str] = None,
    ):
        ds = xr.Dataset(
            data_vars={
                BackscatterCurve.MEAN_BS: (
                    [BackscatterCurve.ANGLE],
                    mean_values.astype(np.float64),
                    {"long_name": "filtered mean backscatter"},
                ),
                BackscatterCurve.RAW_MEAN_BS: (
                    [BackscatterCurve.ANGLE],
                    (
                        raw_mean_values.astype(np.float64)
                        if raw_mean_values is not None
                        else mean_values.astype(np.float64)
                    ),
                    {"long_name": "raw mean backscatter"},
                ),
                BackscatterCurve.VALUE_COUNT: (
                    [BackscatterCurve.ANGLE],
                    count.astype(np.int32),
                    {"long_name": "value count per bin"},
                ),
            },
            coords={
                BackscatterCurve.ANGLE: (
                    [BackscatterCurve.ANGLE],
                    bin_centers.astype(np.float64),
                    {"long_name": "incidence angle of the center of the bin"},
                ),
            },
        )
        return cls(xr_dataset=ds, origin=origin)

    @classmethod
    def from_csv(cls, filepath, sep: Optional[str] = None):
        df = pd.read_csv(
            filepath,
            sep=sep,
            names=[BackscatterCurve.ANGLE, BackscatterCurve.MEAN_BS],
            comment="#",
            skip_blank_lines=True,
            skipinitialspace=True,
            engine="python",
        )
        mean_values = df[BackscatterCurve.MEAN_BS].to_numpy()
        bin_centers = df[BackscatterCurve.ANGLE].to_numpy()
        count = np.full(fill_value=1, shape=(bin_centers.shape[0]), dtype=np.int32)
        return cls.build(mean_values=mean_values, count=count, bin_centers=bin_centers, origin=filepath)


class BackscatterCurveByIncidenceByPing(BackscatterCurve):
    @classmethod
    def build(
        cls,
        mean_values: np.ndarray,
        count: np.ndarray,
        bin_centers: np.ndarray,
        ping_time: np.ndarray,
        mode: np.ndarray,
        origin: Optional[str] = None,
    ):
        ds = xr.Dataset(
            data_vars={
                BackscatterCurve.MEAN_BS: (
                    [BackscatterCurve.PING_TIME, BackscatterCurve.ANGLE],
                    mean_values.astype(np.float64),
                    {"long_name": "mean backscatter"},
                ),
                BackscatterCurve.VALUE_COUNT: (
                    [BackscatterCurve.PING_TIME, BackscatterCurve.ANGLE],
                    count.astype(np.float64),  # use float to be able to use nan and merge with other curves
                    {"long_name": "value count per bin"},
                ),
                BackscatterCurve.MODE: (
                    [BackscatterCurve.PING_TIME],
                    mode.astype(np.float64),  # use float to be able to use nan and merge with other curves
                    {"long_name": "mode index per ping"},
                ),
            },
            coords={
                BackscatterCurve.PING_TIME: (
                    [BackscatterCurve.PING_TIME],
                    ping_time,
                    {"long_name": "ping time"},
                ),
                BackscatterCurve.ANGLE: (
                    [BackscatterCurve.ANGLE],
                    bin_centers.astype(np.float64),
                    {"long_name": "incidence angle of the center of the bin"},
                ),
            },
        )
        ds[BackscatterCurve.PING_TIME].encoding["calendar"] = "gregorian"
        ds[BackscatterCurve.PING_TIME].encoding["units"] = "nanoseconds since 1970-01-01 00:00:00Z"

        ds[BackscatterCurve.MEAN_BS].encoding["zlib"] = True
        ds[BackscatterCurve.MEAN_BS].encoding["complevel"] = DEFAULT_COMPRESSION_LEVEL
        ds[BackscatterCurve.MEAN_BS].encoding["chunksizes"] = [1000, 0]
        ds[BackscatterCurve.VALUE_COUNT].encoding["zlib"] = True
        ds[BackscatterCurve.VALUE_COUNT].encoding["complevel"] = DEFAULT_COMPRESSION_LEVEL
        ds[BackscatterCurve.VALUE_COUNT].encoding["chunksizes"] = [1000, 0]
        ds[BackscatterCurve.MODE].encoding["zlib"] = True
        ds[BackscatterCurve.MODE].encoding["complevel"] = DEFAULT_COMPRESSION_LEVEL
        ds[BackscatterCurve.MODE].encoding["chunksizes"] = [1000]

        return cls(xr_dataset=ds, origin=origin)


class BackscatterCurveByTransmission(BackscatterCurve):
    @classmethod
    def build(
        cls,
        rx_antenna_count,
        tx_beam_count,
        mean_values: np.ndarray,
        mean_residual_values: np.ndarray,
        count: np.ndarray,
        bin_centers: np.ndarray,
        raw_mean_residual_values: Optional[np.ndarray] = None,
        origin: Optional[str] = None,
    ):
        if mean_values.shape[0] != count.shape[0] or mean_values.shape[0] != rx_antenna_count:
            raise UnexpectedError(
                f"{BackscatterCurve.__name__} expect 3D array indexed per rx antenna, per tx sector, the first dimension does not match rx_antenna_count = {rx_antenna_count}"
            )

        if mean_values.shape[1] != count.shape[1] or mean_values.shape[1] != tx_beam_count:
            raise UnexpectedError(
                f"{BackscatterCurve.__name__} expect 3D array indexed per rx antenna, per tx sector, the second dimension does not match tx_beam_count = {tx_beam_count}"
            )

        ds = xr.Dataset(
            data_vars={
                BackscatterCurve.MEAN_BS: (
                    [BackscatterCurve.RX_ANTENNA, BackscatterCurve.TX_BEAM, BackscatterCurve.ANGLE],
                    mean_values.astype(np.float64),
                    {"long_name": "mean backscatter"},
                ),
                BackscatterCurve.MEAN_RESIDUAL_BS: (
                    [BackscatterCurve.RX_ANTENNA, BackscatterCurve.TX_BEAM, BackscatterCurve.ANGLE],
                    mean_residual_values.astype(np.float64),
                    {
                        "long_name": "filtered mean residual backscatter",
                        "comment": "difference between measured bs and computed bs from incidence angle",
                    },
                ),
                BackscatterCurve.RAW_MEAN_RESIDUAL_BS: (
                    [BackscatterCurve.RX_ANTENNA, BackscatterCurve.TX_BEAM, BackscatterCurve.ANGLE],
                    (
                        raw_mean_residual_values.astype(np.float64)
                        if raw_mean_residual_values is not None
                        else mean_residual_values.astype(np.float64)
                    ),
                    {"long_name": "raw mean residual backscatter"},
                ),
                BackscatterCurve.VALUE_COUNT: (
                    [BackscatterCurve.RX_ANTENNA, BackscatterCurve.TX_BEAM, BackscatterCurve.ANGLE],
                    count.astype(np.int32),
                    {"long_name": "value count per bin"},
                ),
            },
            coords={
                BackscatterCurve.RX_ANTENNA: (
                    [BackscatterCurve.RX_ANTENNA],
                    np.arange(0, rx_antenna_count, dtype=np.int32),
                    {"long_name": "rx antenna index"},
                ),
                BackscatterCurve.TX_BEAM: (
                    [BackscatterCurve.TX_BEAM],
                    np.arange(0, tx_beam_count, dtype=np.int32),
                    {"long_name": "tx beam index"},
                ),
                BackscatterCurve.ANGLE: (
                    [BackscatterCurve.ANGLE],
                    bin_centers.astype(np.float64),
                    {"long_name": "transmission angle of the center of the bin"},
                ),
            },
        )
        return cls(xr_dataset=ds, origin=origin)


class BackscatterCurveByTransmissionByPing(BackscatterCurve):
    @classmethod
    def build(
        cls,
        rx_antenna_count,
        tx_beam_count,
        mean_values: np.ndarray,
        mean_residual_values: np.ndarray,
        count: np.ndarray,
        bin_centers: np.ndarray,
        ping_time: np.ndarray,
        origin: Optional[str] = None,
    ):
        if (
            mean_values.shape[0] != mean_residual_values.shape[0]
            or mean_residual_values.shape[0] != count.shape[0]
            or mean_residual_values.shape[0] != rx_antenna_count
        ):
            raise UnexpectedError(
                f"{BackscatterCurve.__name__} expect 4D array indexed per rx antenna, tx sector, ping and angle the first dimension does not match rx_antenna_count = {rx_antenna_count}"
            )

        if (
            mean_values.shape[1] != count.shape[1]
            or mean_residual_values.shape[1] != count.shape[1]
            or mean_residual_values.shape[1] != tx_beam_count
        ):
            raise UnexpectedError(
                f"{BackscatterCurve.__name__} expect 4D array indexed per rx antenna, tx sector, ping and angle the second dimension does not match tx_beam_count = {tx_beam_count}"
            )

        ds = xr.Dataset(
            data_vars={
                BackscatterCurve.MEAN_BS: (
                    [
                        BackscatterCurve.RX_ANTENNA,
                        BackscatterCurve.TX_BEAM,
                        BackscatterCurve.PING_TIME,
                        BackscatterCurve.ANGLE,
                    ],
                    mean_values.astype(np.float64),
                    {"long_name": "mean backscatter"},
                ),
                BackscatterCurve.MEAN_RESIDUAL_BS: (
                    [
                        BackscatterCurve.RX_ANTENNA,
                        BackscatterCurve.TX_BEAM,
                        BackscatterCurve.PING_TIME,
                        BackscatterCurve.ANGLE,
                    ],
                    mean_residual_values.astype(np.float64),
                    {
                        "long_name": "mean residual backscatter",
                        "comment": "difference between measured bs and computed bs from incidence angle",
                    },
                ),
                BackscatterCurve.VALUE_COUNT: (
                    [
                        BackscatterCurve.RX_ANTENNA,
                        BackscatterCurve.TX_BEAM,
                        BackscatterCurve.PING_TIME,
                        BackscatterCurve.ANGLE,
                    ],
                    count.astype(np.float64),  # use float to be able to use nan and merge with other curves
                    {"long_name": "value count per bin"},
                ),
            },
            coords={
                BackscatterCurve.RX_ANTENNA: (
                    [BackscatterCurve.RX_ANTENNA],
                    np.arange(0, rx_antenna_count, dtype=np.int32),
                    {"long_name": "rx antenna index"},
                ),
                BackscatterCurve.TX_BEAM: (
                    [BackscatterCurve.TX_BEAM],
                    np.arange(0, tx_beam_count, dtype=np.int32),
                    {"long_name": "tx beam index"},
                ),
                BackscatterCurve.PING_TIME: (
                    [BackscatterCurve.PING_TIME],
                    ping_time,
                    {"long_name": "ping time"},
                ),
                BackscatterCurve.ANGLE: (
                    [BackscatterCurve.ANGLE],
                    bin_centers.astype(np.float64),
                    {"long_name": "transmission angle of the center of the bin"},
                ),
            },
        )
        ds[BackscatterCurve.PING_TIME].encoding["calendar"] = "gregorian"
        ds[BackscatterCurve.PING_TIME].encoding["units"] = "nanoseconds since 1970-01-01 00:00:00Z"
        ds[BackscatterCurve.MEAN_BS].encoding["zlib"] = True
        ds[BackscatterCurve.MEAN_BS].encoding["complevel"] = DEFAULT_COMPRESSION_LEVEL
        ds[BackscatterCurve.MEAN_BS].encoding["chunksizes"] = [1, 1, 1000, 0]
        ds[BackscatterCurve.MEAN_RESIDUAL_BS].encoding["zlib"] = True
        ds[BackscatterCurve.MEAN_RESIDUAL_BS].encoding["complevel"] = DEFAULT_COMPRESSION_LEVEL
        ds[BackscatterCurve.MEAN_RESIDUAL_BS].encoding["chunksizes"] = [1, 1, 1000, 0]
        ds[BackscatterCurve.VALUE_COUNT].encoding["zlib"] = True
        ds[BackscatterCurve.VALUE_COUNT].encoding["complevel"] = DEFAULT_COMPRESSION_LEVEL
        ds[BackscatterCurve.VALUE_COUNT].encoding["chunksizes"] = [1, 1, 1000, 0]

        return cls(xr_dataset=ds, origin=origin)


BackscatterCurvesByMode: TypeAlias = tuple[BackscatterCurveByIncidence, BackscatterCurveByTransmission]
BackscatterCurvesByModeByPing: TypeAlias = tuple[
    BackscatterCurveByIncidenceByPing, BackscatterCurveByTransmissionByPing
]


class MeanBSModel:
    """
    Class for mean backscatter model per angle
    Mean values are computed per mode
    """

    # File extension
    EXTENSION = ".bsar.nc"

    # netcdf mode subgroup
    INCIDENCE_SUBGROUP = "by_incidence_angle"
    TRANSMISSION_SUBGROUP = "by_transmission_angle"

    # netcdf attributes
    TITLE = "title"
    VERSION = "bs_angular_response_version"
    SOUNDER_TYPE = "sounder_type"

    USE_SVP = "use_sound_velocity_profiles"
    USE_SNIPPETS = "use_snippets"
    USE_INSONIFIED_AREA = "use_insonified_area"
    REMOVE_CALIBRATION = "remove_calibration"
    REMOVE_COMPENSATION = "remove_compensation"
    INTEGRATION_METHOD = "integration_method"
    LINEAR_SCALE = "linear_scale"

    MODE_SERIALIZED = "mode_serialized"

    def __init__(
        self,
        sounder_type: Optional[str],
        mode_curves: Dict[KeyMode, BackscatterCurvesByMode],
    ):
        self.model = mode_curves
        self.sounder_type = sounder_type

    def load_in_memory(self):
        """Load the model entireley and release bounded file resources"""
        for key, (curve_by_incidence, curve_by_sector) in self.model.items():
            if curve_by_incidence is not None:
                curve_by_incidence.ds = curve_by_incidence.ds.load().copy(deep=True)
            if curve_by_sector is not None:
                curve_by_sector.ds = curve_by_sector.ds.load().copy(deep=True)
        return self

    def get_curve_by_incidence(self, mode: KeyMode) -> Optional[BackscatterCurveByIncidence]:
        # retrieve curve by incidence
        if mode in self.model.keys():
            curve_by_incidence, _ = self.model[mode]
        elif KeyModeCommon() in self.model.keys():
            curve_by_incidence, _ = self.model[KeyModeCommon()]
        else:
            curve_by_incidence = None
        return curve_by_incidence

    @classmethod
    def build_from_incidence(
        cls, curve_by_incidence: BackscatterCurveByIncidence, sounder_type: str | None = None
    ) -> MeanBSModel:
        mode_curves = {KeyModeCommon(): (curve_by_incidence, None)}
        return MeanBSModel(mode_curves=mode_curves, sounder_type=sounder_type or SounderType.COMMON)

    def save_to_netcdf(self, output_file: str, overwrite: bool = False):
        """Export model to netcdf"""
        #
        if not overwrite and os.path.exists(output_file):
            default_config.logger.error(f"Output file {output_file} already exist and overwrite is not allowed")
            raise IOError(f"Output file {output_file} already exist and overwrite is not allowed")
        with open_nc_file(output_file, mode="w") as ncdataset:
            ncdataset.setncattr(self.TITLE, "Mean backscatter angular response")
            ncdataset.setncattr(self.VERSION, "0.3")
            ncdataset.setncattr(self.SOUNDER_TYPE, self.sounder_type)
            if self.sounder_type is not SounderType.COMMON:  # only set these attributes for specific sounders
                ncdataset.setncattr(self.USE_SVP, str(default_config.use_svp))
                ncdataset.setncattr(self.USE_SNIPPETS, str(default_config.use_snippets))
                ncdataset.setncattr(self.USE_INSONIFIED_AREA, str(default_config.use_insonified_area))
                ncdataset.setncattr(self.REMOVE_CALIBRATION, str(default_config.remove_calibration))
                ncdataset.setncattr(self.REMOVE_COMPENSATION, str(default_config.remove_compensation))

            ncdataset.setncattr(self.INTEGRATION_METHOD, default_config.integration_method.name)
            ncdataset.setncattr(self.LINEAR_SCALE, default_config.linear_scale.name)

            for current_mode in self.model.keys():
                mode_astxt = current_mode.to_json()
                grp = ncdataset.createGroup(str(current_mode))
                grp.setncattr(self.MODE_SERIALIZED, mode_astxt)

        # use xarray to serialize everything else,
        for current_mode in self.model.keys():
            group_name = str(current_mode)
            curve_by_incidence, curve_by_sector = self.model[current_mode]
            curve_by_incidence.ds.to_netcdf(
                output_file, mode="a", engine="netcdf4", group=f"{group_name}/{MeanBSModel.INCIDENCE_SUBGROUP}"
            )
            if curve_by_sector:
                curve_by_sector.ds.to_netcdf(
                    output_file, mode="a", engine="netcdf4", group=f"{group_name}/{MeanBSModel.TRANSMISSION_SUBGROUP}"
                )
        default_config.logger.info(f"Write compensation model to {output_file}")

    @staticmethod
    def read_from_netcdf(input_file, apply_conf: bool = True, in_memory: bool = True) -> MeanBSModel:
        mode_curves = {}
        with open_nc_file(input_file, mode="r") as ncdataset:
            sounder_type = ncdataset.getncattr(MeanBSModel.SOUNDER_TYPE)
            if apply_conf:
                attrs = ncdataset.ncattrs()
                if MeanBSModel.USE_SVP in attrs:
                    use_svp = ncdataset.getncattr(MeanBSModel.USE_SVP)
                    default_config.set_use_svp(use_svp=(use_svp.lower() == "true"))
                if MeanBSModel.USE_SNIPPETS in attrs:
                    use_snippets = ncdataset.getncattr(MeanBSModel.USE_SNIPPETS)
                    default_config.set_use_snippets(use_snippets=(use_snippets.lower() == "true"))
                if MeanBSModel.USE_INSONIFIED_AREA in attrs:
                    use_insonified_area = ncdataset.getncattr(MeanBSModel.USE_INSONIFIED_AREA)
                    default_config.set_use_insonified_area(use_insonified_area=(use_insonified_area.lower() == "true"))
                if MeanBSModel.REMOVE_CALIBRATION in attrs:
                    remove_calibration = ncdataset.getncattr(MeanBSModel.REMOVE_CALIBRATION)
                    default_config.set_remove_calibration(remove_calibration=(remove_calibration.lower() == "true"))
                if MeanBSModel.REMOVE_COMPENSATION in attrs:
                    remove_compensation = ncdataset.getncattr(MeanBSModel.REMOVE_COMPENSATION)
                    default_config.set_remove_compensation(remove_compensation=(remove_compensation.lower() == "true"))

            for grp in ncdataset.groups:
                if MeanBSModel.MODE_SERIALIZED in ncdataset.groups[grp].ncattrs():
                    # this group contains is a serialized bs angular curve
                    mode_astxt = ncdataset.groups[grp].getncattr(MeanBSModel.MODE_SERIALIZED)
                    current_mode = ModeComputerFactory.key_mode_from_json(
                        sounder_type=sounder_type, json_text=mode_astxt
                    )
                    if MeanBSModel.INCIDENCE_SUBGROUP in ncdataset.groups[grp].groups:
                        curve_by_incidence = BackscatterCurveByIncidence.from_netcdf(
                            filepath=input_file, group_name=f"{grp}/{MeanBSModel.INCIDENCE_SUBGROUP}"
                        )
                    else:
                        curve_by_incidence = None
                    if MeanBSModel.TRANSMISSION_SUBGROUP in ncdataset.groups[grp].groups:
                        curve_by_sector = BackscatterCurveByTransmission.from_netcdf(
                            filepath=input_file, group_name=f"{grp}/{MeanBSModel.TRANSMISSION_SUBGROUP}"
                        )
                    else:
                        curve_by_sector = None
                    mode_curves[current_mode] = (curve_by_incidence, curve_by_sector)
                else:
                    default_config.logger.warning(
                        f"Group {grp} does not have {MeanBSModel.MODE_SERIALIZED} attribute and is ignored"
                    )
        meanbs_model = MeanBSModel(mode_curves=mode_curves, sounder_type=sounder_type)
        if in_memory:
            meanbs_model.load_in_memory()

        return meanbs_model

    def export_incidence_to_csv(self, output_dir: str, overwrite: bool = False):
        """Export model to csv"""
        common_mode = KeyModeCommon()
        if common_mode not in self.model.keys():
            output_file = os.path.join(output_dir, f"{common_mode.short_name()}.txt")
            default_config.check_output_path(output_file, overwrite)
            bin_centers = default_config.incidence_angles.bin_centers
            squeleton = BackscatterCurveByIncidence.build(
                mean_values=np.full(fill_value=0.0, shape=bin_centers.shape[0]),
                count=np.full(fill_value=1, shape=(bin_centers.shape[0]), dtype=np.int32),
                bin_centers=bin_centers,
                origin=None,
            )
            with open(file=output_file, mode="w", encoding="utf_8") as f:
                f.write(f"#{self.sounder_type} {common_mode.mode_to_json()}\n")
            squeleton.to_csv(output_file, mode="a")

        for current_mode in self.model.keys():
            output_file = os.path.join(output_dir, f"{current_mode.short_name()}.txt")
            default_config.check_output_path(output_file, overwrite)
            curve_by_incidence, _ = self.model[current_mode]
            with open(file=output_file, mode="w", encoding="utf_8") as f:
                f.write(f"#{self.sounder_type} {current_mode.mode_to_json()}\n")
            curve_by_incidence.to_csv(output_file, mode="a")

    @staticmethod
    def import_from_csv(input_files: Iterable[str], sounder_type: Optional[str] = None, sep: Optional[str] = None):
        mode_curves = {}
        for input_file in input_files:
            with open(file=input_file, mode="r", encoding="utf_8") as f:
                first_line = f.readline()
                mode = KeyModeCommon()
                for stype in SounderType.SOUNDER_TYPES:
                    if stype in first_line:
                        if sounder_type and sounder_type != stype:
                            raise IOError(f"Different sounder type found : {sounder_type} {stype}")
                        sounder_type = stype
                        json_list = re.findall("{(.*?)}", first_line)
                        if len(json_list) and len(json_list[0]):
                            mode = ModeComputerFactory.key_mode_from_json(
                                sounder_type=stype, json_text=f"{{{json_list[0]}}}"
                            )
                        break
            curve_by_incidence = BackscatterCurveByIncidence.from_csv(input_file, sep=sep)
            mode_curves[mode] = (curve_by_incidence, None)
        return MeanBSModel(mode_curves=mode_curves, sounder_type=sounder_type or SounderType.COMMON)

    def __eq__(self, other):
        if not isinstance(other, MeanBSModel):
            return False
        if self.sounder_type != other.sounder_type:
            return False
        if len(self.model) != len(other.model):
            return False
        for sk, so in zip(self.model.keys(), other.model.keys()):
            if sk != so:
                return False
            if not self.model[sk] == other.model[so]:
                return False

        return True

    def find_equivalent_mode(self, bscorr_mode: KeyModeBscorr) -> Optional[KeyMode]:
        """Find equivalent mode in the model"""

        # retrieve dualswath mode indices
        swath_indices = []
        for current_mode in self.model.keys():
            if isinstance(current_mode, KeyModeAllGeneric):
                if current_mode.swath_mode is not None and current_mode.swath_mode > 0:
                    swath_indices.append(current_mode.swath_index)
            elif isinstance(current_mode, KeyModeKmallGeneric):
                if current_mode.swath_count is not None and current_mode.swath_count == 2:
                    swath_indices.append(current_mode.swath_index)

        # remove duplicates and sort (could be [0,1] or [1,2])
        swath_indices = np.unique(swath_indices)

        for current_mode in self.model.keys():
            if isinstance(current_mode, KeyModeAllGeneric):
                # check if depth mode is the same
                if bscorr_mode.ping_mode != current_mode.ping_mode:
                    continue
                # check if swath mode is the same in single swath mode
                if bscorr_mode.swath_index == 0 and current_mode.swath_mode != 0:
                    continue
                # check if dual swath index is the same in dual swath mode
                if (
                    bscorr_mode.swath_index is not None
                    and bscorr_mode.swath_index > 0
                    and (
                        bscorr_mode.swath_index > len(swath_indices)
                        or current_mode.swath_index != swath_indices[bscorr_mode.swath_index - 1]
                    )
                ):
                    continue
                # all checks passed, return the current mode
                return current_mode
            elif isinstance(current_mode, KeyModeKmallGeneric):
                # check if depth mode is the same
                if bscorr_mode.ping_mode != current_mode.depth_mode:
                    continue
                # check if swath mode is the same in single swath mode
                if bscorr_mode.swath_index == 0 and current_mode.swath_count != 1:
                    continue
                # check if dual swath index is the same in dual swath mode
                if (
                    bscorr_mode.swath_index is not None
                    and bscorr_mode.swath_index > 0
                    and (
                        bscorr_mode.swath_index > len(swath_indices)
                        or current_mode.swath_index != swath_indices[bscorr_mode.swath_index - 1]
                    )
                ):
                    continue
                # all checks passed, return the current mode
                return current_mode
        return None


class SlidingMeanBSModel:
    """
    Class for sliding mean backscatter model per angle
    Mean values are computed per ping for incidence angles
    Mean values are computed per mode for transmission angles
    """

    # netcdf mode subgroup
    INCIDENCE_SUBGROUP = "by_incidence_angle"
    TRANSMISSION_SUBGROUP = "by_transmission_angle"

    # netcdf attributes
    TITLE = "title"
    VERSION = "bs_angular_response_version"
    SOUNDER_TYPE = "sounder_type"

    USE_SVP = "use_sound_velocity_profiles"
    USE_SNIPPETS = "use_snippets"
    USE_INSONIFIED_AREA = "use_insonified_area"
    REMOVE_CALIBRATION = "remove_calibration"

    MODE_SERIALIZED = "mode_serialized"

    def __init__(
        self,
        sounder_type: str,
        model_curves: BackscatterCurvesByModeByPing,
    ):
        self.incidence_model = model_curves[0]
        self.transmission_model = model_curves[1]
        self.sounder_type = sounder_type

    def save_to_netcdf(self, output_file: str, overwrite: bool = False):
        """Export model to netcdf"""
        #
        default_config.check_output_path(output_file, overwrite)
        with open_nc_file(output_file, mode="w") as ncdataset:
            ncdataset.setncattr(self.TITLE, "Sliding mean backscatter angular response")
            ncdataset.setncattr(self.VERSION, "0.3")
            ncdataset.setncattr(self.SOUNDER_TYPE, self.sounder_type)
            ncdataset.setncattr(self.USE_SVP, str(default_config.use_svp))
            ncdataset.setncattr(self.USE_SNIPPETS, str(default_config.use_snippets))
            ncdataset.setncattr(self.USE_INSONIFIED_AREA, str(default_config.use_insonified_area))
            ncdataset.setncattr(self.REMOVE_CALIBRATION, str(default_config.remove_calibration))

        # use xarray to serialize everything else,
        self.incidence_model.ds.to_netcdf(
            output_file, mode="a", engine="netcdf4", group=f"{MeanBSModel.INCIDENCE_SUBGROUP}"
        )
        self.transmission_model.ds.to_netcdf(
            output_file, mode="a", engine="netcdf4", group=f"{MeanBSModel.TRANSMISSION_SUBGROUP}"
        )

        default_config.logger.info(f"Write compensation model to {output_file}")

    @staticmethod
    def read_from_netcdf(input_file, apply_conf=True):
        with open_nc_file(input_file, mode="r") as ncdataset:
            sounder_type = ncdataset.getncattr(MeanBSModel.SOUNDER_TYPE)
            if apply_conf:
                attrs = ncdataset.ncattrs()
                if MeanBSModel.USE_SVP in attrs:
                    use_svp = ncdataset.getncattr(MeanBSModel.USE_SVP)
                    default_config.set_use_svp(use_svp=(use_svp.lower() == "true"))
                if MeanBSModel.USE_SNIPPETS in attrs:
                    use_snippets = ncdataset.getncattr(MeanBSModel.USE_SNIPPETS)
                    default_config.set_use_snippets(use_snippets=(use_snippets.lower() == "true"))
                if MeanBSModel.USE_INSONIFIED_AREA in attrs:
                    use_insonified_area = ncdataset.getncattr(MeanBSModel.USE_INSONIFIED_AREA)
                    default_config.set_use_insonified_area(use_insonified_area=(use_insonified_area.lower() == "true"))
                if MeanBSModel.REMOVE_CALIBRATION in attrs:
                    remove_calibration = ncdataset.getncattr(MeanBSModel.REMOVE_CALIBRATION)
                    default_config.set_remove_calibration(remove_calibration=(remove_calibration.lower() == "true"))

            incidence_model_curve = BackscatterCurveByIncidenceByPing.from_netcdf(
                filepath=input_file, group_name=f"{MeanBSModel.INCIDENCE_SUBGROUP}"
            )
            transmission_model_curve = BackscatterCurveByTransmissionByPing.from_netcdf(
                filepath=input_file, group_name=f"{MeanBSModel.TRANSMISSION_SUBGROUP}"
            )

        return SlidingMeanBSModel(
            sounder_type=sounder_type, model_curves=(incidence_model_curve, transmission_model_curve)
        )

    def __eq__(self, other):
        if not isinstance(other, SlidingMeanBSModel):
            return False
        if self.sounder_type != other.sounder_type:
            return False
        if self.incidence_model != other.incidence_model:
            return False
        if self.transmission_model != other.transmission_model:
            return False
        return True
