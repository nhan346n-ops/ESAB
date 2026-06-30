# choose function to work as energy or as amplitude
import os
from enum import Enum
from typing import List

from pygws.service.progress_monitor import ProgressMonitor

from pyat.sonarscope.common.angles import IncidenceAngleBins, TransmissionAngleBins
from pyat.sonarscope.model.sounder_lib import SounderManufacturer, SounderType
from pyat.utils import pyat_logger
from pyat.utils.exceptions.exception_list import InputError
from pyat.utils.signal import (
    amplitude_to_db,
    db_to_amplitude,
    db_to_energy,
    energy_to_db,
)
from pyat.xsf import xsf_driver
from pyat.xsf.xsf_driver import XsfDriver


class LinearScale(Enum):
    ENERGY = 1
    AMPLITUDE = 2


class IntegrationMethod(Enum):
    MEAN = 1
    MEDIAN = 2


class InterpolationMethod(Enum):
    NEAREST = 1
    LINEAR = 2


# set default behaviour
class Parameters:
    def __init__(self):
        self.db_to_linear = None
        self.linear_to_db = None
        self.integration_method = IntegrationMethod.MEAN
        self.set_linear_scale(working_scale=LinearScale.ENERGY)
        self.frequency_interpolation_method = InterpolationMethod.LINEAR
        self.logger = pyat_logger.logging.getLogger(__name__)
        self.monitor = ProgressMonitor()
        self.incidence_angles = IncidenceAngleBins()
        self.transmission_angles = TransmissionAngleBins()
        self.xsf_min_version = 0.5
        self.use_snippets = False
        self.use_svp = True
        self.use_insonified_area = True
        self.use_reference_by_sector = True
        self.remove_compensation = True
        self.remove_calibration = True
        self.sounder_type = None
        self.sounder_manufacturer = None

    def setup(self, sounder_type: str):
        self.sounder_manufacturer = SounderManufacturer.from_type(sounder_type)
        if SounderManufacturer.KONGSBERG == self.sounder_manufacturer:
            # default settings are ok
            return
        elif SounderManufacturer.RESON == self.sounder_manufacturer:
            self.remove_calibration = False
            self.use_insonified_area = False
        else:
            raise NotImplementedError(
                f"Sounder {sounder_type} not supported yet, coding error in {Parameters.__name__}"
            )

    def set_linear_scale(self, working_scale: LinearScale):
        """Set if db should be converted to energy or amplitude"""
        self.linear_scale = working_scale
        if working_scale == LinearScale.ENERGY:
            self.db_to_linear = db_to_energy
            self.linear_to_db = energy_to_db
        elif working_scale == LinearScale.AMPLITUDE:
            self.db_to_linear = db_to_amplitude
            self.linear_to_db = amplitude_to_db

    def set_integration_method(self, integration_method: IntegrationMethod):
        """Set if insonified area should be recomputed"""
        self.integration_method = integration_method

    def set_frequency_interpolation_method(self, interpolation_method: InterpolationMethod):
        """Set the method to use for interpolating frequencies from reference incidence curves"""
        self.frequency_interpolation_method = interpolation_method

    def set_use_reference_by_sector(self, use_reference_by_sector: bool):
        """Set if reference incidence curves should be used by sector or for the whole acquisition mode"""
        self.use_reference_by_sector = use_reference_by_sector

    def set_use_insonified_area(self, use_insonified_area: bool):
        """Set if insonified area should be recomputed"""
        self.use_insonified_area = use_insonified_area

    def set_remove_calibration(self, remove_calibration: bool):
        """Set if backscatter calibration should be removed or not"""
        self.remove_calibration = remove_calibration

    def set_remove_compensation(self, remove_compensation: bool):
        """Set if backscatter calibration should be removed or not"""
        self.remove_compensation = remove_compensation

    def set_use_svp(self, use_svp: bool):
        """Set if embedded sound velocity profiles should be used"""
        self.use_svp = use_svp

    def set_use_snippets(self, use_snippets: bool):
        """Set if detection backscatter should be computed from snippets"""
        self.use_snippets = use_snippets

    def check_version(self, xsf_dataset: XsfDriver):
        if xsf_dataset.get_version() < self.xsf_min_version:
            error = (
                f"Input xsf file {xsf_dataset.sounder_file.file_path} version ({xsf_dataset.get_version()}) is lower than minimal ({self.xsf_min_version}). "
                f"Please regenerate or upgrade your files."
            )
            self.logger.error(error)
            raise InputError(error)

    def check_files_version(self, input_files: List[str]):
        self.logger.info("Check input files version")
        for f in input_files:
            with xsf_driver.open_xsf(file_path=f) as xsf_file:
                self.check_version(xsf_dataset=xsf_file)

    def check_output_path(self, output_path: str, overwrite: bool):
        if not overwrite and os.path.exists(output_path):
            self.logger.error(f"Output file {output_path} already exist and overwrite is not allowed")
            raise IOError(f"Output file {output_path} already exist and overwrite is not allowed")

    def check_files_soundertype(self, input_files: List[str], sounder_type: str | None) -> str:
        """
        Check that all input files are of the same sounder type, or auto detect it if possible
        If sounder_type is AUTO or None, try to auto detect it from first input file
        If sounder_type is specified, check that all input files are of the same type
        Return the sounder type to be used
        """

        if sounder_type == SounderType.AUTO or sounder_type is None:
            self.logger.info("Auto detecting sounder type from first input file")
            sounder_type = None
        for f in input_files:
            with xsf_driver.open_xsf(file_path=f) as xsf_file:
                try:
                    file_sounder_type = SounderType.from_dataset(xsf_dataset=xsf_file)
                except NotImplementedError:
                    file_sounder_type = None
                    self.logger.warning(
                        f"Input xsf file {xsf_file.sounder_file.file_path} sounder type not recognized."
                    )
                if sounder_type is not None and file_sounder_type != sounder_type:
                    self.logger.warning(
                        f"Input xsf file {xsf_file.sounder_file.file_path} sounder type ({file_sounder_type}) is different from specified ({sounder_type})."
                    )
                elif sounder_type is None and file_sounder_type is not None:
                    sounder_type = file_sounder_type
                    self.logger.info(
                        f"Sounder type {sounder_type} from file {xsf_file.sounder_file.file_path} is used."
                    )
        if sounder_type is None:
            error = "Sounder type could not be auto detected, please specify it."
            self.logger.error(error)
            raise InputError(error)
        return sounder_type


default_config = Parameters()
