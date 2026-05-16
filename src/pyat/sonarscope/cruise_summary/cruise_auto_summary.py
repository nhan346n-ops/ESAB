import os.path
from builtins import IOError
from typing import Dict

import numpy as np
import sonar_netcdf.sonar_groups as sg
import sonar_netcdf.vendor_types as vt

# pylint: disable=import-error
from bokeh.palettes import Category10_3, Category10_4, Category10_9
from sonar_netcdf.utils.print_color import error

import pyat.utils.pyat_logger as log
from pyat.sensor.nmea import GPSQualityIndicator
from pyat.sonarscope.cruise_summary.display.display import Plotter
from pyat.sonarscope.cruise_summary.display.display2D import Plotter2
from pyat.sonarscope.cruise_summary.display.metadata import Metadata
from pyat.sonarscope.cruise_summary.file_data import FileDataStore
from pyat.sonarscope.cruise_summary.global_data import GlobalDataModel
from pyat.sonarscope.model.constants import VariableKeys as Key
from pyat.sonarscope.model.signal.ping_detection_signal import PingDetectionSignal
from pyat.sonarscope.model.signal.ping_signal import PingSignal
from pyat.sonarscope.model.sonar_factories import ModeComputerFactory
from pyat.sonarscope.model.sonar_metadata import SonarFileMetaData
from pyat.sonarscope.model.sound_velocity_profile import SVPProfiles
from pyat.sonarscope.model.sounder_lib import SounderType
from pyat.utils.exceptions.exception_list import InputError
from pyat.utils.path_utils import scan_dir
from pyat.xsf import xsf_driver

# pylint: enable=import-error


class CruiseAutoSummary:
    def __init__(self, input_dir: str, work_dir: str):
        self.logger = log.logging.getLogger(CruiseAutoSummary.__name__)
        self.input_dir = input_dir
        self.work_dir = work_dir  # workdir is use to put all intermediate files that are generated
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        if not os.path.isdir(self.work_dir):
            raise IOError(f"{work_dir} exists but is not a directory")
        self.data = None  # created after parse of files
        self.display_ping_time = None  # created after parsing of files
        self.display_ping_detection_time = None  # created after parsing of files

    def parse_files(self, file_pattern="*"):
        """Read all files, we do not merge them but instead create one light platform dataset for each file"""
        file_list = list(scan_dir(self.input_dir, [f"{file_pattern}.xsf.nc"]))
        input_files = list(map(str, file_list))
        if len(input_files) == 0:
            log.info(f"No file to read with pattern {file_pattern}.xsf.nc")
            return

        sounder_types = []
        file_data: Dict[str, FileDataStore] = {}
        # read all files
        for f in input_files:
            # Read minimum dataset for summary report
            log.info(f"Reading file  {f}")
            try:
                xsf = xsf_driver.XsfDriver(file_path=f)
                xsf.open()
                model = PingSignal(xsf_dataset=xsf)
                model.read([Key.TRANSMIT_TYPE], reductor_function=np.nanmax)
                model.read([Key.TX_SECTOR_COUNT], reductor_function=np.nanmax)
                model.read(
                    [
                        Key.PING_TIME,
                        Key.PLATFORM_ROLL,
                        Key.PLATFORM_PITCH,
                        Key.PLATFORM_HEADING,
                        Key.PLATFORM_LONGITUDE,
                        Key.PLATFORM_LATITUDE,
                        Key.SOUND_SPEED_AT_TRANSDUCER,
                        Key.DETECTION_Z,
                        Key.DETECTION_QUALITY_FACTOR,
                        Key.BATHYMETRY_STATUS,
                        Key.MEAN_ABS_COEFF,
                        Key.BACKSCATTER_NORMAL_INCIDENCE_LEVEL,
                        Key.BACKSCATTER_OBLIQUE_INCIDENCE_LEVEL,
                        Key.FREQUENCY_MODE,
                        Key.PING_MODE,
                        Key.PULSE_LENGTH_MODE,
                        Key.TX_SECTOR_COUNT,
                        Key.DUAL_SWATH_MODE,
                        Key.SAMPLE_COUNT,
                        Key.POSITION_SENSOR_QUALITY_INDICATOR,
                        Key.INTERPINGS_DISTANCE,
                        Key.WC_PRESENCE,
                        Key.RANGE_SELECTION,
                        Key.GAIN_SELECTION,
                        Key.POWER_SELECTION,
                    ],
                    ignore_unknown_variables=True,
                )
                model.read(keys=[Key.SWATH_PER_PING])

                # # continue parsing, read global attributes

                file_dates = model.variables[sg.BeamGroup1Grp.PING_TIME_VNAME].values

                # check time for rejection
                file_min_date = file_dates.min()
                file_max_date = file_dates.max()
                time_delta = file_max_date - file_min_date
                if time_delta > np.timedelta64(120, "D"):
                    d = time_delta.astype("timedelta64[D]")
                    error(
                        f"reject file {f} : file is spread over more that 120 days ({d}),\
                         which probably means there is an issue in dates"
                    )
                    continue

                # continue parsing, read global attributes
                metadata = SonarFileMetaData(xsf_dataset=xsf)
                metadata.read()

                # SSP drive use xarray,
                ssp_ds = SVPProfiles(file_path=f)
                ssp_ds.read_svp()

                # parse 2d variables
                model_detection = PingDetectionSignal(xsf_dataset=xsf)
                model_detection.read([Key.DETECTION_BACKSCATTER], reductor_function=np.nanmax)

                # compute sounder type
                sounder_types.append(SounderType.from_dataset(xsf_dataset=xsf))

                xsf.close()

                # create a data store to retain values
                fds = FileDataStore(
                    file=f,
                    ping_timed_dataset=model,
                    global_attributes=metadata,
                    sound_speed_profile=ssp_ds,
                    ping_detection_dataset=model_detection,
                )
                file_data[f] = fds
            except Exception as e:
                log.error(f"An error occurred while reading file {f} ", e)
                if f in file_data:
                    file_data.pop(f)

        self.data = GlobalDataModel(file_data=file_data)

        # check that only one sounder type is defined
        sounder_types_list = np.unique(sounder_types)
        if len(sounder_types_list) > 1:
            raise InputError(f"Unsupported input: several sounder types detected in repository {sounder_types_list}")
        sounder_type = list(sounder_types_list)[0]

        mode_computer = ModeComputerFactory.create_mode_computer(sounder_type)
        keyDict, mode_ids_dict = mode_computer.compute(input_files)

        # update list of non filtered dataset
        self.display_ping_time = Plotter(self.work_dir, self.data)
        self.display_ping_detection_time = Plotter2(self.work_dir, self.data)
        self.display_metadata = Metadata(self.data)
        self.modes = (keyDict, mode_ids_dict)

    def plot_surface_sound_speed(self, use_finest_level=False):
        variable = "sound_speed_at_transducer"
        title = "Sound speed at transducer"
        self.display_ping_time.plot_ping_time_variable(variable, title=title, use_finest_level=use_finest_level)

    def plot_gps_quality(self, use_finest_level=False):
        title = "Position sensor quality"
        # add a special attribute to store enum values
        label = {e.name: e.value for e in GPSQualityIndicator}
        color_palette = Category10_9

        self.display_ping_time.plot_geovariable_discrete(
            graphic_name=Key.POSITION_SENSOR_QUALITY_INDICATOR,
            variable_name=Key.POSITION_SENSOR_QUALITY_INDICATOR,
            title=title,
            labels=label,
            color_palette=color_palette,
            use_finest_level=use_finest_level,
        )

    def plot_synthetic_mode(self):
        title = "Computed mode, mode is a combination of several variables matching a set of reference curve"
        self.display_ping_time.plot_modes(
            title=title, variable_name="computed_mode", values_dict=self.modes[1], labels=self.modes[0]
        )

    def _plot_runtime_ping_mode(self, use_finest_level=False):
        if Key.PING_MODE in self.data.metadata.variable_metadata:
            title = "Kongsberg Ping Mode"
            # add a special attribute to store enum values
            label = {e.name: e.value for e in vt.KmPingMode}
            color_palette = Category10_9

            self.display_ping_time.plot_geovariable_discrete(
                graphic_name=Key.PING_MODE,
                variable_name=Key.PING_MODE,
                title=title,
                labels=label,
                color_palette=color_palette,
                use_finest_level=use_finest_level,
            )

    def _plot_runtime_pulse_length_mode(self, use_finest_level=False):
        if Key.PULSE_LENGTH_MODE in self.data.metadata.variable_metadata:
            title = "Kongsberg Pulse length Mode"
            # add a special attribute to store enum values
            label = {e.name: e.value for e in vt.KmPulseLengthMode}
            color_palette = Category10_9

            self.display_ping_time.plot_geovariable_discrete(
                graphic_name=Key.PULSE_LENGTH_MODE,
                variable_name=Key.PULSE_LENGTH_MODE,
                title=title,
                labels=label,
                color_palette=color_palette,
                use_finest_level=use_finest_level,
            )

    def plot_bsn(self, use_finest_level=False):
        self.display_ping_time.plot_geovariable_contigous(
            variable_name=Key.BACKSCATTER_NORMAL_INCIDENCE_LEVEL,
            title="Kongsberg backscatter normal incidence level per RxAntenna (max value)",
            use_finest_level=use_finest_level,
        )

    def plot_bso(self, use_finest_level=False):
        self.display_ping_time.plot_geovariable_contigous(
            variable_name=Key.BACKSCATTER_OBLIQUE_INCIDENCE_LEVEL,
            title="Kongsberg backscatter oblique incidence level per RxAntenna (max value)",
            use_finest_level=use_finest_level,
        )

    def plot_height_above_seafloor(self, use_finest_level=False):
        self.display_ping_time.plot_geovariable_contigous(
            variable_name=Key.DETECTION_Z,
            title="Mean elevation above seafloor per ping (m)",
            use_finest_level=use_finest_level,
        )

    def plot_mean_qf(self, use_finest_level=False):
        self.display_ping_time.plot_geovariable_contigous(
            variable_name=Key.DETECTION_QUALITY_FACTOR,
            title="Detection quality factor (mean value)",
            use_finest_level=use_finest_level,
        )

    def plot_mean_abs(self, use_finest_level=False):
        self.display_ping_time.plot_geovariable_contigous(
            variable_name=Key.MEAN_ABS_COEFF,
            title="Mean Absorption Coefficient (mean value)",
            use_finest_level=use_finest_level,
        )

    def plot_interping_distance(self, use_finest_level=False):
        self.display_ping_time.plot_geovariable_contigous(
            variable_name=Key.INTERPINGS_DISTANCE,
            title="Interping distance (m)",
            use_finest_level=use_finest_level,
        )

    def plot_nbswath_per_ping(self, use_finest_level=False):
        title = "Indicative number of swath per ping"
        # force min max values 2 for kongsberg, 4 for reson
        # we set max to 2, this allow to distinguish unknown, single ping or multiping values
        labels = {"Unknown": 0, "1": 1, "2 or more": 2}

        self.display_ping_time.plot_geovariable_discrete(
            graphic_name=Key.SWATH_PER_PING,
            variable_name=Key.SWATH_PER_PING,
            title=title,
            labels=labels,
            color_palette=Category10_3,
            use_finest_level=use_finest_level,
        )

    def _plot_runtime_fq_mode(self, use_finest_level=False):
        self.display_ping_time.plot_geovariable_contigous(
            variable_name=Key.FREQUENCY_MODE,
            title="Kongsberg Frequency Mode (in Hz)",
            min_value=0,
            max_value=700_000,
            use_finest_level=use_finest_level,
        )

    def plot_navigation(self):
        self.display_ping_time.plot_navigation()

    def plot_sound_speed_profiles(self, max_depth=-200):
        self.display_ping_time.plot_sound_speed_profiles(max_depth=max_depth)

    def plot_runtime_mode_infos(self, use_finest_level=False):
        self._plot_runtime_ping_mode(use_finest_level=use_finest_level)
        self._plot_runtime_pulse_length_mode(use_finest_level=use_finest_level)
        self._plot_runtime_fq_mode(use_finest_level=use_finest_level)

    def plot_tx_sector_count(self, use_finest_level=False):
        self.display_ping_time.plot_geovariable_contigous(
            variable_name=Key.TX_SECTOR_COUNT,
            title="Tx sector count",
            use_finest_level=use_finest_level,
        )

    def plot_transmit_type(self, use_finest_level=False):
        title = "Transmit type (CW/FM)"

        # we compute a view of transmit type, CW will mean that all tx sector are CW, FM will mean that at least one is FM
        # CW  = 0
        # LFM = 1 HFM=2
        # we could get a FM indicator per ping

        # initialize label and color palette
        label = {"UNK": -1, "CW": 0, "LFM": 1, "HFM": 2}
        color_palette = Category10_4

        self.display_ping_time.plot_geovariable_discrete(
            graphic_name=Key.TRANSMIT_TYPE,
            variable_name=Key.TRANSMIT_TYPE,
            title=title,
            labels=label,
            color_palette=color_palette,
            use_finest_level=use_finest_level,
        )

    def plot_wc_presence(self, use_finest_level=False):
        title = "WC presence"

        # initialize label and color palette
        label = {"UNK": -1, "No WC": 0, "WC": 1}
        color_palette = Category10_3

        self.display_ping_time.plot_geovariable_discrete(
            variable_name=Key.WC_PRESENCE,
            title=title,
            labels=label,
            color_palette=color_palette,
            use_finest_level=use_finest_level,
        )

    def plot_heading(self, use_finest_level=False):
        variable = "platform_heading"
        title = "Heading"
        self.display_ping_time.plot_ping_time_variable(variable, title=title, use_finest_level=use_finest_level)

    def plot_attitudes(self, use_finest_level=False):
        self.display_ping_time.plot_ping_time_variable(
            Key.PLATFORM_ROLL, title="Platform Roll", use_finest_level=use_finest_level
        )
        self.display_ping_time.plot_ping_time_variable(
            Key.PLATFORM_PITCH, title="Platform Pitch", use_finest_level=use_finest_level
        )

    def plot_backscatter(self):
        self.display_ping_detection_time.plot_ping_time_variable(
            Key.DETECTION_BACKSCATTER, title=Key.DETECTION_BACKSCATTER
        )

    # Reson specific
    def plot_range_selection(self, use_finest_level=False):
        if Key.RANGE_SELECTION in self.data.metadata.variable_metadata:
            self.display_ping_time.plot_geovariable_contigous(
                variable_name=Key.RANGE_SELECTION,
                title="Reson range selection (m)",
                use_finest_level=use_finest_level,
            )
        else:
            error(f"range_selection not found")

    def plot_gain_selection(self, use_finest_level=False):
        if Key.GAIN_SELECTION in self.data.metadata.variable_metadata:
            self.display_ping_time.plot_geovariable_contigous(
                variable_name=Key.GAIN_SELECTION,
                title="Reson gain selection (dB)",
                use_finest_level=use_finest_level,
            )
        else:
            error(f"gain_selection not found")

    def plot_power_selection(self, use_finest_level=False):
        if Key.POWER_SELECTION in self.data.metadata.variable_metadata:
            self.display_ping_time.plot_geovariable_contigous(
                variable_name=Key.POWER_SELECTION,
                title="Reson power selection (dB)",
                use_finest_level=use_finest_level,
            )
        else:
            error(f"power_selection not found")
