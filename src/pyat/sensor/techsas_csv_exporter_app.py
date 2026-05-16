import pathlib
from datetime import datetime

import dateutil
import numpy as np

from pytechsas.sensor.techsas_csv_exporter import TechsasCSVExporter
import pyat.utils.application_utils as app_util
import pyat.navigation.navigation_factory as navigation_factory
import pyat.utils.pyat_logger as log


class TechsasCSVExporterLauncher:
    def parse_parameter(self, **params):
        if "i_paths_dir" in params:
            self.input_files_dir = params["i_paths_dir"]
        else:
            raise Exception("Parameters i_paths_dir is missing")

        if "i_paths" in params:
            self.input_files = params["i_paths"]

        if "o_paths_dir" in params:
            self.out_files_dir = params["o_paths_dir"]
        else:
            raise Exception("Parameters o_paths_dir is missing")

        if "sensors" in params:
            self.sensors = params["sensors"]

        if "navigation_file" in params:
            self.nav_file = params["navigation_file"]
        elif "navigation_sensor" in params:
            self.nav_sensor = params["navigation_sensor"]
        else:
            raise Exception("Parameters navigation_sensor or navigation_file is missing")

        if "enable_time_interval" in params:
            self.enable_time_interval = params["enable_time_interval"]
            if self.enable_time_interval:
                self.start_date = dateutil.parser.isoparse(params["start_date"])
                self.end_date = dateutil.parser.isoparse(params["end_date"])
                if self.start_date == self.end_date:
                    self.enable_time_interval = False

        self.sampling = int(params["sampling"]) if "sampling" in params else 0

    def is_date_in_time_interval(self, file_date: datetime):
        if self.enable_time_interval:
            return self.start_date.date() <= file_date.date() <= self.end_date.date()
        else:
            return True

    def __init__(self, **params):
        self.logger = log.logging.getLogger(TechsasCSVExporterLauncher.__name__)
        self.input_files = []
        self.input_files_dir = None
        self.out_files_dir = None
        self.sensors = None
        self.nav_sensor = None
        self.nav_file = None
        self.enable_time_interval = False
        self.start_date = None
        self.end_date = None
        self.sampling = 0
        self.parse_parameter(**params)

    def __call__(self):
        aggr_time_nav = []
        aggr_latitudes = []
        aggr_longitudes = []

        # find and read navigation file
        if self.nav_file:
            self.logger.info(f"Read {self.nav_file}")
            with navigation_factory.from_file(self.nav_file) as nav_data:
                aggr_time_nav.extend(nav_data.get_times())
                aggr_latitudes.extend(nav_data.get_latitudes())
                aggr_longitudes.extend(nav_data.get_longitudes())
        else:
            self.logger.info(f"Read navigation sensor {self.nav_sensor}")
            nav_path = pathlib.Path(self.input_files_dir)
            for nav_file in nav_path.rglob("*"):
                if nav_file.is_file():
                    name_fields = nav_file.name.split("-", 4)
                    if len(name_fields) < 4:
                        continue
                    (date, time, prefix, sensor) = name_fields
                    sensor_name = prefix + "-" + sensor
                    if sensor_name == self.nav_sensor:
                        file_date = dateutil.parser.isoparse(date)
                        if not self.is_date_in_time_interval(file_date):
                            continue
                        self.logger.info(f"Read {nav_file}")
                        with navigation_factory.from_file(nav_file) as nav_data:
                            aggr_time_nav.extend(nav_data.get_times())
                            aggr_latitudes.extend(nav_data.get_latitudes())
                            aggr_longitudes.extend(nav_data.get_longitudes())

        # sort positions by time
        indexer = np.asarray(aggr_time_nav).argsort()
        aggr_time_nav = np.asarray(aggr_time_nav)[indexer]
        aggr_latitudes = np.asarray(aggr_latitudes)[indexer]
        aggr_longitudes = np.asarray(aggr_longitudes)[indexer]

        # parse input files directory
        if len(self.input_files) == 0:
            input_files_path = pathlib.Path(self.input_files_dir)
            for infile in input_files_path.rglob("*"):
                self.input_files.append(infile)

        techsas_csv_exporter_parameters = TechsasCSVExporter(
            logger=self.logger,
            input_files=self.input_files,
            out_files_dir=self.out_files_dir,
            sensors=self.sensors,
            start_date=self.start_date,
            end_date=self.end_date,
            sampling=self.sampling,
            enable_time_interval=self.enable_time_interval,
        )
        techsas_csv_exporter_parameters.parse_and_export(aggr_latitudes, aggr_longitudes, aggr_time_nav)


if __name__ == "__main__":
    app_util.launch_application(app_util.get_json_configuration_file(__file__), TechsasCSVExporterLauncher)
