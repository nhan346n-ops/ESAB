import traceback
import os.path
from abc import ABC

import pandas as pd
import pyat.utils.application_utils as app_util
import pyat.utils.pyat_logger as log
from pyat.navigation import navigation_utils, navigation_factory


class SensorCSVExporter(ABC):
    """Abstract class for parsing sensor file, georeferencing and exporting to csv"""

    def parse_parameter(self, **params):
        if "i_paths" in params:
            self.input_files = params["i_paths"]
        else:
            raise Exception("Parameters i_paths is missing")

        if "o_paths" in params:
            self.out_files = params["o_paths"]
        else:
            raise Exception("Parameters o_paths is missing")
        if "navigation_file" in params:
            self.navigation_file = params["navigation_file"]
        else:
            raise Exception("Parameter navigation_file is missing")

        self.overwrite = bool(params["overwrite"]) if "overwrite" in params else False

    def __init__(self):
        self.logger = log.logging.getLogger(SensorCSVExporter.__name__)
        self.input_files = None
        self.out_files = None
        self.navigation_file = None
        self.overwrite = False
        self.parser_func = None

    def __call__(self):
        # read navigation file
        self.logger.info(f"Read navigation file {self.navigation_file}")
        nav_data = navigation_factory.from_file(self.navigation_file)

        files_in_error = []
        # process files
        for input_file, output_file in zip(self.input_files, self.out_files):
            try:
                if os.path.exists(output_file) and not self.overwrite:
                    raise Exception(f"Output file {output_file} already exist and overwrite is set to False")
                self.logger.info(f"Read {input_file}")
                if self.parser_func is None:  # implementation error, should never be null
                    raise RuntimeError("Coding error, parser_func should not be null")
                # pylint: disable=E1102

                (time_sensor, values) = self.parser_func(input_file)
                self.logger.info(f"{input_file} {len(time_sensor)} value read")
                self.logger.info(f"Georeference dataset for {input_file} ")
                interpolated_nav = navigation_utils.interpolate(
                    navigation_data=nav_data, interpolation_time=time_sensor
                )

                # Create panda dataframe for cvs export
                final_values = {
                    "longitude": interpolated_nav.get_longitudes(),
                    "latitude": interpolated_nav.get_latitudes(),
                    "date_time": time_sensor,
                }
                for k in values.keys():
                    final_values[values[k][0]] = values[k][1]

                df = pd.DataFrame(final_values)
                self.logger.info(f"Exporting {input_file} to {output_file}")
                df.to_csv(output_file, index=False)
            except Exception as err:
                self.logger.error(f"Error occurred while processing {input_file}")
                self.logger.error(f"Error message is {err}")
                traceback.print_stack()
                files_in_error.append(f"{input_file}")

        # print final results
        if len(files_in_error) == 0:
            self.logger.info(f"{len(self.input_files)} files converted")
        else:
            self.logger.error(f"{len(files_in_error)}/{len(self.input_files)} files failed")
            self.logger.error(f"files in error {files_in_error}")


if __name__ == "__main__":
    app_util.launch_application(app_util.get_json_configuration_file(__file__), SensorCSVExporter)
