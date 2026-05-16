import pandas as pd

import pyat.utils.application_utils as app_util
from pyat.sensor.sensor_cvs_exporter_app import SensorCSVExporter


class CSVExporter(SensorCSVExporter):
    def __init__(self, **params):
        super().__init__()
        self.parse_parameter(**params)
        self.parser_func = self.read_SB21_NMEA

    def read_SB21_NMEA(self, file: str):
        df = pd.read_csv(
            file,
            delimiter=",",
            header=None,
        )
        df = df.rename(
            columns={
                0: "code",
                1: "date",
                2: "time",
                3: "sensor",
                4: "state of time sampling",
                5: "pressure",
                6: "water conductivity",
                7: "Intake temperature (temp de la cuve)",
                8: "water salinity",
                9: "water mass density",
                10: "water sound speed",
                11: "temperature a la prise d eau",
            }
        )
        df["date_time"] = df["date"] + " " + df["time"]
        df["date_time"] = pd.to_datetime(df["date_time"], format="%d/%m/%y %H:%M:%S.%f")
        df = df.drop(columns=["date", "time", "sensor", "code", 12, 13])
        time_sensor = df["date_time"].to_numpy()
        dictionary = {}
        for k in df.keys():
            if k != "date_time":
                dictionary[k] = (k, df[k].to_numpy())
        return time_sensor, dictionary


if __name__ == "__main__":
    app_util.launch_application(app_util.get_json_configuration_file(__file__), CSVExporter)
