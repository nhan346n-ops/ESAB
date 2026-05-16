import numpy as np
import pandas as pd
import requests

import pyat.utils.pyat_logger as log

ISO_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class TideClient:
    def __init__(self, tide_server: str):
        """
        :param reference_file:
        :return:
        """
        self.logger = log.logging.getLogger(TideClient.__name__)
        self.tide_server = tide_server
        self.vertical_reference = None
        self.vertical_datum = None
        self.prediction_model = None

    def predict_from_nav(self, dates: pd.Series, longitudes: pd.Series, latitudes: pd.Series) -> pd.Series:
        """
        Predict tide using tide server
        """
        json_data = pd.DataFrame({"date": dates, "longitude": longitudes, "latitude": latitudes}).to_json(
            orient="records", date_format=ISO_DATE_FORMAT, double_precision=14
        )

        response = requests.post(
            self.tide_server + "/predicts",
            data=json_data,
            headers={"Content-Type": "application/json"},
            timeout=100,
        )
        if response.status_code != 200:
            response.raise_for_status()

        if "prediction_model" in response.json():
            self.prediction_model = response.json()["prediction_model"]
        if "vertical_reference" in response.json():
            self.vertical_reference = response.json()["vertical_reference"]
        if "vertical_datum" in response.json():
            self.vertical_datum = response.json()["vertical_datum"]

        return pd.DataFrame(response.json()["tide"])["tide"]

    def reference_offset_from_ellipsoid(self, longitudes: pd.Series, latitudes: pd.Series, reference: str) -> pd.Series:
        """
        Retrieve vertical offset reference from ellpsoid using tide server
        """
        json_data = pd.DataFrame({"longitude": longitudes, "latitude": latitudes}).to_json(
            orient="records", double_precision=14
        )
        response = requests.post(
            self.tide_server + f"/estimate_offset?reference={reference}",
            data=json_data,
            headers={"Content-Type": "application/json"},
            timeout=100,
        )
        if response.status_code != 200:
            response.raise_for_status()

        if "vertical_datum" in response.json():
            self.vertical_datum = response.json()["vertical_datum"]

        return pd.DataFrame(response.json()["offset"])["offset"]
