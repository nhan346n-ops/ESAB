import os
import tempfile
from typing import List

import numpy as np
import pandas as pd
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from pytide.core.fes2014.fes_tide_predict import FESTideComputer
from pytide.core.shom_harmonic.shom_harmonic_tide_predict import ShomTidePredictor
from pytide.core.surface_reference.sea_surface import SeaSurfaceProjector
from pytide.driver import tide_driver
from pytide.driver.export_tide_args import ExportTideArgs
from pytide.driver.tide_netcdf import tide_netcdf_groups

import pyat.utils.pyat_logger as log
from pyat.navigation import navigation_factory
from pyat.navigation.abstract_navigation import AbstractNavigation
from pyat.tide.tide_client import ISO_DATE_FORMAT, TideClient
from pyat.utils.exceptions.exception_list import BadParameter


class TidePredictor:
    def __init__(self):
        self.prediction_model = None
        self.vertical_reference = None
        self.vertical_datum = None

    # pylint:disable=unused-argument
    def update_surface_reference(self, longitudes: np.ndarray, latitudes: np.ndarray, tides: np.ndarray):
        """Update surface reference to switch to LAT or MSL"""
        self.vertical_reference = tide_netcdf_groups.VerticalReference.LAT.value
        return tides

    def process(self, data_file, predictor, output_file, monitor):
        # monitor come here with a tick count of 10
        sub = monitor.split(10)
        sub.begin_task(name="Reading input dataset", n=1)

        ds = pd.read_csv(data_file, sep=";", names=["date", "longitude", "latitude"], skiprows=1)

        _dates = pd.to_datetime(ds["date"], utc=True).to_numpy(dtype="datetime64[us]")
        sub.done()
        sub = monitor.split(60)
        _lats, _lons, _dates, _computed_tides, *_ = predictor.compute_tides(
            latitudes=ds["latitude"].astype("float64").to_numpy(),
            longitudes=ds["longitude"].astype("float64").to_numpy(),
            dates=_dates.astype("datetime64[s]"),
        )
        self.prediction_model = predictor.get_prediction_model()

        sub.done()
        sub = monitor.split(20)
        sub.begin_task(name="Changing reference surface from MSL to LAT", n=1)
        _tides_ref_lat = self.update_surface_reference(longitudes=_lons, latitudes=_lats, tides=_computed_tides)

        sub.done()
        sub = monitor.split(10)

        sub.begin_task(name="Exporting results", n=1)

        # write to file in TTB or CSV format
        if output_file.endswith(".ttb") or output_file.endswith(".tide.nc"):
            metadata = {}
            if self.prediction_model:
                metadata[tide_netcdf_groups.PREDICTION_MODEL] = self.prediction_model
            if self.vertical_reference:
                metadata[tide_netcdf_groups.VERTICAL_REFERENCE] = self.vertical_reference
            if self.vertical_datum:
                metadata[tide_netcdf_groups.VERTICAL_DATUM] = self.vertical_datum

            export_args = ExportTideArgs(
                time=_dates.astype("datetime64[ns]").astype("uint64"),
                tide=_tides_ref_lat,
                latitude=_lats,
                longitude=_lons,
                o_path=output_file,
                overwrite=True,
                metadata=metadata,
            )
            tide_driver.exports_with_ExportTideArgs(export_args)
        else:
            date_serie = pd.Series(data=_dates, name="date")
            lat_serie = pd.Series(data=_lats, name="latitude")
            lon_serie = pd.Series(data=_lons, name="longitude")
            tide_serie = pd.Series(data=_tides_ref_lat, name="tide")
            df = pd.concat([date_serie, lon_serie, lat_serie, tide_serie], axis=1)
            df.to_csv(output_file, sep=";", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")

        sub.done()
        monitor.done()


class FESTide(TidePredictor):
    def update_surface_reference(self, longitudes: np.ndarray, latitudes: np.ndarray, tides: np.ndarray):
        proj = SeaSurfaceProjector(model_dir=self.model_dir)
        self.vertical_reference = tide_netcdf_groups.VerticalReference.LAT.value
        self.vertical_datum = proj.get_model_name()
        return proj.project_msl_to_lat(longitudes=longitudes, latitudes=latitudes, values=tides)

    def __init__(self, **kwargs):
        """
        :param target_file:
        :param reference_file:
        :return:
        """
        super().__init__()
        self.logger = log.logging.getLogger(FESTide.__name__)
        if "model_dir" in kwargs:
            self.model_dir = kwargs["model_dir"]
            self.fes_model_dir = os.path.join(self.model_dir, "fes2014")
            if not os.path.exists(self.fes_model_dir):
                raise BadParameter(f"fes directory does not exist ({self.fes_model_dir})")
        elif "tide_server" in kwargs:
            self.tide_server = kwargs["tide_server"]
        else:
            raise BadParameter("model directory path or tide serve url is mandatory")

        if "data_file" in kwargs:
            self.data_file = kwargs["data_file"]
        else:
            raise BadParameter("data_file file parameter is mandatory")

        if "output_file" in kwargs:
            self.output_file = kwargs["output_file"]
        else:
            raise BadParameter("output file parameter is mandatory")

        if "monitor" in kwargs:
            self.monitor = kwargs["monitor"]
        else:
            self.monitor = DefaultMonitor

    def __call__(self):
        self.logger.info(f"Starting fes model tide prediction for {self.data_file} dataset")
        # initialize tide predictor
        predictor = FESTideComputer(model_dir=self.fes_model_dir)
        self.logger.info(f"Parsing input file {self.data_file}")

        # start the processing
        self.process(self.data_file, predictor, self.output_file, self.monitor)

        self.logger.info(f"End of tide prediction result written to file {self.output_file}")


class ShomTide(TidePredictor):
    def update_surface_reference(self, longitudes: np.ndarray, latitudes: np.ndarray, tides: np.ndarray):
        self.vertical_reference = tide_netcdf_groups.VerticalReference.LAT.value
        return tides

    def __init__(self, **kwargs):
        """
        :param target_file:
        :param reference_file:
        :return:
        """
        super().__init__()
        self.logger = log.logging.getLogger(ShomTide.__name__)
        if "harmonic_file" in kwargs:
            self.harmonic_file = kwargs["harmonic_file"]
        else:
            raise BadParameter("harmonic_file file path is mandatory")

        if "data_file" in kwargs:
            self.data_file = kwargs["data_file"]
        else:
            raise BadParameter("data_file file parameter is mandatory")
        if "output_file" in kwargs:
            self.output_file = kwargs["output_file"]
        else:
            raise BadParameter("output file parameter is mandatory")
        if "monitor" in kwargs:
            self.monitor = kwargs["monitor"]
        else:
            self.monitor = DefaultMonitor

    def __call__(self):
        self.logger.info(
            f"Starting harmonic tide prediction for {self.data_file} dataset with {self.harmonic_file} harmonic file "
        )
        predictor = ShomTidePredictor(harmonic_file=self.harmonic_file)
        self.logger.info(f"Parsing input file {self.data_file}")

        self.process(self.data_file, predictor, self.output_file, self.monitor)

        self.logger.info(f"End harmonic tide prediction result written to file {self.output_file}")


class FESTidePredictorForNavigationFiles:
    def __init__(
        self,
        input_files: List[str],
        output_files: List[str],
        model_dir: str | None = None,
        tide_server: str | None = None,
        overwrite: bool = False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        """
        Initialize the FES tide prediction for navigation files

        :param input_files: list of input navigation files
        :param output_file: resulting tide estimation. File will be a TTB file
        :param model_dir: directory containing the FES2014 model
        :param tide_server: server to use for tide prediction
        :param overwrite: overwrite output file if it exists
        """
        # The file will be a csv file with columns date, longitude, latitude, tide
        self.logger = log.logging.getLogger(FESTidePredictorForNavigationFiles.__name__)
        self.input_files = input_files
        self.output_files = output_files
        self.overwrite = overwrite
        self.monitor = monitor

        self.model_dir = model_dir
        self.tide_server = tide_server
        if self.model_dir is not None:
            self.logger.info(f"Using model directory {self.model_dir}")
            self.tide_server = None
        elif tide_server is not None:
            self.logger.info(f"Using tide server {tide_server}")
            self.model_dir = None
        else:
            raise BadParameter("model_dir or tide_server parameter is mandatory")

    def __call__(self):
        """
        Load navigation files and estimate tide
        """
        self.monitor.begin_task(name="Estimating tide for input files", n=len(self.input_files))
        with tempfile.TemporaryDirectory() as temp_dir:
            for input_file, output_file in zip(self.input_files, self.output_files):
                sub_monitor = self.monitor.split(1)
                sub_monitor.begin_task(name=f"Processing file {input_file}", n=100)

                if not self.overwrite and os.path.exists(output_file):
                    self.logger.warning("File %s already exists, and can not be overwritten.", output_file)
                else:
                    with navigation_factory.from_file(input_file) as nav_data:
                        nav_data_frame = self._load_data_in_DataFrame(nav_data)
                        # initialize and call the tide predictor
                        if self.model_dir is not None:
                            self._predict_with_local_tide_model(nav_data_frame, output_file, temp_dir, sub_monitor)
                        elif self.tide_server is not None:
                            self._predict_with_tide_server(nav_data_frame, output_file)
                            sub_monitor.done()

            self.logger.info(f"End of tide prediction result written to file {self.output_files}")
        self.monitor.done()

    def _predict_with_tide_server(self, nav_data: pd.DataFrame, output_file: str) -> None:
        """
        Predict tide using tide server
        """
        dates = nav_data["date"]
        longitudes = nav_data["longitude"]
        latitudes = nav_data["latitude"]
        tide_client = TideClient(self.tide_server)
        tides = tide_client.predict_from_nav(dates=dates, longitudes=longitudes, latitudes=latitudes)

        metadata = {}
        if tide_client.prediction_model:
            metadata[tide_netcdf_groups.PREDICTION_MODEL] = tide_client.prediction_model
        if tide_client.vertical_reference:
            metadata[tide_netcdf_groups.VERTICAL_REFERENCE] = tide_client.vertical_reference
        if tide_client.vertical_datum:
            metadata[tide_netcdf_groups.VERTICAL_DATUM] = tide_client.vertical_datum

        export_args = ExportTideArgs(
            time=dates.to_numpy().astype("datetime64[ns]").astype("uint64"),
            tide=tides.to_numpy(),
            latitude=latitudes.to_numpy(),
            longitude=longitudes.to_numpy(),
            o_path=output_file,
            overwrite=True,
            metadata=metadata,
        )
        tide_driver.exports_with_ExportTideArgs(export_args)

    def _predict_with_local_tide_model(
        self, nav_data_frame: pd.DataFrame, output_file: str, temp_dir, monitor: ProgressMonitor
    ) -> None:
        """
        Predict tide using local tide model
        """
        csv_file = os.path.join(temp_dir, "tide_estimation.csv")
        nav_data_frame.to_csv(
            csv_file, sep=";", columns=["date", "longitude", "latitude"], index=False, date_format=ISO_DATE_FORMAT
        )
        FESTide(data_file=csv_file, output_file=output_file, model_dir=self.model_dir, monitor=monitor)()

    def _load_data_in_DataFrame(self, abs_nav_data: AbstractNavigation) -> pd.DataFrame:
        """
        Build a DataFrame from navigation data
        """
        time_navigation = abs_nav_data.get_times()

        start_date = time_navigation[0] - np.timedelta64(1, "h")
        end_date = time_navigation[-1] + np.timedelta64(1, "h")
        interpolation_time = np.arange(start_date, end_date, np.timedelta64(1, "m"))

        # We need to change time to float in order to be able to interpolate data
        time_navigation_float = (time_navigation - start_date) / np.timedelta64(1, "s")
        time_sensor_float = (interpolation_time - start_date) / np.timedelta64(1, "s")

        # Interpolate longitudes and latitudes
        longitudes = np.interp(time_sensor_float, time_navigation_float, abs_nav_data.get_longitudes())
        latitudes = np.interp(time_sensor_float, time_navigation_float, abs_nav_data.get_latitudes())
        # Check if longitudes are between -180 and 180
        longitudes = np.where(longitudes < -180.0, 360.0 + longitudes, longitudes)
        longitudes = np.where(longitudes > 180.0, longitudes - 360.0, longitudes)

        return pd.DataFrame(
            {
                "date": interpolation_time,
                "longitude": longitudes,
                "latitude": latitudes,
            }
        )


if __name__ == "__main__":
    estimator = FESTidePredictorForNavigationFiles(
        input_files=[r"E:\temp\tide\0132_20120607_070030_ShipName.xsf.nc"],
        output_files=[r"E:\temp\tide\python_prediction.ttb"],
        model_dir=r"E:\ifremer\data\tide_models_lite\datasets\tide",
        # tide_server="http://localhost:4400",
        overwrite=True,
    )
    estimator()
