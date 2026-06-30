#! /usr/bin/env python3
# coding: utf-8
from abc import abstractmethod
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy
import scipy.signal
from pytide.core.fes2014.fes_tide_predict import FESTideComputer
from pytide.core.surface_reference.bathyelli import BathyElli
from pytide.core.surface_reference.sea_surface import SeaSurfaceProjector
from pytide.driver import tide_driver
from pytide.driver.export_tide_args import ExportTideArgs
from pytide.driver.tide_netcdf import tide_netcdf_groups
from scipy import interpolate

import pyat.utils.pyat_logger as log
from pyat.navigation import navigation_factory
from pyat.tide.tide_client import ISO_DATE_FORMAT, TideClient
from pyat.utils.exceptions.exception_list import BadParameter
from pyat.xsf.xsf_driver import XsfDriver, open_xsf


class SensorTide:
    def __init__(self, **params):
        self.logger = log.logging.getLogger(SensorTide.__name__)
        self.model_dir = None
        self.tide_server = None
        self.input_files = []
        self.output_file = None
        self.start_date = None
        self.end_date = None
        self.interval_minutes = 1  # in minutes
        self.predictive_mode = False
        self.display = False
        self.positioning_type_filter = False
        self.vertical_reference = "LAT"
        self.altitude_reference = 0.0
        self.water_level = None
        self.vertical_datum = None
        self.parse_parameter(**params)

    def parse_parameter(self, **params):
        if "input_files" in params:
            self.input_files = params["input_files"]

        if "output_file" in params:
            self.output_file = params["output_file"]
        else:
            raise BadParameter("output_file is mandatory")

        if "model_dir" in params:
            self.model_dir = params["model_dir"]
        if "tide_server" in params:
            self.tide_server = params["tide_server"]

        if "start_date" in params:
            self.start_date = pd.to_datetime(params["start_date"], utc=True).tz_convert(None)

        if "end_date" in params:
            self.end_date = pd.to_datetime(params["end_date"], utc=True).tz_convert(None)

        if "interval_minutes" in params:
            self.interval_minutes = int(params["interval_minutes"])
            if self.interval_minutes <= 0:
                raise BadParameter("interval must be a positive integer")

        if "water_level" in params:
            self.water_level = float(params["water_level"])

        if "vertical_reference" in params:
            self.vertical_reference = params["vertical_reference"]

        if "vertical_datum" in params:
            self.vertical_datum = params["vertical_datum"]

        if "altitude_reference" in params:
            self.altitude_reference = float(params["altitude_reference"])

        if "predictive_mode" in params:
            self.predictive_mode = params["predictive_mode"]

        if "positioning_type_filter" in params:
            self.positioning_type_filter = params["positioning_type_filter"]

        if "display" in params:
            self.display = params["display"]

    def compute_fes_prediction(self, longitudes: pd.Series, latitudes: pd.Series) -> pd.Series:
        dates = longitudes.axes[0]
        if self.model_dir:
            _, _, _, tide_FES_pred_sub, *_ = FESTideComputer(self.model_dir + "/fes2014").compute_tides(
                longitudes=longitudes.values,
                latitudes=latitudes.values,
                dates=dates.to_numpy(dtype="datetime64[us]"),
            )
            return pd.Series(data=tide_FES_pred_sub, index=dates)
        elif self.tide_server:
            tide_client = TideClient(self.tide_server)
            return tide_client.predict_from_nav(dates=dates, longitudes=longitudes, latitudes=latitudes)
        return pd.Series(data=np.zeros_like(dates), index=dates)

    def subsample_and_interpolate_navigation(
        self, longitudes: pd.Series, latitudes: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Subsample navigation to reduce points for external tide prediction requests and interpolate gaps.
        Returns subsampled/interpolated longitudes and latitudes indexed by the subsampled timestamps.
        """
        if longitudes is None or longitudes.empty:
            return longitudes, latitudes

        sampling = f"{max(1, self.interval_minutes)}min"
        sub_index = pd.date_range(
            start=longitudes.index[0] - pd.Timedelta(sampling),
            end=longitudes.index[-1] + pd.Timedelta(sampling),
            freq=sampling,
        )

        longitudes_sub = longitudes.reindex(index=sub_index, method="nearest", limit=1).interpolate(
            method="linear", limit_area="inside"
        )
        latitudes_sub = latitudes.reindex(index=sub_index, method="nearest", limit=1).interpolate(
            method="linear", limit_area="inside"
        )
        # extend to start_date and end_date if needed
        ext_index = pd.date_range(
            start=self.start_date - pd.Timedelta(sampling), end=self.end_date + pd.Timedelta(sampling), freq=sampling
        )
        longitudes_sub = longitudes_sub.reindex(index=ext_index, method="nearest")
        latitudes_sub = latitudes_sub.reindex(index=ext_index, method="nearest")
        return longitudes_sub, latitudes_sub

    def process(self):
        self.logger.info(f"Read input files")

        longitudesMasked, latitudesMasked, altitudesMasked = self.read_sensor_data()
        if altitudesMasked.empty:
            self.logger.info(f"No data to process")
            return

        # update dates range if needed (None or 0 in unix timestamp)
        if self.start_date is None or self.start_date.timestamp() == 0:
            self.start_date = altitudesMasked.index[0]
        if self.end_date is None or self.end_date.timestamp() == 0:
            self.end_date = altitudesMasked.index[-1]

        # get subsampled interpolated navigation for performance with tide server
        longitudesSubsampled, latitudesSubsampled = self.subsample_and_interpolate_navigation(
            longitudesMasked, latitudesMasked
        )
        self.logger.info(f"Change reference")

        if self.vertical_reference == "ZH":
            surface_ref = self.get_surface_reference_zh(longitudes=longitudesSubsampled, latitudes=latitudesSubsampled)
            if np.any(surface_ref == 0):
                self.logger.error("Data outside of BathyElli zone, falling back to WGS84")
                self.vertical_reference = "WGS84"
                self.vertical_datum = "WGS84 ellipsoid"
        elif self.vertical_reference == "LAT":
            surface_ref = self.get_surface_reference_lat(longitudes=longitudesSubsampled, latitudes=latitudesSubsampled)
        else:  # ellipsoid
            surface_ref = np.zeros_like(longitudesSubsampled.values)
            self.vertical_datum = "WGS84 ellipsoid"
            if self.altitude_reference:
                surface_ref += self.altitude_reference

        # apply waterlevel offset if provided or retrieved from xsf (positive when waterline is below ref point)
        if self.water_level is not None:
            self.logger.info(f"Apply water_level={self.water_level} offset to platform altitude")
            surface_ref += self.water_level

        # apply reference surface to altitudes
        altitudesMasked.values[:] -= interpolate.interp1d(
            x=longitudesSubsampled.index.astype(np.int64),
            y=surface_ref,
            kind="linear",
            bounds_error=False,
            fill_value=(surface_ref[0], surface_ref[-1]),
        )(x=altitudesMasked.index.astype(np.int64))

        if self.display:
            display_ref_alt_series = altitudesMasked.copy()

        if self.predictive_mode:
            self.logger.info(f"Predictive mode is active : substract tide prediction before lowpass filtering")
            # compute fes prediction
            tide_pred_sub = self.compute_fes_prediction(
                longitudes=longitudesSubsampled, latitudes=latitudesSubsampled
            ).values
            tide_pred_interpolator = interpolate.interp1d(
                x=longitudesSubsampled.index.astype(np.int64),
                y=tide_pred_sub,
                kind="linear",
                bounds_error=False,
                fill_value=(tide_pred_sub[0], tide_pred_sub[-1]),
            )
            # substract fes prediction
            altitudesMasked.values[:] -= tide_pred_interpolator(x=altitudesMasked.index.astype(np.int64))

        self.logger.info("Apply lowpass filtering")
        filtered_long, filtered_lat, filtered_alt = self._filter_and_interpolate_nav(
            longitudesMasked, latitudesMasked, altitudesMasked
        )

        filtered_time = filtered_alt.index
        if self.predictive_mode:
            self.logger.info("Predictive mode is active : reapply tide prediction after lowpass filtering")
            # add fes prediction
            filtered_alt += tide_pred_interpolator(x=filtered_time.astype(np.int64))

        # format data for export as csv
        date_series = pd.Series(data=filtered_time, name="date", index=filtered_time)
        lon_series = pd.Series(data=filtered_long, name="longitude", index=filtered_time)
        lat_series = pd.Series(data=filtered_lat, name="latitude", index=filtered_time)
        tide_series = pd.Series(data=filtered_alt, name="tide", index=filtered_time)
        df = pd.concat([date_series, lon_series, lat_series, tide_series], axis=1)
        sampling = f"{self.interval_minutes}min"
        date_resample = pd.date_range(
            start=filtered_time[0] - pd.Timedelta(sampling),
            end=filtered_time[-1] + pd.Timedelta(sampling),
            freq=sampling,
        )
        date_resample = date_resample.round(sampling)
        df_resample = df.reindex(index=date_resample, method="nearest")

        if self.output_file.endswith(tide_driver.TTB_EXTENSION) or self.output_file.endswith(
            tide_driver.TIDE_NETCDF_EXTENSION
        ):
            metadata = {}
            if self.vertical_datum:
                metadata[tide_netcdf_groups.VERTICAL_DATUM] = self.vertical_datum
            if self.vertical_reference:
                metadata[tide_netcdf_groups.VERTICAL_REFERENCE] = self.vertical_reference
            if self.altitude_reference:
                # write altitude reference to string with centimeter precision
                metadata[tide_netcdf_groups.REFERENCE_HEIGHT_ABOVE_ELLIPSOID] = f"{self.altitude_reference:.3f}"

            export_args = ExportTideArgs(
                time=df_resample["date"].to_numpy(dtype="datetime64[ns]").astype("uint64"),
                tide=df_resample["tide"].to_numpy(),
                latitude=df_resample["latitude"].to_numpy(),
                longitude=df_resample["longitude"].to_numpy(),
                o_path=self.output_file,
                overwrite=True,
                metadata=metadata,
            )
            tide_driver.exports_with_ExportTideArgs(export_args)
        else:
            df_resample.to_csv(self.output_file, sep=";", index=False, date_format=ISO_DATE_FORMAT)

        if self.display:
            # matplotlib
            plt.plot(display_ref_alt_series, ".", color="red", markersize=1)
            plt.plot(filtered_alt, color="black")
            plt.xlabel("date")
            plt.ylabel("tide")
            plt.show()

    def get_surface_reference_lat(self, longitudes: pd.Series, latitudes: pd.Series):
        self.logger.info("Changing reference surface to LAT")

        if self.model_dir:
            reference = SeaSurfaceProjector(model_dir=self.model_dir)
            ref_values = reference.get_ellipsoid_to_lat(longitudes=longitudes.values, latitudes=latitudes.values)
            self.vertical_datum = reference.get_model_name()
        elif self.tide_server:
            tide_client = TideClient(self.tide_server)
            ref_values = tide_client.reference_offset_from_ellipsoid(
                longitudes=longitudes, latitudes=latitudes, reference="LAT"
            ).values
            self.vertical_datum = tide_client.vertical_datum
        else:
            ref_values = np.zeros_like(longitudes)
        return ref_values

    def get_surface_reference_zh(self, longitudes: pd.Series, latitudes: pd.Series):
        self.logger.info("Changing reference surface to ZH")

        if self.model_dir:
            reference = BathyElli(source_directory=self.model_dir)
            ref_values = reference.get_ellipsoid_to_zerohydro(
                longitudes=longitudes.values, latitudes=latitudes.values
            ).flat[:]
            self.vertical_datum = reference.get_model_name()
        elif self.tide_server:
            tide_client = TideClient(self.tide_server)
            ref_values = tide_client.reference_offset_from_ellipsoid(
                longitudes=longitudes, latitudes=latitudes, reference="ZH"
            ).values
            self.vertical_datum = tide_client.vertical_datum
        else:
            ref_values = np.zeros_like(longitudes)
        return ref_values

    def _filter_and_interpolate_nav(
        self, longitudes: pd.Series, latitudes: pd.Series, altitudes: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        filtered_altitudes = self._resample_interpolate_and_filter_data(data=altitudes, apply_lowpass_filter=True)
        filtered_longitudes = self._resample_interpolate_and_filter_data(data=longitudes, apply_lowpass_filter=False)
        filtered_latitudes = self._resample_interpolate_and_filter_data(data=latitudes, apply_lowpass_filter=False)
        return filtered_longitudes, filtered_latitudes, filtered_altitudes

    def _resample_interpolate_and_filter_data(self, data: pd.Series, apply_lowpass_filter: bool) -> pd.Series:
        # first apply rolling window mean and interpolate
        resampled_data = self._resample_and_interpolate(data)
        # then apply lowpass filter
        if apply_lowpass_filter:
            filtered_data = _apply_low_pass_cheby2_filter(resampled_data[np.isfinite(resampled_data)])
        else:
            filtered_data = resampled_data
        # extend/crop time range to self.start_date and self.end_date if needed
        time = pd.date_range(start=self.start_date, end=self.end_date, freq="1s")
        filtered_data = filtered_data.reindex(index=time, method="nearest")
        return filtered_data

    def _resample_and_interpolate(self, data: pd.Series) -> pd.Series:
        # first apply a rolling window mean
        sampling = "1s"
        window_size = 30
        time = pd.date_range(
            start=data.index[0] - pd.Timedelta(sampling), end=data.index[-1] + pd.Timedelta(sampling), freq=sampling
        )
        # extend/crop time range to self.start_date and self.end_date if needed

        resampled_data_raw = data.reindex(index=time, method="nearest", limit=1)
        resampled_data_mean = resampled_data_raw.rolling(window=window_size, center=True).mean()

        if len(resampled_data_raw) > window_size:
            resampled_data_mean[:window_size] = (
                resampled_data_raw[:window_size].rolling(window=window_size, center=True, min_periods=10).mean()
            )
            resampled_data_mean[-window_size:] = (
                resampled_data_raw[-window_size:].rolling(window=window_size, center=True, min_periods=10).mean()
            )
        # then linear interpolation of missing data
        return resampled_data_mean.interpolate(method="linear", limit_area="inside")

    @abstractmethod
    def read_sensor_data(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        return raw longitudes, latitudes, altitudes as Series with masked data
        """


class XsfTide(SensorTide):
    def __init__(self, **params):
        super().__init__(**params)
        self.logger = log.logging.getLogger(XsfTide.__name__)
        self.global_df = None

    def __call__(self):
        self.process()

    def read_sensor_data(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        implementation of SensorTide abstract method
        """
        for input_file in self.input_files:
            self.read_xsf_data(input_file)

        # sort
        self.global_df.sort_index(inplace=True)
        # remove duplicates
        self.global_df = self.global_df.drop_duplicates("date")

        quality = self.global_df["quality"]

        # uncomment to see unfiltered raw data with colored quality on plot
        # if self.display:
        #     self.global_df.plot.scatter(x="date", y="altitude", c=quality, cmap="viridis", marker=".")

        gps_quality_mask = quality > 2 if self.positioning_type_filter else quality != 0
        latitudes_masked = self.global_df["latitude"][gps_quality_mask]
        longitudes_masked = self.global_df["longitude"][gps_quality_mask]
        altitudes_masked = self.global_df["altitude"][gps_quality_mask]
        return longitudes_masked, latitudes_masked, altitudes_masked

    def read_xsf_data(self, input_file: str):
        """Read a xsf file,
        :return a time variable, and a list of tuple containing the name of a column and the associated data
        """
        with open_xsf(input_file) as i_xsf_driver:
            if i_xsf_driver.get_version() < 0.7:
                df = self._evaluate_nmea(i_xsf_driver)
            else:
                df = self._evaluate_xsf(i_xsf_driver)
            self.global_df = pd.concat(objs=[df, self.global_df], sort=False)
            if self.water_level is None:
                self.water_level = i_xsf_driver.get_waterlevel()

    def _evaluate_nmea(self, i_sounder_driver: XsfDriver):
        """
        Use the spatial resolution in meter as distance from one point of the navigation to evaluate the resolution in degree
        """
        times = i_sounder_driver.read_position_times()
        lons = i_sounder_driver.read_position_longitudes()
        lats = i_sounder_driver.read_position_latitudes()
        sensor_position_offset = i_sounder_driver.read_position_offset()

        nmeas = i_sounder_driver.read_position_nmea()

        # Field Number:
        #   1) Universal Time Coordinated (UTC)
        #   2) Latitude
        #   3) N or S (North or South)
        #   4) Longitude
        #   5) E or W (East or West)
        #   6) GPS Quality Indicator,
        #      0 - fix not available,
        #      1 - GPS fix,
        #      2 - Differential GPS fix
        #      (values above 2 are 2.3 features)
        #      3 = PPS fix
        #      4 = Real Time Kinematic
        #      5 = Float RTK
        #      6 = estimated (dead reckoning)
        #      7 = Manual input mode
        #      8 = Simulation mode
        #   7) Number of satellites in view, 00 - 12
        #   8) Horizontal Dilution of precision (meters)
        #   9) Antenna Altitude above/below mean-sea-level (geoid) (in meters)
        #  10) Units of antenna altitude, meters
        #  11) Geoidal separation, the difference between the WGS-84 earth
        #      ellipsoid and mean-sea-level (geoid), "-" means mean-sea-level
        #      below ellipsoid
        #  12) Units of geoidal separation, meters
        #  13) Age of differential GPS data, time in seconds since last SC104
        #      type 1 or 9 update, null field when DGPS is not used
        #  14) Differential reference station ID, 0000-1023
        #  15) Checksum
        gps_quality_index = 6
        antenna_altitude_index = 9
        geoidal_separation_index = 11

        self.logger.info(f"Evaluation of the nmea")

        qualities = []
        altitudes = []
        for nmea in nmeas:
            fields = nmea.split(",")
            if len(fields) > geoidal_separation_index:
                qualities.append(float(fields[gps_quality_index]))
                altitude = float(fields[antenna_altitude_index])
                geoidal_sep = float(fields[geoidal_separation_index]) if fields[geoidal_separation_index] else 0.0
                altitudes.append(altitude + geoidal_sep + sensor_position_offset[-1])
            else:
                qualities.append(float(0))
                altitudes.append(float(0))

        df = pd.DataFrame(
            data={"date": times, "longitude": lons, "latitude": lats, "quality": qualities, "altitude": altitudes},
            index=times,
        )

        return df

    def _evaluate_xsf(self, i_sounder_driver: XsfDriver):
        """
        Use the spatial resolution in meter as distance from one point of the navigation to evaluate the resolution in degree
        """
        times = i_sounder_driver.read_position_times()
        lons = i_sounder_driver.read_position_longitudes()
        lats = i_sounder_driver.read_position_latitudes()
        altitudes = i_sounder_driver.read_position_height_above_ellipsoid()
        qualities = i_sounder_driver.read_position_sensor_quality_indicators()

        df = pd.DataFrame(
            data={
                "date": times,
                "longitude": lons,
                "latitude": lats,
                "quality": qualities,
                "altitude": altitudes,
            },
            index=times,
        )

        return df


class NavTide(SensorTide):
    def __init__(self, **params):
        super().__init__(**params)
        self.logger = log.logging.getLogger(NavTide.__name__)

    def __call__(self):
        self.process()

    def read_sensor_data(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        implementation of SensorTide abstract method
        """
        global_df = None

        # read navigation
        for input_file in self.input_files:
            self.logger.info(f"Read {input_file}")
            with navigation_factory.from_file(input_file) as nav_data:
                df = pd.DataFrame(
                    data={
                        "date": nav_data.get_times(),
                        "longitude": nav_data.get_longitudes(),
                        "latitude": nav_data.get_latitudes(),
                        "altitude": nav_data.get_altitudes(),
                        "quality": nav_data.get_sensor_quality_indicators(),
                    },
                    index=nav_data.get_times(),
                )
                global_df = pd.concat(objs=[df, global_df], sort=False)

        # sort
        global_df.sort_index(inplace=True)
        # remove duplicates
        global_df = global_df.drop_duplicates("date")

        if "quality" in global_df:
            quality = global_df["quality"]
        else:
            quality = None
        has_quality_data = quality is not None and quality.any()

        # uncomment to see unfiltered raw data with colored quality on plot
        # if self.display:
        #     if has_quality_data:
        #         global_df.plot.scatter(x="date", y="altitude", c=quality, cmap="viridis", marker=".")
        #     else:
        #         global_df.plot.scatter(x="date", y="altitude", color="red", marker=".")

        if self.positioning_type_filter and has_quality_data:
            quality_mask = quality > 2  # quality, positioning type better than differential GPS
            altitudes_masked = global_df["altitude"][quality_mask]
            longitudes_masked = global_df["longitude"][quality_mask]
            latitudes_masked = global_df["latitude"][quality_mask]
        else:
            altitudes_masked = global_df["altitude"][:]
            longitudes_masked = global_df["longitude"][:]
            latitudes_masked = global_df["latitude"][:]

        return longitudes_masked, latitudes_masked, altitudes_masked


def _apply_low_pass_cheby2_filter(data: pd.Series) -> pd.Series:
    order = 2
    attenuation_cut = 40
    frequency_cut = 1 / 300  # (-40dB point)
    frequency_sampling = 1  # 1Hz

    b, a = scipy.signal.cheby2(order, attenuation_cut, frequency_cut, "lowpass", fs=frequency_sampling, analog=False)
    filtered_data = scipy.signal.filtfilt(b=b, a=a, x=data, method="gust")
    return pd.Series(data=filtered_data, index=data.axes[0])


def _apply_low_pass_butter_filter(data: pd.Series) -> pd.Series:
    order = 4
    frequency_cut = 1 / 300  # Hz (-3dB point)
    frequency_sampling = 1  # 1Hz

    # Prepare filter
    b, a = scipy.signal.butter(order, frequency_cut, "lowpass", fs=frequency_sampling, analog=False)
    # Apply filter
    filtered_data = scipy.signal.filtfilt(b=b, a=a, x=data)
    return pd.Series(data=filtered_data, index=data.axes[0])
