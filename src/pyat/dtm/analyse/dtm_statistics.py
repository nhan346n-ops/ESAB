#! /usr/bin/env python3
# coding: utf-8

import os
from enum import IntEnum, auto
from io import TextIOWrapper
from os import PathLike, path
from pathlib import Path
from typing import Dict, List, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.utils.pyat_logger as log

__logger = log.logging.getLogger("calculating_statistics")


class Metrics(NamedTuple):
    """
    Tuple grouping all computed values for one layer
    """

    dtm_name: str
    layer: str
    min: float = np.nan
    max: float = np.nan
    mean: float = np.nan
    median: float = np.nan
    std: float = np.nan
    min_confidence_interval: float = np.nan
    max_confidence_interval: float = np.nan
    confidence_level_on_interval: float = np.nan
    confidence_level_1_sigma: float = np.nan
    confidence_level_2_sigma: float = np.nan


class AllMetrics:
    """
    Class aggregating all calculated metrics
    """

    def __init__(self) -> None:
        self._metrics: List[Metrics] = []

    def append(self, metrics: Metrics):
        """Addins one metrics to the list"""
        self._metrics.append(metrics)

    @property
    def metrics(self) -> List[Metrics]:
        return self._metrics

    def to_dict(self):
        """Translate this instance to a list"""
        return {
            "file": [metrics.dtm_name for metrics in self._metrics],
            "layer": [metrics.layer for metrics in self._metrics],
            "min": [metrics.min for metrics in self._metrics],
            "max": [metrics.max for metrics in self._metrics],
            "mean": [metrics.mean for metrics in self._metrics],
            "median": [metrics.median for metrics in self._metrics],
            "std": [metrics.std for metrics in self._metrics],
            "min_confidence_interval": [metrics.min_confidence_interval for metrics in self._metrics],
            "max_confidence_interval": [metrics.max_confidence_interval for metrics in self._metrics],
            "confidence_level_on_interval": [metrics.confidence_level_on_interval for metrics in self._metrics],
            "confidence_level_1_sigma": [metrics.confidence_level_1_sigma for metrics in self._metrics],
            "confidence_level_2_sigma": [metrics.confidence_level_2_sigma for metrics in self._metrics],
        }

    def is_empty(self) -> bool:
        """True when no metrics was calculated"""
        return len(self._metrics) == 0


class ScopeArg(IntEnum):
    "Scope of computations"

    PER_FILE = auto()  # one result per data file
    GLOBAL = auto()  # one result for all


class StatArgs(NamedTuple):
    """
    Class representing all arguments for configuring the process
    See pyat/app/emodnet/conf/calculating_statistics.json for more details
    """

    i_paths: List[PathLike]
    output_dir: PathLike | None = None
    scope: ScopeArg = ScopeArg.PER_FILE
    layers: List[str] | None = DtmConstants.LAYERS
    min_valid_sounds: float | None = None
    histogram_bins: int = 10
    histogram_stat: str = "count"
    show_histogram: bool = False
    confidence_level: float | None = None
    confidence_interval_min: float | None = None
    confidence_interval_max: float | None = None
    confidence_interval_1_sigma: bool = True
    confidence_interval_2_sigma: bool = True
    monitor: ProgressMonitor = DefaultMonitor


def computes(**kwargs) -> None:
    """
    Function accepting all arguments of the process as a dict. Possible arguments are listed in "StatArgs" class
    """
    kwargs["scope"] = (
        ScopeArg.GLOBAL if "scope" in kwargs and kwargs["scope"].lower() == "global" else ScopeArg.PER_FILE
    )

    output_dir = Path.home()
    if "output_dir" in kwargs:
        output_dir = kwargs["output_dir"]
        if isinstance(output_dir, list):
            output_dir = output_dir[0]
        kwargs["output_dir"] = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    computes_with_statArgs(StatArgs(**kwargs))


def computes_with_statArgs(stat_args: StatArgs) -> AllMetrics:
    """
    Main function

    Calculation of statistical data on the different layers of one or more DTMs.

    returns the resulting statistics are grouped in a AllMetrics instance
    """
    if not stat_args.i_paths:
        raise ValueError(f"Argument i_paths can not be empty")
    if not stat_args.layers:
        raise ValueError(f"Argument layers can not be empty")

    all_metrics = AllMetrics()
    if stat_args.output_dir is not None:
        # when output_dir is present, generates image of chart and CSV report
        csv_path = Path(stat_args.output_dir, "metrics.csv")
        try:
            with open(csv_path, mode="wb") as csv_file:
                _computes_metrics_on_layers(stat_args, all_metrics)
                # Generates Csv file and report
                if not all_metrics.is_empty():
                    dataFrame = _prepare_report(stat_args, all_metrics)
                    _write_csv(csv_file, dataFrame)
                    _log_report(dataFrame)

        except PermissionError as e:
            __logger.error(str(e))
    else:
        _computes_metrics_on_layers(stat_args, all_metrics)
        # Generates the report
        if not all_metrics.is_empty():
            dataFrame = _prepare_report(stat_args, all_metrics)
            _log_report(dataFrame)

    if stat_args.show_histogram:
        plt.show(block=True)

    return all_metrics


def _computes_metrics_on_layers(stat_args: StatArgs, all_metrics: AllMetrics) -> None:
    """
    Browse all layers and compute statistics a Metrics for all of them.
    """
    for layer in stat_args.layers:
        __logger.info(f"Processing {layer}")
        dataframes = _read_dtm_as_dataframe(layer, stat_args)
        if len(dataframes.items()) == 0:
            __logger.warning(f"Layer not found in input file(s) : {layer.title()}, statistic not computable.")
            continue

        for dtm_name, dataframe in dataframes.items():
            # computes statistics
            metrics = _computes_statistics(dataframe[layer], dtm_name, layer, stat_args)
            all_metrics.append(metrics)

            if stat_args.histogram_bins > 0 and (stat_args.output_dir is not None or stat_args.show_histogram):
                try:
                    _plots_statistics(dtm_name, layer, dataframe, stat_args)
                except Exception as exc:
                    __logger.warning(f"Error while ploting stats : {str(exc)}.")


def _read_dtm_as_dataframe(layer: str, stat_args: StatArgs) -> Dict[str, pd.DataFrame]:
    """
    Read and aggregates all data of a specific layer from all DTM in i_paths list
    """
    dataframes = {}
    for i_path in stat_args.i_paths:
        with dtm_driver.open_dtm(i_path) as i_driver:
            if layer in i_driver:
                layer_data = i_driver[layer][:]

                # Applying mask if any
                if (
                    stat_args.min_valid_sounds
                    and stat_args.min_valid_sounds > 0
                    and DtmConstants.VALUE_COUNT in i_driver
                ):
                    value_count = i_driver[DtmConstants.VALUE_COUNT][:]
                    layer_data = layer_data[value_count >= stat_args.min_valid_sounds]

                # Convert to 1D DataFrame
                dtm_name = path.basename(i_path)
                dataframes[dtm_name] = pd.DataFrame(data={layer: layer_data.flatten(), "dtm_name": dtm_name})

    # Merge the dataframes in case of GLOBAL computation
    if stat_args.scope == ScopeArg.GLOBAL:
        dataframes = {"": pd.concat(list(dataframes.values()), ignore_index=True)}
    return dataframes


def _computes_statistics(series: pd.Series, dtm_name: str, layer: str, stat_args: StatArgs) -> Metrics:
    """
    Calcultates all statistics of the specified Series.
    Return the resulting Metrics
    """
    stat_min, stat_max, mean, median, std = series.agg(["min", "max", "mean", "median", "std"])

    # Confidence level. Not useful for elevations
    if DtmConstants.ELEVATION_NAME in layer:
        return Metrics(dtm_name, layer, stat_min, stat_max, mean, median, std)

    min_confidence_interval, max_confidence_interval = np.nan, np.nan
    confidence_level_on_interval = np.nan
    if stat_args.confidence_interval_min is not None and stat_args.confidence_interval_max is not None:
        # Confidence level
        min_confidence_interval = stat_args.confidence_interval_min
        max_confidence_interval = stat_args.confidence_interval_max
        confidence_level_on_interval = _computes_confidence_level_on_interval(
            series,
            stat_args.confidence_interval_min,
            stat_args.confidence_interval_max,
        )
    elif stat_args.confidence_level is not None:
        # Confidence interval
        confidence_level_on_interval = stat_args.confidence_level
        min_confidence_level = (100.0 - stat_args.confidence_level) / 200.0
        min_confidence_interval = series.quantile(q=min_confidence_level)
        max_confidence_interval = series.quantile(q=1.0 - min_confidence_level)

    confidence_level_1_sigma = np.nan
    if stat_args.confidence_interval_1_sigma:
        # Confidence level, 1 sigma
        confidence_level_1_sigma = _computes_confidence_sigma(series, 1.0, mean, std)

    confidence_level_2_sigma = np.nan
    if stat_args.confidence_interval_2_sigma:
        # Confidence level, 2 sigma
        confidence_level_2_sigma = _computes_confidence_sigma(series, 2.0, mean, std)

    return Metrics(
        dtm_name,
        layer,
        stat_min,
        stat_max,
        mean,
        median,
        std,
        min_confidence_interval,
        max_confidence_interval,
        confidence_level_on_interval,
        confidence_level_1_sigma,
        confidence_level_2_sigma,
    )


def _plots_statistics(dtm_name: str, layer: str, values: pd.DataFrame, stat_args: StatArgs):
    """
    Generates an histogram
    """
    hue = hue = "dtm_name" if stat_args.scope == ScopeArg.GLOBAL else None
    plt.figure()
    sns.histplot(values, x=layer, stat=stat_args.histogram_stat, hue=hue, bins=int(stat_args.histogram_bins))
    file_name = f"{layer}.png" if stat_args.scope == ScopeArg.GLOBAL else f"{dtm_name}_{layer}.png"
    if stat_args.output_dir is not None:
        plt.savefig(Path(stat_args.output_dir, file_name), dpi=600)
    if not stat_args.show_histogram:
        plt.close()


def _computes_confidence_level_on_interval(
    values: pd.Series, confidence_interval_min: float, confidence_interval_max: float
) -> float:
    """
    Computes confidence level for the specified interval
    """
    nb_values_in_range = values.between(confidence_interval_min, confidence_interval_max).value_counts()
    return (nb_values_in_range[True] / values.count()) * 100 if True in nb_values_in_range else 0.0


def _computes_confidence_sigma(values: pd.Series, sigma: float, mean: float, std: float) -> float:
    """
    Computes confidence level for an interval mean +/- sigma
    """
    return _computes_confidence_level_on_interval(values, mean - sigma * std, mean + sigma * std)


def _prepare_report(stat_args: StatArgs, all_metrics: AllMetrics) -> pd.DataFrame:
    """Merge all metrics in a DataFrame"""
    dict_of_metrics = all_metrics.to_dict()
    if stat_args.scope == ScopeArg.GLOBAL:
        del dict_of_metrics["file"]
    return pd.DataFrame(dict_of_metrics)


def _write_csv(csv_file: TextIOWrapper, metrics_frame: pd.DataFrame) -> None:
    """Export metrics in a CSV file"""
    metrics_frame.to_csv(csv_file, sep=";", index=False)


def _log_report(metrics_frame: pd.DataFrame) -> None:
    """Export metrics in logs"""
    report = metrics_frame.to_string(index=False, float_format="%.2f", na_rep="---")
    for line in report.splitlines():
        __logger.info(line)
