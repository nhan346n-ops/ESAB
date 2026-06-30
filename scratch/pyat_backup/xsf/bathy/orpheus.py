"""


see https://gitlab.ifremer.fr/fleet/pyat/-/issues/16


"""

import os
from enum import Enum
from functools import partial
from logging import Logger
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor

import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.utils.argument_utils as arg_utils
from pyat.dtm import dtm_driver
from pyat.sounder import sounder_driver_factory
from pyat.sounder.sounder_driver import SounderDriver
from pyat.utils import pyat_logger, numpy_utils
from pyat.utils.argument_utils import Geobox
from pyat.utils.iho.vertical_uncertainty import SurveyOrder, maximum_avu_for_order
from pyat.utils.path_utils import basename_of_fname

try:
    import seaborn as sns
except ImportError:
    sns = None

PLOT_DPI = 600

CONFIDENCE_LEVEL_FUNC = "confidence level"

plt.rcParams["toolbar"] = "toolmanager"


class ToogleTvu(ToolToggleBase):
    """Tool to toogle the visibility of one TVU value."""

    def __init__(self, *args, line: Line2D, text: Text, **kwargs):
        self.line = line
        self.text = text
        super().__init__(*args, **kwargs)

    # pylint: disable=unused-argument
    def enable(self, *args):
        self.setVisible(visible=True)

    # pylint: disable=unused-argument
    def disable(self, *args):
        self.setVisible(visible=False)

    def setVisible(self, visible: bool):
        self.line.set(visible=visible)
        self.text.set(visible=visible)
        self.canvas.draw()


class Scope(Enum):
    "Scope of computations"

    PER_FILE = 0  # one result per data file
    GLOBAL = 1  # one result for all


class SounderCriteria(Enum):
    DETECTION_TYPE = 0
    EMISSION_SECTOR = 1
    EMISSION_PLAN = 2
    PING_FREQUENCY = 3
    NOT = 4


class VizX(Enum):
    BEAM = 0
    ANGLE = 1


class VizY(Enum):
    BIAS = 0  # mean of Observed - reference and its stdev
    QUALITY_FACTOR = 1  # Quality Factor


class UnitY(Enum):
    METER, PERCENT = 0, 1


def get_enum_from_name(cls: Type[Enum], name: Union[str, Enum, None], default=None, optional: bool = False) -> Enum:
    "Tries hard to find the enum value given as `name` which can be a value or a name"
    if isinstance(name, cls):
        return name
    for obj in cls:
        if name == obj.value or (isinstance(name, str) and name.upper() == obj.name.upper()):
            return obj
    if optional:
        return default
    else:
        raise ValueError(f"Given {repr(name)} doesn't match any member of Enum {cls.__name__}")


class ReferenceDtm:
    """Access to the reference data (.dtm.nc)"""

    def __init__(self, dtm_filename: str):
        self.filename = dtm_filename
        logger = pyat_logger.logging.getLogger(Orpheus.__name__)
        logger.info(f"Memory loading of reference file {dtm_filename}...")
        # Read Metadata
        self.driver = dtm_driver.DtmDriver(dtm_filename)
        self.driver.open("r")
        self.dtm_file = self.driver.dtm_file
        self.elevations = self.driver[DtmConstants.ELEVATION_NAME][:].data
        logger.info(f"Reference loaded!")

    def __del__(self):
        self.driver.close()


class SounderData:
    """Access to the xsf sounder data"""

    def __init__(self, filename: str):
        self.filename = filename
        logger = pyat_logger.logging.getLogger(Orpheus.__name__)
        logger.info(f"Loading of sounder data file {filename}...")
        self.driver: SounderDriver = sounder_driver_factory.get_sounder_driver(filename)
        self.driver.open("r")
        self.angles: np.ndarray | None = None
        self.elevations: np.ndarray | None = None
        self.beams: np.ndarray | None = None
        self.valids: np.ndarray | None = None
        self.ref_elevations: np.ndarray | None = None
        self.quality_factors: np.ndarray | None = None
        self.emission_sectors: np.ndarray | None = None
        self.detection_types: np.ndarray | None = None
        self.emission_plans: np.ndarray | None = None
        self.ping_frequencies: np.ndarray | None = None

    def __del__(self):
        self.driver.close()

    def read_valids(self) -> np.ndarray:
        if self.valids is None:
            swath_count = self.driver.sounder_file.swath_count
            self.valids = self.driver.read_validity_flags(0, swath_count)
        return self.valids

    def read_angles(self) -> np.ndarray:
        if self.angles is None:
            swath_count = self.driver.sounder_file.swath_count
            valids: np.ndarray = self.read_valids()
            self.angles = self.driver.read_across_angles(0, swath_count)
            if np.ma.isMA(self.angles):
                self.angles = self.angles.filled(fill_value=np.nan)
            self.angles[np.logical_not(valids)] = np.nan
        return self.angles

    def read_elevations(self) -> np.ndarray:
        if self.elevations is None:
            swath_count = self.driver.sounder_file.swath_count
            valids = self.read_valids()
            self.elevations = -self.driver.read_fcs_depths(0, swath_count)
            self.elevations[np.logical_not(valids)] = np.nan
        return self.elevations

    def read_beams(self) -> np.ndarray:
        if self.beams is None:
            swath_count = self.driver.sounder_file.swath_count
            beam_count = self.driver.sounder_file.beam_count
            self.beams = np.full([swath_count, beam_count], np.nan, dtype=int)
            for swath in np.arange(0, swath_count):
                for beam in np.arange(0, beam_count):
                    self.beams[swath][beam] = beam
        return self.beams

    def read_ref_elevations(self, ref: ReferenceDtm) -> np.ndarray:
        if self.ref_elevations is None:
            swath_count = self.driver.sounder_file.swath_count
            beam_count = self.driver.sounder_file.beam_count
            longitudes = self.driver.read_detection_longitude()
            latitudes = self.driver.read_detection_latitude()

            # longitudes = np.nan_to_num(longitudes, copy=False, nan=10000)
            # latitudes = np.nan_to_num(latitudes, copy=False, nan=10000)
            # projects on ref DTM
            ref_geobox = Geobox(
                ref.driver.dtm_file.north, ref.driver.dtm_file.south, ref.driver.dtm_file.west, ref.driver.dtm_file.east
            )
            ref_geobox.spatial_reference = ref.driver.dtm_file.spatial_reference
            ref_spatial_resolution = ref.driver.dtm_file.spatial_resolution_x
            sounder_spatial_reference = self.driver.sounder_file.spatial_reference
            ref_xs, ref_ys = numpy_utils.project_coords_as_index(
                longitudes, latitudes, ref_geobox, ref_spatial_resolution, sounder_spatial_reference
            )
            ref_width, ref_height = ref.elevations.shape[1], ref.elevations.shape[0]

            self.ref_elevations = np.full([swath_count, beam_count], np.nan, dtype=float)
            for swath in np.arange(0, swath_count):
                for beam in np.arange(0, beam_count):
                    idx = ref_xs[swath][beam]
                    idy = ref_ys[swath][beam]
                    if 0 <= idx < ref_width and 0 <= idy < ref_height:
                        self.ref_elevations[swath][beam] = ref.elevations[idy][idx]
        return self.ref_elevations

    def read_quality_factors(self) -> np.ndarray:
        if self.quality_factors is None:
            valids = self.read_valids()
            self.quality_factors = self.driver.read_detection_quality_factor()
            if np.ma.isMA(self.quality_factors):
                self.quality_factors = self.quality_factors.filled(fill_value=np.nan)
            self.quality_factors[np.logical_not(valids)] = np.nan
        return self.quality_factors

    def read_emission_sectors(self) -> np.ndarray:
        # pylint: disable=assignment-from-none
        if self.emission_sectors is None:
            self.emission_sectors = self.driver.read_detection_tx_beam()
        return self.emission_sectors

    def read_detection_types(self) -> np.ndarray:
        if self.detection_types is None:
            valids = self.read_valids()
            self.detection_types = self.driver.read_detection_type()
            self.detection_types[np.logical_not(valids)] = 1
        return self.detection_types

    def read_emission_plans(self) -> np.ndarray:
        # pylint: disable=assignment-from-none
        if self.emission_plans is None:
            try:
                # S7K, #Kmall
                self.emission_plans = self.driver.read_multiping_sequence()
            except IndexError:
                # all
                # no plan info in .all, so deduce it from center frequencies
                center_frequency = self.driver.read_multiping_center_frequency()
                # find unique row values and replace rows by indices
                _, self.emission_plans = np.unique(center_frequency, axis=0, return_inverse=True)
        return self.emission_plans

    def read_ping_frequencies(self) -> np.ndarray:
        # pylint: disable=assignment-from-none
        if self.ping_frequencies is None:
            swath_count = self.driver.sounder_file.swath_count
            beam_count = self.driver.sounder_file.beam_count
            try:
                # normally found from .all .kmall
                center_frequency = self.driver.read_multiping_center_frequency()
                tx_beam = self.read_emission_sectors()
                if tx_beam is not None:
                    self.ping_frequencies = np.full([swath_count, beam_count], np.nan, dtype=float)
                    for swath in np.arange(0, swath_count):
                        for beam in np.arange(0, beam_count):
                            self.ping_frequencies[swath][beam] = center_frequency[swath][tx_beam[swath][beam]]
                    center_frequency = True
            except IndexError:
                center_frequency = False

            if not center_frequency:
                # center_frequency not found, try ping_frequency for .s7k
                ping_frequency = self.driver.read_detection_ping_frequency()
                if np.ma.isMA(ping_frequency):
                    ping_frequency = ping_frequency.filled(fill_value=np.nan)
                self.ping_frequencies = np.repeat(ping_frequency, beam_count)

        return self.ping_frequencies


class SounderDataFiles:
    """Access to multiple xsf sounder data.

    Internally save all given sounder file with the ActiveCache superclass"""

    def __init__(self, fnames: Iterable[str]):
        self.fnames = tuple(fnames)
        self.data = {f: SounderData(f) for f in self.fnames}

    def get(self, fname: str, default=None) -> Optional[SounderData]:
        return self.data.get(fname, default)


TITLES = {
    ("beam", "bias"): "Elevation bias by beam",
    ("angle", "bias"): "Elevation bias by angle",
    ("beam", "quality_factor"): "Quality factor by beam",
    ("angle", "quality_factor"): "Quality factor by angle",
}
SUBTITLES = {
    ("beam", "bias"): " (elevation is positive up)",
    ("angle", "bias"): " (elevation is positive up)",
    ("beam", "quality_factor"): "",
    ("angle", "quality_factor"): "",
}
LABEL_X = {
    "angle": "Across angle (geometric)",
    "beam": "Beam identifier",
}
LABEL_Y = {
    "measured": "Elevation distribution",
    "bias": "Elevation bias",
    "quality_factor": "Quality factor",
    "mean": "Mean",
    "std": "Standard deviation",
    "count": "Sample count",
    CONFIDENCE_LEVEL_FUNC: "Max value in confidence interval",
}


def write_plots(
    logger,
    x: str,
    y: str,
    y_axis_unit: str,
    hue: str,
    hue_labels: Dict[any, str],
    hue_colors: Dict[any, str],
    metrics: pd.DataFrame,
    mavu: Dict[SurveyOrder, float],
    survey_orders: List[SurveyOrder],
    outdir: str,
    overwrite: bool,
    plot_size: Tuple[int, int],
):
    "Writes matplotlib representations of data"

    def has_to_be_created(name: str) -> bool:
        "True if plot of given name must be created"
        return overwrite or not os.path.exists(name_of(name))

    def name_of(plot_type: str, ext: str = "png") -> str:
        if hue is None:
            basename = f"{y}-{x}-{plot_type}"
        else:
            basename = f"{y}-{x}-{hue}-{plot_type}"
        if y_axis_unit is not None:
            basename = f"{basename}-{y_axis_unit}"
        return os.path.join(outdir, f"plot-{basename}.{ext}")

    def savefig(plot_type: str):
        fname = name_of(plot_type)
        plt.savefig(fname, dpi=PLOT_DPI)
        logger.info(f"File {fname} written!")

    def plot_vertical_uncertainty(
        fig: Figure, plot: Axes, order: SurveyOrder, vertical_uncertainty: float, visible: bool
    ):
        """
        Plot maximum Allowable Vertical Uncertainty
        """
        labels = {
            SurveyOrder.ORDER_2: "IHO2",
            SurveyOrder.ORDER_1B: "IHO1b",
            SurveyOrder.ORDER_1A: "IHO1a",
            SurveyOrder.SPECIAL_ORDER: "IHOsp",
            SurveyOrder.EXCLUSIVE_ORDER: "IHOex",
        }
        line = plot.axhline(y=vertical_uncertainty, linewidth=2, color="r", visible=visible)
        text = plot.text(
            plot.get_xlim()[1], vertical_uncertainty, labels[order], verticalalignment="center", visible=visible
        )
        # Create an instance of ToogleTvu of the SurveyOrder
        fig.canvas.manager.toolmanager.add_tool(labels[order], ToogleTvu, line=line, text=text, toggled=visible)
        # Add the ToogleTvu to the toolbar at specific location inside the Pyat group
        fig.canvas.manager.toolbar.add_tool(labels[order], "PyatGroup", -1)

    if sns is not None:  # seaborn is available, let's draw!
        sns.set(rc={"figure.figsize": plot_size})  # NB: doesn't seems to work in python 3.7
        plt.rcParams["figure.figsize"] = plot_size
        # plt.rcParams["xtick.labelsize"] = 5
        palette = hue_colors if hue_colors else None

        if metrics is not None:
            # draw plots from metrics
            for column in metrics.columns.values:
                if has_to_be_created(name=column):
                    fig = plt.figure()
                    plot = sns.scatterplot(
                        x=x,
                        y=column,
                        hue=hue,
                        data=metrics,
                        palette=palette,
                        marker=".",
                        linewidth=0,
                    )
                    plot.set_title(TITLES[x, y] + f" ({column}){SUBTITLES[x, y]}")
                    plot.set_xlabel(LABEL_X[x])
                    if y_axis_unit is not None and column != "count":
                        if CONFIDENCE_LEVEL_FUNC in column:
                            plot.set_ylabel(f"{LABEL_Y[CONFIDENCE_LEVEL_FUNC]} ({y_axis_unit})")
                        else:
                            plot.set_ylabel(f"{LABEL_Y[column]}  ({y_axis_unit})")
                    else:
                        if CONFIDENCE_LEVEL_FUNC in column:
                            plot.set_ylabel(LABEL_Y[CONFIDENCE_LEVEL_FUNC])
                        else:
                            plot.set_ylabel(LABEL_Y[column])
                    if hue_labels is not None:
                        # replace labels
                        for t in plot.legend_.texts:
                            t.set_text(hue_labels[int(t.get_text())])

                    if column != "count":
                        for order, tvu in mavu.items():
                            plot_vertical_uncertainty(fig, plot, order, tvu, visible=order in survey_orders)

                    savefig(column)
                else:
                    fname = name_of(column)
                    logger.info(f"File {fname} already exists!")


def write_csv(
    logger,
    x: str,
    y: str,
    unit_y: str,
    hue: str = None,
    hue_labels: Dict[any, str] = None,
    metrics: pd.DataFrame = None,
    outdir: str = ".",
    outfile_template: str = "metrics-{name}.{ext}",
    overwrite: bool = False,
):
    "Writes csv representations of data metrics"
    # writes metrics
    if metrics is not None:
        if hue is None:
            basename = f"{y}-{x}"
        else:
            basename = f"{y}-{x}-{hue}"
        if unit_y is not None:
            basename = f"{basename}-{unit_y}"
        fname = os.path.join(outdir, outfile_template.format(name=basename, ext="csv"))
        if overwrite or not os.path.exists(fname):
            if hue is None:
                metrics.to_csv(fname, sep=";")
            else:
                unstacked = metrics.unstack(hue)
                if hue_labels is not None:
                    unstacked.rename(columns=hue_labels, level=hue, inplace=True)
                unstacked.to_csv(fname, sep=";")
            logger.info(f"Export data to file {fname}!")
        else:
            logger.info(f"File {fname} already exists!")


class Orpheus:
    """High-level interface to report generation"""

    def __init__(
        self,
        sounder_files: List[str],
        ref_file: str,
        out_dir: str = ".",
        scope: Scope = Scope.GLOBAL,
        viz_x: VizX = VizX.ANGLE,
        viz_y: VizY = VizY.BIAS,
        unit_y: UnitY = UnitY.METER,
        sounder_criteria: SounderCriteria = SounderCriteria.NOT,
        metrics_names: Optional[List[str]] = None,
        confidence_level: float = 95.0,
        survey_orders: List[SurveyOrder] | None = None,
        angle_bin_width: Union[str, int] = 1,
        count_threshold: Union[str, int] = 0,
        plot_width: Union[str, int] = 8,
        overwrite: bool = False,
        show_plots: bool = True,
        export_csv: bool = True,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        self.logger = pyat_logger.logging.getLogger(Orpheus.__name__)
        self.sounder_files = frozenset(sounder_files)
        self.ref_file = ref_file
        self.loaded_ref = ReferenceDtm(self.ref_file)
        self.loaded_data = SounderDataFiles(self.sounder_files)
        self.report_dir = out_dir
        self.sounder_criteria = get_enum_from_name(SounderCriteria, sounder_criteria, optional=True)
        self.viz_scope = get_enum_from_name(Scope, scope)
        self.viz_x = get_enum_from_name(VizX, viz_x)
        self.viz_y = get_enum_from_name(VizY, viz_y)
        self.unit_y = get_enum_from_name(UnitY, unit_y)
        self.metrics_names = arg_utils.parse_list_of_str(metrics_names)
        self.count_threshold = arg_utils.parse_int("count_threshold", count_threshold)
        self.confidence_level = arg_utils.parse_float("confidence_level", confidence_level, default=95.0) / 100.0

        survey_orders = arg_utils.parse_list_of_str(survey_orders)
        self.survey_orders = [get_enum_from_name(SurveyOrder, order) for order in survey_orders]

        self.angle_bin_width = arg_utils.parse_int("angle_bin_width", angle_bin_width, default=1, min_value=1)
        self.plot_width = arg_utils.parse_int("plot_width", plot_width)
        self.overwrite = overwrite
        self.show_plots = show_plots
        self.export_csv = export_csv
        self.monitor = monitor

    def read_sounderfiles_as_dataframe(
        self, viz_x: VizX, viz_y: VizY, unit_y: UnitY
    ) -> Tuple[pd.DataFrame, Dict[SurveyOrder, float]]:
        all_sounder_df = []
        x_axis_name, y_axis_name, hue_name = self.get_axis_names(viz_x, viz_y)
        for sounder_key, sounder_data in self.loaded_data.data.items():
            # get sounder filename
            file_name = basename_of_fname(sounder_key)
            # computes X axis data
            if viz_x is VizX.ANGLE:
                angles = sounder_data.read_angles()
                # discretize angles
                angles = self.angle_bin_width * np.round(angles / self.angle_bin_width)
                x_axis_data = angles
            else:  # viz_x is VizX.BEAM:
                x_axis_data = sounder_data.read_beams()

            # computes Y axis data
            # Maximum Allowable Vertical Uncertainty. Key is SurveyOrder
            mavu = {}
            if viz_y is VizY.BIAS:
                # computes bias
                elevations = sounder_data.read_elevations()
                ref_elevations = sounder_data.read_ref_elevations(self.loaded_ref)
                bias = elevations - ref_elevations
                mavu = {order: maximum_avu_for_order(order, ref_elevations) for order in SurveyOrder}
                if unit_y is UnitY.PERCENT:
                    bias = 100 * bias / abs(ref_elevations)
                    mavu = {order: 100 * mavu[order] / abs(np.nanmean(ref_elevations)) for order in SurveyOrder}
                y_axis_data = bias
            else:  # viz_y is VizY.QUALITY_FACTOR:
                y_axis_data = sounder_data.read_quality_factors()

            # computes Hue data (category)
            if self.sounder_criteria is SounderCriteria.NOT:
                if self.viz_scope is Scope.GLOBAL:
                    sounder_df = pd.DataFrame(data={x_axis_name: x_axis_data.flat, y_axis_name: y_axis_data.flat})
                else:  # if self.viz_scope is Scope.PER_FILE:
                    sounder_df = pd.DataFrame(
                        data={x_axis_name: x_axis_data.flat, y_axis_name: y_axis_data.flat, hue_name: file_name}
                    )
            else:
                if self.sounder_criteria is SounderCriteria.EMISSION_SECTOR:
                    hue_data = sounder_data.read_emission_sectors()
                elif self.sounder_criteria is SounderCriteria.DETECTION_TYPE:
                    hue_data = sounder_data.read_detection_types()
                elif self.sounder_criteria is SounderCriteria.EMISSION_PLAN:
                    beam_count = sounder_data.driver.sounder_file.beam_count
                    hue_data = np.repeat(sounder_data.read_emission_plans(), beam_count)
                elif self.sounder_criteria is SounderCriteria.PING_FREQUENCY:
                    hue_data = sounder_data.read_ping_frequencies()
                else:
                    raise NotImplementedError("SounderCriteria value is not yet implemented")

                if hue_data is None:
                    raise ValueError(f"{self.sounder_criteria} not supported for the input file type")

                if self.viz_scope is Scope.GLOBAL:
                    sounder_df = pd.DataFrame(
                        data={
                            x_axis_name: x_axis_data.flat,
                            y_axis_name: y_axis_data.flat,
                            hue_name: hue_data.flat,
                        }
                    )
                else:  # if self.viz_scope is Scope.PER_FILE:
                    sounder_df = pd.DataFrame(
                        data={
                            x_axis_name: x_axis_data.flat,
                            y_axis_name: y_axis_data.flat,
                            hue_name: hue_data.flat,
                            "file": file_name,
                        }
                    )

            # add sounder dataframe
            all_sounder_df.append(sounder_df)

        return pd.concat(all_sounder_df, ignore_index=True), mavu

    def compute_metrics(self, df: pd.DataFrame, viz_x: VizX, viz_y: VizY) -> pd.DataFrame:
        "Add new columns to given dataframe, and return a brand new dataframe with mean/stdev metrics that are grouped by the x-axis."
        if self.metrics_names:  # ["mean", "std", "count", "confidence level"]
            # List of aggregating function
            agg_funcs = [func for func in self.metrics_names if func != CONFIDENCE_LEVEL_FUNC]
            if CONFIDENCE_LEVEL_FUNC in self.metrics_names and self.confidence_level:
                # Replace the unknown function name "confidence level" by a calling function
                quantile = partial(pd.Series.quantile, q=self.confidence_level)
                quantile.__name__ = f"{CONFIDENCE_LEVEL_FUNC} {self.confidence_level * 100:.0f}%"
                agg_funcs.append(quantile)

            x_axis_name, y_axis_name, hue_name = self.get_axis_names(viz_x, viz_y)
            if hue_name:
                if self.viz_scope is Scope.GLOBAL or self.sounder_criteria is SounderCriteria.NOT:
                    group = [x_axis_name, hue_name]
                else:
                    group = [x_axis_name, "file", hue_name]
                group_by_hue = df.groupby(group)[y_axis_name]
                metrics = group_by_hue.agg(agg_funcs)
                if self.count_threshold > 0:
                    hue_counts = group_by_hue.count()
                    all_counts = df.groupby(x_axis_name)[y_axis_name].count()
                    all_metrics_counts = metrics.join(all_counts, how="inner")
                    metrics = metrics.mask(
                        hue_counts / all_metrics_counts[y_axis_name] < self.count_threshold / 100,
                        other=np.nan,
                    )
                return metrics
            else:
                # GLOBAL and SounderCriteria == NOT
                metrics = df.groupby(x_axis_name)[y_axis_name].agg(agg_funcs)
                return metrics
        else:
            return None

    def get_axis_names(self, viz_x: VizX, viz_y: VizY) -> Tuple[str, str, str | None]:
        if viz_x is VizX.ANGLE:
            x_axis_name = viz_x.name.lower()
        elif viz_x is VizX.BEAM:
            x_axis_name = viz_x.name.lower()
        else:
            raise NotImplementedError(f"viz_x={viz_x} support is not yet implemented")

        if viz_y is VizY.BIAS:
            y_axis_name = viz_y.name.lower()
        elif viz_y is VizY.QUALITY_FACTOR:
            y_axis_name = viz_y.name.lower()
        else:
            raise NotImplementedError(f"viz_y={viz_y} support is not yet implemented")

        if self.sounder_criteria is SounderCriteria.NOT:
            if self.viz_scope is Scope.GLOBAL:
                hue_axis_name = None
            else:  # if self.viz_scope is Scope.PER_FILE:
                hue_axis_name = "file"
        elif self.sounder_criteria is SounderCriteria.EMISSION_SECTOR:
            hue_axis_name = self.sounder_criteria.name.lower()
        elif self.sounder_criteria is SounderCriteria.DETECTION_TYPE:
            hue_axis_name = self.sounder_criteria.name.lower()
        elif self.sounder_criteria is SounderCriteria.EMISSION_PLAN:
            hue_axis_name = self.sounder_criteria.name.lower()
        elif self.sounder_criteria is SounderCriteria.PING_FREQUENCY:
            hue_axis_name = self.sounder_criteria.name.lower()
        else:
            raise NotImplementedError("SounderCriteria value is not yet implemented")

        return x_axis_name, y_axis_name, hue_axis_name

    def get_hue_labels(self) -> Dict[any, str]:
        hue_labels = None
        if self.sounder_criteria is SounderCriteria.DETECTION_TYPE:
            hue_labels = {
                0: "invalid",
                1: "amplitude",
                2: "phase",
                127: "unknown",
                -127: "unknown",
                -128: "unknown",
                np.nan: "unknown",
            }
        return hue_labels

    def get_hue_colors(self) -> Dict[any, str]:
        hue_colors = None
        if self.sounder_criteria is SounderCriteria.DETECTION_TYPE:
            hue_colors = {0: "grey", 1: "blue", 2: "red", 127: "grey", -127: "grey", -128: "grey", np.nan: "grey"}
        return hue_colors

    def report_mavu(self, logger: Logger, mavu: Dict[SurveyOrder, float]) -> None:
        """
        Log values or TVU for all required order
        """
        for order in mavu:
            logger.info(f"TVU for survey order {order.name.lower()} : {mavu[order]}")

    def create_plots(
        self,
        viz_x: VizX = VizX.ANGLE,
        viz_y: VizY = VizY.BIAS,
        unit_y: UnitY = UnitY.METER,
        overwrite: bool = False,
        show_plots: bool = False,
        plot_width: int = 6,
        export_csv: bool = True,
    ):
        logger = pyat_logger.logging.getLogger(Orpheus.__name__)
        self.monitor.begin_task(name="Creating plots", n=4)

        logger.info("Reading sounder files")
        dataframe, mavu = self.read_sounderfiles_as_dataframe(viz_x, viz_y, unit_y)
        self.report_mavu(logger, mavu)
        self.monitor.worked(1)
        logger.info("Computing metrics")
        metrics = self.compute_metrics(dataframe, viz_x, viz_y)  # then, get metrics
        self.monitor.worked(1)
        x_axis_name, y_axis_name, hue_name = self.get_axis_names(viz_x, viz_y)
        hue_labels = self.get_hue_labels()
        hue_colors = self.get_hue_colors()
        y_axis_unit = unit_y.name.lower() if viz_y is VizY.BIAS else None
        logger.info(
            f"PlotCreator will plot {y_axis_name.title()} against {x_axis_name.title()} of {len(dataframe)} records."
        )

        write_plots(
            logger=logger,
            x=x_axis_name,
            y=y_axis_name,
            y_axis_unit=y_axis_unit,
            hue=hue_name,
            hue_labels=hue_labels,
            hue_colors=hue_colors,
            metrics=metrics,
            mavu=mavu,
            survey_orders=self.survey_orders,
            outdir=self.report_dir,
            overwrite=overwrite,
            plot_size=(plot_width, round(plot_width // 1.5)),
        )
        self.monitor.worked(1)

        logger.info("Exporting csv")
        if export_csv:
            write_csv(
                logger=logger,
                x=x_axis_name,
                y=y_axis_name,
                unit_y=y_axis_unit,
                hue=hue_name,
                hue_labels=hue_labels,
                metrics=metrics,
                outdir=self.report_dir,
                overwrite=overwrite,
            )
        self.monitor.worked(1)

        # show plots
        if show_plots:
            plt.show(block=True)

    def __call__(self):
        """Globe asked to run"""
        self.logger.info(f"Orpheus started...")
        self.create_plots(
            viz_x=self.viz_x,
            viz_y=self.viz_y,
            unit_y=self.unit_y,
            overwrite=self.overwrite,
            show_plots=self.show_plots,
            plot_width=self.plot_width,
            export_csv=self.export_csv,
        )
