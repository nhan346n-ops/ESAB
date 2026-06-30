r"""Laloxy: high-level formatter helper for csv files.

Main API functions are:

    convert_csv(<csv file>, <outfile desc>, <longitude col idx>, <latitude column index>, <formatting template>) -> list[str]

The coordinates formatting template uses the palceholders fully described
in docstring of function `coords.formatted_coordinates`, but here are some examples:

    {D} Degrees (integer, positive)
    {M} Minutes (integer)
    {S} Seconds (float)
    {B} Degrees (integer, signed)

For instance:
>>> coords.formatted_coordinates(-164, -77, '{B}') == coords.formatted_coordinates(-164, -77, '{s}{D}')
True


There is also some defaults formats that can be given by name to the run method,
including those following:

    DEG_MIN_SEC: '{D} {M} {S}'
    RICHDEG_MIN_SEC: '{B}° {M}? {S}?{w}'
    CARAIBES_DEGREES: '{s}{p:.6f}'

The Caraibes-related default formats are emulating the output of the Caraïbes
software, which provide 3 possible outputs:
    DEGREES:  `sdd.dddddd` for latitude, and `sddd.dddddd` for longitudes
    DEG_MIN_DEC: `sddd  mm.mmmmm` for latitude, and `sddd  mm.mmmmm` for longitudes
    XY:  `smmmmmmmmm.mm` for latitude, and `smmmmmmmmm.mm` for longitudes


The `convert_csv` method is converting the longitude and lattitude values found
in columns of given indexes, writing the same CSV into the given output file.

For instance:

    convert_csv('mycsvfile.csv', sys.stdout, 0, 1, "{D} {M} {S} {w}")

"""

import datetime
import locale
import os
from enum import Enum
from typing import IO, Callable, Optional, Tuple, Type, Union

import pandas
from pygws.service.progress_monitor import DefaultMonitor, ProgressMonitor
from pyproj import CRS

import pyat.dtm.utils.process_utils as process_util
import pyat.utils.pyat_logger as log
from pyat.utils import coords

PANDAS_CHUNK_SIZE = 1500000


class DefaultInputFormats(Enum):
    "Default input formats that user may provide as input"

    XY = "XY"  # UTM rectangular coordinates, e.g. 933813.46. Need a projection to be valid.
    DEGREES = "DEGREES"  # Degrees, such as -77.508333°
    DEG_MIN_DEC = "DEG_MIN_DEC"  # Degrees and Minutes, e.g. -45° 17,896' N
    DEG_MIN_SEC = "DEG_MIN_SEC"  # Degrees Minutes and Seconds, e.g. 164° 45' 15.0012" W
    CARAIBES_DEGREES = (
        "CARAIBES_DEGREES"  # Degrees, such as -77.508333, but it is expected that the delimiter is spaces
    )
    CARAIBES_XY = "CARAIBES_XY"  # UTM rectangular coordinates, but it is expected that the delimiter is spaces


class DefaultFormats(Enum):
    "Default formats that user may ask for instead of devising a format by himself"

    DEGREES = "{b}"  # Degrees only
    DEG_MIN_DEC = "{D} {m}"  # Degrees and Minutes
    DEG_MIN_SEC = "{D} {M} {S}"  # Degrees Minutes and Seconds
    RICHDEG_MIN_SEC = "{B}\u00b0 {M}' {S}'' {w}"  # Degrees Minutes and Seconds stylized with marks
    XY = "{x}"  # `sm.mm` for latitude, and `sm.mm` for longitudes
    CARAIBES_DEGREES = "{p:+.6f}"  # `sdd.dddddd` for latitude, and `sddd.dddddd` for longitudes
    CARAIBES_DEG_MIN_DEC = "{s}{P}  {m:08.6f}"  # `sddd  mm.mmmmm` for latitude, and `sddd  mm.mmmmm` for longitudes
    CARAIBES_XY = "{x:+013.2f}"  # `smmmmmmmmm.mm` for latitude, and `smmmmmmmmm.mm` for longitudes


def get_enum_from_name(cls: Type[Enum], name: Union[str, Enum], default=None) -> Enum:
    if isinstance(name, cls):
        return name
    assert isinstance(name, str)
    for obj in cls:
        if name.upper() == obj.name.upper():
            return obj
    return default


# pylint: disable=function-redefined
def coordinates_converter(
    input_format: DefaultInputFormats,
    output_format: str,
    input_proj: Optional[str],
    output_proj: Optional[str],
    rounding: int,
) -> Callable:
    """Return a function able to convert input coordinates in `input_format` to `output_format`"""
    # get function getting DEGREES from input.
    if input_format is DefaultInputFormats.XY and output_format is DefaultFormats.CARAIBES_XY.value:
        # special case: the converter already exists
        xy_to_xy = coords.reprojection_converter(input_proj, output_proj, rounding)

        def direct_converter(xs, ys, *, xy_to_xy=xy_to_xy):
            "This converter will convert all xs/ys, then call format on each point"
            pxs, pys = xy_to_xy(xs, ys)
            fxs, fys = zip(
                *(
                    (DefaultFormats.CARAIBES_XY.value.format(x=x), DefaultFormats.CARAIBES_XY.value.format(x=y))
                    for x, y in zip(pxs, pys)
                )
            )
            return tuple(fxs), tuple(fys)

        return direct_converter

    if input_format is DefaultInputFormats.DEGREES or input_format is DefaultInputFormats.CARAIBES_DEGREES:
        # trivial: input is already in lonlat degrees.
        def dd_from(lonx: str, laty: str) -> Tuple[str, str]:
            return coords.DEGREES_from_wildDEGREES(lonx), coords.DEGREES_from_wildDEGREES(laty)

    elif input_format is DefaultInputFormats.DEG_MIN_DEC:

        def dd_from(lonx: str, laty: str) -> Tuple[str, str]:
            return coords.DEGREES_from_DEG_MIN_DEC(lonx), coords.DEGREES_from_DEG_MIN_DEC(laty)

    elif input_format is DefaultInputFormats.DEG_MIN_SEC:

        def dd_from(lonx: str, laty: str) -> Tuple[str, str]:
            return coords.DEGREES_from_DEG_MIN_SEC(lonx), coords.DEGREES_from_DEG_MIN_SEC(laty)

    elif input_format is DefaultInputFormats.XY or input_format is DefaultInputFormats.CARAIBES_XY:
        xy_to_lonlat = coords.create_xy_to_lonlat_converter(proj=input_proj, rounding=rounding)

        def dd_from(x, y, *, xy_to_lonlat=xy_to_lonlat):
            xs, ys = xy_to_lonlat([x], [y])
            return xs[0], ys[0]

    else:  # input format is unknown
        raise ValueError(f"Given input format {input_format} of type {type(input_format)} is not handled.")

    dds2xys = coords.create_lonlat_to_xy_converter(output_proj)
    converter = coords.reprojection_converter
    if "x" in output_format:

        def converter(xs, ys, *, output_format=output_format, rounding=rounding, dd_from=dd_from, dds2xys=dds2xys):
            "This converter will convert all xs/ys, then call format on each point"
            ddxs, ddys = zip(*(dd_from(x, y) for x, y in zip(xs, ys)))
            pddxs, pddys = dds2xys(ddxs, ddys)
            fxs, fys = zip(
                *(
                    coords.formatted_coordinates(ddx, ddy, fmt=output_format, x=px, y=py, rounding=rounding)
                    for ddx, ddy, px, py in zip(ddxs, ddys, pddxs, pddys)
                )
            )
            return tuple(fxs), tuple(fys)

    else:  # don't convert to XY

        def converter(xs, ys, *, output_format=output_format, rounding=rounding, dd_from=dd_from):
            "This converter will call format on each point"
            ddxs, ddys = zip(*(dd_from(x, y) for x, y in zip(xs, ys)))
            fxs, fys = zip(
                *(
                    coords.formatted_coordinates(ddx, ddy, fmt=output_format, rounding=rounding)
                    for ddx, ddy in zip(ddxs, ddys)
                )
            )
            return tuple(fxs), tuple(fys)

    return converter


# pylint: enable=function-redefined


class ConvertCSV:
    """This class group methods related to the extraction from an input CSV file
    of coordinates to be written in another format in an output CSV file.

    """

    def __init__(
        self,
        infile: str,
        outfile: IO,
        lon_or_x_col_index: int,
        lat_or_y_col_index: int,
        input_format: Optional[Union[str, DefaultInputFormats]] = DefaultInputFormats.DEGREES,
        output_format: Union[str, DefaultFormats] = DefaultFormats.RICHDEG_MIN_SEC,
        input_proj: Optional[str] = None,
        output_proj: Optional[str] = None,
        rounding: int = 5,
        keep_bad_lines: bool = False,
        logger: object = None,
        delimiter: str = ";",
        skiprows: int = 0,
        line_callback: Optional[Callable] = None,
    ):

        self.infile, self.outfile = infile, outfile
        self.logger = logger or log.logging.getLogger(self.__class__.__name__)

        # Input projection
        self.input_proj = input_proj
        if self.input_proj:
            input_crs = CRS.from_proj4(self.input_proj)
            if input_crs.is_projected and input_proj != DefaultInputFormats.XY:
                self.logger.warning(f"Input projection imposes XY input format")
                input_format = DefaultInputFormats.XY

        self.output_proj = output_proj
        self.lat_or_y_col_index, self.lon_or_x_col_index = map(int, (lat_or_y_col_index, lon_or_x_col_index))
        self.keep_bad_lines, self.line_callback = keep_bad_lines, line_callback
        self.csv_delimiter, self.csv_skiprows = delimiter, skiprows
        # handle multiple spaces
        if self.csv_delimiter == " ":
            self.csv_delimiter = r"\s+"
        if isinstance(input_format, DefaultInputFormats):
            self.input_format = input_format  # nothing to do
        elif isinstance(input_format, str) and get_enum_from_name(DefaultInputFormats, input_format) is not None:
            self.input_format = get_enum_from_name(
                DefaultInputFormats, input_format, DefaultInputFormats.DEGREES
            )  # here, default is useless but mypy needs it
        else:
            self.logger.warning(
                f"Input Format '{input_format}' doesn't exists. Available are: {', '.join(m.name for m in DefaultInputFormats)}. {DefaultInputFormats.DEGREES} will be used."
            )
            self.input_format = DefaultInputFormats.DEGREES

        if isinstance(output_format, DefaultFormats):
            self.output_format = output_format  # nothing to do
        elif isinstance(output_format, str) and get_enum_from_name(DefaultFormats, output_format) is not None:
            self.output_format = get_enum_from_name(
                DefaultFormats, output_format, DefaultFormats.RICHDEG_MIN_SEC
            )  # here, default is useless but mypy needs it
        else:
            self.logger.warning(
                f"Output Format '{output_format}' doesn't exists. Available are: {', '.join(m.name for m in DefaultFormats)}. {DefaultFormats.RICHDEG_MIN_SEC} will be used"
            )
            self.output_format = DefaultFormats.RICHDEG_MIN_SEC

        # Create the coordinates converter
        self.converted = coordinates_converter(
            self.input_format, self.output_format.value, self.input_proj, self.output_proj, rounding=rounding
        )

    def __call__(self):
        # Detect malformed parameters.
        if "+proj=longlat" not in self.output_proj and r"x" not in self.output_format.value:
            self.logger.error(
                "Output projection is a XY value, but the format used is not a projectable format. Use XY if needed. Aborting."
            )
            return

        self.logger.info("Loading chunks of input data...")
        ifd_chunks = pandas.read_csv(
            self.infile,
            chunksize=PANDAS_CHUNK_SIZE,
            on_bad_lines="warn",
            sep=r"\s+" if self.csv_delimiter == "…" else self.csv_delimiter,
            skiprows=self.csv_skiprows,
            index_col=False,
            header=None,
        )
        for idx, chunk in enumerate(ifd_chunks, start=1):
            self.logger.info(f"chunk {idx:04d} loaded, converting...")
            self.numpy_convert_whole_array(chunk)
            self.logger.info(f"write chunk {idx:04d}...")
            self.save_as_csv(chunk)
        self.logger.info("Done.")

    def save_as_csv(self, df: pandas.DataFrame):
        "given df will be written in the self.outfile. If necessary, will ensure a readable alignment"
        output_must_be_tabulated = self.csv_delimiter in {r"\s", r"\s+", " ", " +", "…"}
        if output_must_be_tabulated:
            sep = ";"  # won't be used ; will merge columns instead as a single string
            # format each column so it is correctly aligned
            df = df.astype(str).apply((" ").join, axis=1)
        else:
            sep = self.csv_delimiter
        df.to_csv(self.outfile, mode="a", sep=sep, index=False, header=None, encoding="utf-8")

    def numpy_convert_whole_array(self, data):
        "Will apply the conversion on the column that contains latitude and longitude"
        x, y = self.converted(data.iloc[:, self.lon_or_x_col_index], data.iloc[:, self.lat_or_y_col_index])
        data.iloc[:, self.lon_or_x_col_index], data.iloc[:, self.lat_or_y_col_index] = x, y


class Laloxy:
    """Interface class, called by Globe. Parse received arguments and use
    the ConvertCSV class to perform the required treatment"""

    def __init__(
        self,
        i_path: str,
        o_path: str,
        indexes: dict,  # Globe interface must handle CSV indexes as dict, not integers
        input_format: Optional[DefaultInputFormats] = None,
        output_format: Union[str, DefaultFormats] = DefaultFormats.RICHDEG_MIN_SEC.name,
        input_proj: Optional[str] = None,
        output_proj: Optional[str] = None,
        rounding: int = coords.DEFAULT_ROUNDING,
        delimiter: Optional[str] = None,
        skip_rows: int = 0,
        overwrite: bool = False,
        monitor: ProgressMonitor = DefaultMonitor,
    ):
        if isinstance(indexes["Latitude/Y"], (tuple, list)) or isinstance(indexes["Longitude/X"], (tuple, list)):
            raise NotImplementedError(
                "Coordinates spanning accross multiple columns is not handled. Integer expected for Latitude/Y and Longitude/X parameters."
            )

        self.logger = log.logging.getLogger(self.__class__.__name__)
        self.monitor, self.overwrite = monitor, overwrite
        self.i_path, self.o_path = i_path, o_path
        self.lat_or_y_col_index, self.lon_or_x_col_index = map(int, (indexes["Latitude/Y"], indexes["Longitude/X"]))
        self.input_proj, self.output_proj = map(str, (input_proj or "", output_proj or ""))
        self.rounding = int(rounding)
        self.input_format = input_format
        self.output_format = output_format
        self.csv_delimiter = str(delimiter or ";")
        self.csv_skiprows = int(skip_rows)

    def __call__(self) -> None:
        """Run method"""
        begin = datetime.datetime.now()
        self.monitor.set_work_remaining(os.path.getsize(self.i_path))
        line_callback = self.monitor.worked
        if self.check_outfile_existence_validity():
            with open(self.o_path, "w", newline="", encoding=locale.getpreferredencoding()) as ofd:
                ConvertCSV(
                    self.i_path,
                    ofd,
                    self.lon_or_x_col_index,
                    self.lat_or_y_col_index,
                    self.input_format,
                    self.output_format,
                    self.input_proj,
                    self.output_proj,
                    self.rounding,
                    True,
                    self.logger,
                    self.csv_delimiter,
                    self.csv_skiprows,
                    line_callback=line_callback,
                )()

        self.monitor.done()
        process_util.log_result(self.logger, begin, [])

    def check_outfile_existence_validity(self) -> bool:
        "Return False if the file of given name exists and overwrite is not allowed"
        if self.i_path == self.o_path:
            self.logger.error("The same file was given as input and output. Abort.")
            return False
        if os.path.exists(self.o_path):
            if self.overwrite:
                self.logger.info(f"Output file {self.o_path} already exists. It will be overwritten.")
            else:
                self.logger.error(f"Output file {self.o_path} already exists. Abort.")
                return False
        return True  # everything is ok
