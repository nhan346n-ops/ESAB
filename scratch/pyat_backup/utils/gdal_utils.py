import os
import tempfile as tmp
from pathlib import Path
from typing import List, Union

import numpy as np
from matplotlib import colormaps
from osgeo import gdal, ogr
from pygws.service.progress_monitor import ProgressMonitor

import pyat.utils.pyat_logger as log

logger = log.logging.getLogger(__name__)

STR_TO_GDAL_ALGORITHM = {
    "near": gdal.GRA_NearestNeighbour,
    "bilinear": gdal.GRA_Bilinear,
    "cubic": gdal.GRA_Cubic,
    "cubicspline": gdal.GRA_CubicSpline,
    "lanczos": gdal.GRA_Lanczos,
    "average": gdal.GRIORA_Average,
    "mode": gdal.GRA_Mode,
    "max": gdal.GRA_Max,
    #        "sum": gdal.GRA_Sum,
    "min": gdal.GRA_Min,
    "med": gdal.GRA_Med,
    "q1": gdal.GRA_Q1,
    "q3": gdal.GRA_Q3,
}
EXT_TO_OGR_DRIVER = {".shp": "ESRI Shapefile", ".kml": "KML", ".gpkg": "GPKG"}
GDAL_OPEN_MODES = {"r": gdal.GA_ReadOnly, "r+": gdal.GA_Update}


def translate_algorithm(arg_name: str, algo: str):
    if algo not in STR_TO_GDAL_ALGORITHM:
        raise ValueError(f"Invalid value '{algo}' for argument {arg_name}")
    return STR_TO_GDAL_ALGORITHM[algo]


class GDALDataset:
    """
    A GDAL context manager for raster dataset
    Allow the opening (read-only) of a GDAL dataset in a "with" statement
    - 'r' : read-only (default)
    - 'r+' : read and write.
    """

    def __init__(self, filename, mode="r"):
        self.filename = filename
        self.mode = self.__get_opening_mode(mode)
        self.dataset = None

    def __enter__(self) -> gdal.Dataset:
        """
        open the GDAL dataset in the specified mode :
        - 'r' : read-only (default)
        - 'r+' : read and write.
        """
        self.dataset = gdal.Open(self.filename, self.mode)
        if self.dataset is None:
            raise IOError(f"Failed to open dataset: {self.filename}")
        return self.dataset

    def __exit__(self, exc_type, exc_value, traceback):
        """ensure GDAL dataset closing on exit"""
        if self.dataset is not None:
            self.dataset = None

    def __getattr__(self, name):
        """Delegate attribute access to the underlying GDAL dataset."""
        if self.dataset is None:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.dataset, name)

    def __get_opening_mode(self, mode: str):
        """Return the GDAL opening mode corresponding to passed character"""
        if mode not in GDAL_OPEN_MODES:
            raise ValueError(
                f"invalid opening mode {mode} for {self.__class__.__name__}. Valid modes are {list(GDAL_OPEN_MODES.keys())}"
            )
        return GDAL_OPEN_MODES[mode]


class OGRDataset:
    """
    an OGR context manager for vector dataset
    Allow the creation and opening of a OGR dataset (.shp for ex) in a "with" statement
    Only supports formats in EXT_TO_OGR_DRIVER
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = None

    def __get_driver_from_ext(self) -> ogr.Driver:
        """Return the OGR driver corresponding to file extension"""
        _, ext = os.path.splitext(self.file_path)
        if ext not in EXT_TO_OGR_DRIVER:
            raise ValueError(
                f"driver {EXT_TO_OGR_DRIVER[ext]} for '{ext}' not implemented yet in {self.__class__.__name__}"
            )
        return ogr.GetDriverByName(EXT_TO_OGR_DRIVER[ext])

    def __enter__(self) -> ogr.DataSource:
        """create and return the OGR datasource"""
        driver = self.__get_driver_from_ext()
        self.dataset = driver.CreateDataSource(self.file_path)
        if self.dataset is None:
            raise IOError(f"Failed to create OGR dataset {self.file_path}")
        return self.dataset

    def __exit__(self, exc_type, exc_value, traceback):
        """ensure OGR datasource writing and closing on exit"""
        if self.dataset is not None:
            self.dataset.FlushCache()
            self.dataset = None


class DatasetWrapper:
    """A gdal dataset wrapper allowing to add a few metadata on dataset"""

    def __init__(self, dataset: gdal.Dataset):
        self.dataset = dataset


class TemporaryDataset(DatasetWrapper):
    """a Temporary dataset, the file associated with the dataset will be deleted by the destructor"""

    # Initializing
    def __init__(self, dataset: gdal.Dataset, filepath: str, **kwargs):
        super().__init__(dataset)
        self.filepath = filepath
        if "verbose" in kwargs:
            self.verbose = bool(kwargs["verbose"])
        else:
            self.verbose = False

    def __del__(self):
        """Temporary dataset destructor, close dataset and remove associated file"""
        # close the dataset
        if self.dataset is not None:
            del self.dataset

        try:
            if self.verbose:
                print("delete file ", self.filepath)
            os.remove(self.filepath)
        except Exception as e:
            logger.warning(f"Error while deleting file {self.filepath} {e}")


def gdal_progress_callback(pct: float, gdal_msg: str, callback_data: List[Union[float, str, ProgressMonitor]]) -> int:
    """
    A GDAL Callback progress function, activated by GDAL processes and used to display progression
    callback_data contains :
    - the last printed percent of progression : float from 0.0 to 1.0
    - the last printed message : default message from GDAL (often None)
    - A ProgressMonitor
    """
    progress = callback_data[0]
    last_msg = callback_data[1]
    monitor = callback_data[2]
    # Stop progression on cancel
    if monitor.check_cancelled():
        monitor.logger.warning("Cancelled")
        return 0  # Stop
    # Init progression
    if gdal_msg != last_msg:
        # GDAL (often) call the callback with msg=None,
        # so we set monitor message from passed msg in callback data at the first call
        monitor.begin_task(name=last_msg, n=100)
        callback_data[1] = gdal_msg  # store the message emitted by GDAL
    # Update progression every 10%
    if pct * 100 >= progress + 10:
        monitor.worked(10)
        callback_data[0] = progress + 10  # store the last printed percent of progression in callback_data
    return 1  # Continue


def netcdf_to_gdal(value: np.ndarray):
    """Convert a netcdf dataset array with origin left bottom to a gdal dataset array with upper left origin convention"""
    return value[::-1, :]


def gdal_to_netcdf(gdal_dataset: gdal.Dataset):
    """Convert a gdal dataset array with upper left origin convention to a netcdf array convention with origin left bottom"""
    return gdal_dataset.ReadAsArray()[::-1, :]


def get_x_y_coordinates(gdal_dataset: gdal.Dataset):
    """
    Compute and return two X (longitude) and Y (latitude) coordinates vector from a gdal dataset
    The coordinates computed refers to the coordinates at the center of the cells
    Args:
        gdal_dataset:

    Returns:
        (x,y) the coordinates vector
    """
    (GT_0, GT_1, GT_2, GT_3, GT_4, GT_5) = gdal_dataset.GetGeoTransform()
    x_values = np.arange(0, gdal_dataset.RasterXSize)
    y_values = np.arange(0, gdal_dataset.RasterYSize)

    x_geo = GT_0 + (x_values + 0.5) * GT_1 + 0 * GT_2
    y_geo = GT_3 + 0 * GT_4 + (y_values + 0.5) * GT_5
    return x_geo, y_geo


def gdal_raster_to_geo(gdal_dataset: gdal.Dataset, x_pixel, y_pixel):
    """
    Convert Xpixel and Ypixel coordinate to geographic coordinates
    Xpixel and Ypixel should be of the same size
    Returns:
        Xgeo, Ygeo
    """
    (GT_0, GT_1, GT_2, GT_3, GT_4, GT_5) = gdal_dataset.GetGeoTransform()
    x_geo = GT_0 + x_pixel * GT_1 + y_pixel * GT_2
    y_geo = GT_3 + x_pixel * GT_4 + y_pixel * GT_5
    return x_geo, y_geo


def create_palette_file(nodata: float, min_elev: float, max_elev: float, colormap: str = "viridis") -> str:
    """
    Convert a color palette (from matlplotlib, or .cpt file) to a GDAL compatible one.
    Return the path to the created palette file.
    """
    if colormap in colormaps:
        # load correct colormap from matplotlib
        cmap = colormaps[colormap]
        # Rgb from float [0, 1] to int [0, 255]
        colors = (np.asarray(cmap.colors) * 255).astype(int)
        reds = colors[:, 0]
        greens = colors[:, 1]
        blues = colors[:, 2]
    elif Path(colormap).is_file() and colormap.endswith(".cpt"):
        # load colormap from .cpt file
        cmap = np.loadtxt(colormap)
        # cmap = pd.read_csv(colormap, sep=" ")
        colors = (cmap.values * 255).astype(int)
        reds = np.dstack((colors[:, 1], colors[:, 5]))  # dstack : 2 columns per color
        reds = reds.reshape((1, reds.size))
        greens = np.dstack((colors[:, 2], colors[:, 6]))
        greens = greens.reshape((1, greens.size))
        blues = np.dstack((colors[:, 3], colors[:, 7]))
        blues = blues.reshape((1, blues.size))
    else:
        raise ValueError(f"Colormap {colormap} not found in matplotlib colormaps or as a .cpt file")

    palette_size = np.min([reds.size, greens.size, blues.size]) - 1
    elevation_step = (max_elev - min_elev) / palette_size
    elevation = min_elev
    with tmp.NamedTemporaryFile(suffix="_gdal_palette.txt", delete=False) as palette_file:
        palette_file_path = palette_file.name
    with open(palette_file_path, "w", encoding="utf8") as palette_file:
        for r, g, b in zip(np.nditer(reds), np.nditer(greens), np.nditer(blues)):
            palette_file.write(f"{elevation:.4f} {r} {g} {b} 255\n")
            elevation = elevation + elevation_step
        if np.isnan(nodata):
            palette_file.write("nv 0 0 0 0\n")
        else:
            palette_file.write(f"{nodata} 0 0 0 0\n")

    return palette_file_path


def apply_colormap(
    ds: gdal.Dataset, o_path: str, monitor: ProgressMonitor = None, logger=logger, colormap: str = "viridis"
) -> bool:
    """
    Run a color-relief colormap transformation using GDAL DEMProcessing.
    The color palette is created on the fly from the dataset statistics.
    The resulting image is a 4-band RGBA geotiff.
    """
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    min_elev, max_elev, _, _ = band.GetStatistics(True, True)
    palette_file_path = create_palette_file(nodata, min_elev, max_elev, colormap=colormap)

    try:
        dem_option = gdal.DEMProcessingOptions(
            format="GTiff",
            colorFilename=palette_file_path,
            colorSelection="nearest_color_entry",
            addAlpha=True,
            callback=gdal_progress_callback if monitor is not None else gdal.TermProgress_nocb,
            callback_data=[0, "applying colormap", monitor.split(1)] if monitor is not None else None,
        )
        color_dataset = gdal.DEMProcessing(destName=o_path, srcDS=ds, processing="color-relief", options=dem_option)
        if color_dataset is None:
            logger.error(f"Gdal failed to create the tiff for {ds.GetDescription()}")
            result = False
        else:
            # release the dataset reference and indicate success
            color_dataset = None
            result = True
    finally:
        # ensure temporary palette file is always removed
        os.remove(palette_file_path)
    return result
