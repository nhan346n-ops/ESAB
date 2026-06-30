import csv
import glob
import json
import locale
import os
from collections import namedtuple
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.measurements as measurements
import scipy.ndimage.morphology as morpho
from osgeo import gdal
from pygws.service.progress_monitor import DefaultMonitor
from scipy.ndimage.filters import percentile_filter
from scipy.signal import fftconvolve
from skimage import measure

import pyat.utils.pyat_logger as log
from pyat.dtm import dtm_standard_constants, dtm_legacy_constants
from pyat.dtm.mask import crop_with_masks

# This is the object encoding all needed information about the peak finder inputs.
InputData = namedtuple("InputData", "reference_target, reference_dataset, filename")


def _extract_data(
    input_file_path: str, filename_uri: str, mask_files: List[str], logger, output_directory: str
) -> InputData:
    "Provide a InputData instance"
    input_file_name = os.path.splitext(os.path.split(input_file_path)[1])[0]
    if mask_files:
        logger.info(f"Reference dataset will now be loaded and cropped according to {len(mask_files)} mask(s)")
        outfile = output_directory + input_file_name + ".out.dtm.nc"
        reference_dataset, malformed_shape_file = crop_with_masks(filename_uri, mask_files, outfile)
        if malformed_shape_file:
            logger.warning(
                f"One of the given shape files {mask_files} isn't closed. We tried to close it manually, but this is a dirty workaround."
            )
        # os.remove(outfile)
    else:  # no mask, no warp: the input filename has yet to be opened
        logger.warning(f"Reference dataset will now be loaded")
        reference_dataset = gdal.Open(filename_uri)

    band = reference_dataset.GetRasterBand(1)
    reference_target = band.ReadAsArray(0, 0, reference_dataset.RasterXSize, reference_dataset.RasterYSize)
    nodata = band.GetNoDataValue()

    # Create a masked array for making calculations without nodata/nan values
    reference_target = np.ma.masked_equal(reference_target, nodata)
    reference_target = np.ma.masked_invalid(reference_target)

    # noinspection PyUnresolvedReferences
    scale_factor = band.GetScale()
    if scale_factor is None:
        scale_factor = 1.0
    add_offset = band.GetOffset()
    if add_offset is None:
        add_offset = 0.0
    if scale_factor != 1 and add_offset != 0:
        # apply scale factor and add offset
        reference_target = reference_target * scale_factor + add_offset

    # set nan instead of no values if need be just because geotiff export does not take mask into accout
    reference_target = np.ma.filled(reference_target, np.nan)
    return InputData(reference_target, reference_dataset, input_file_path)


class PeakFinder:
    def replace_with_mean(self, in_array, size):
        """replace each value with the mean value of its neighbors"""
        g = np.ones((2 * size + 1, 2 * size + 1))
        g[int((g.shape[0] - 1) / 2) : int((g.shape[0] - 1) / 2)] = 0
        g = (g / g.sum()).astype(in_array.dtype)
        return ndimage.convolve(in_array, g, mode="nearest")

    def _local_st_computation(self, in_array, size):
        self.logger.info("start to compute local stdev, can be long")
        local_stdev = ndimage.generic_filter(in_array, np.std, size)
        self.logger.info("done compute local stdev...")
        return local_stdev

    def _gaussian_blur(self, in_array, size):
        # expand in_array to fit edge of kernel
        padded_array = np.pad(in_array, size, "symmetric")
        # build kernel
        x, y = np.mgrid[-size : size + 1, -size : size + 1]
        g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
        # set the center point with a zero weight
        g = (g / g.sum()).astype(in_array.dtype)
        # do the Gaussian blur
        return fftconvolve(padded_array, g, mode="valid")

    def _compute_output_name(self, suffix, extension="tif", directory=None):
        """ "
        compute output file name (add a suffix) and change director if needed
        """
        if directory is None:
            return self.input.filename + "_" + suffix + "." + extension
        file = os.path.basename(self.input.filename)
        return os.path.join(directory, file + "_" + suffix + "." + extension)

    def __init__(
        self,
        i_paths: str,
        o_paths: str,
        geo_masks: List[str] = None,
        clear_output: bool = True,
        display: bool = False,
        use_gradient: bool = False,
        peak_detection_threshold: float = 10,
        percentile: float = 80,
        size: int = 5,
        use_percent: bool = True,
        percent: int = 10,
        percent_kernel: int = 2,
        use_stdev: bool = False,
        maximum_allowed_std: float = 130,
        kernel_size_for_mean_computation: float = 2,
        kernel_size_for_stdev: int = 2,
        use_holes_detection: bool = False,
        maximum_hole_area_in_pixel: int = 100,
        monitor=DefaultMonitor,
    ):
        """Initialisation of the abstract process. Initialisation of the output path with prefix
        and suffix. Check the output path if is custom. Find coordinates from a kml if a kml path is
        given.

        Arguments:
            i_paths {list} -- List of dtm file paths.

        Keyword Arguments:
            o_paths {list} -- List of dtm file paths (default: {None}).
            params {dict} -- Dict of parameters. (default: {None})
        """
        self.logger = log.logging.getLogger(self.__class__.__name__)
        self.monitor = monitor or DefaultMonitor
        if not os.path.exists(o_paths):
            os.makedirs(o_paths)
        self.output_directory = o_paths

        self.input = self._load_input(i_paths, geo_masks or [], self.output_directory)
        self.show_intermediate_plot = False
        self.show_intermediate_plot = display

        self.params = {  # for later use by computations
            "clear_output": clear_output,
            "use_gradient": use_gradient,
            "use_percent": use_percent,
            "use_stdev": use_stdev,
            "use_holes_detection": use_holes_detection,
            "percent": percent,
            "percent_kernel": percent_kernel,
            "maximum_allowed_std": maximum_allowed_std,
            "kernel_size_for_mean_computation": kernel_size_for_mean_computation,
            "kernel_size_for_stdev": kernel_size_for_stdev,
            "maximum_hole_area_in_pixel": maximum_hole_area_in_pixel,
            "peak_detection_threshold": peak_detection_threshold,
            "percentile": percentile,
            "size": size,
        }

    def detect_holes(self, maximum_hole_area_in_pixel=100):
        """
        detect real holes in the dtm, meaning that non value areas less that maximum_hole_area_in_pixel will be identified and saved to file
        """
        self.logger.info(f"Starting holes method detection")
        src_data = self.input.reference_target
        out = np.ma.masked_invalid(src_data)

        # pylint: disable=E1101
        self.display(out.mask, "holes")
        self._generate_features(out.mask, name="holes", filter_max_area=maximum_hole_area_in_pixel)
        self.logger.info(f"End of holes method detection")

    def check_gaussian_curvature(self):
        """use gaussian curvature interpolation to remove highest curvatures data"""

        src_data = self.input.reference_target.copy()
        self.display(src_data, "input data")
        self.gy, self.gx = np.gradient(src_data)
        gxy, gxx = np.gradient(self.gx)
        self.display(np.maximum(np.abs(gxx), np.abs(gxy)), "gradient")

        gyy, _ = np.gradient(self.gy)
        gauss_curv = (gxx * gyy - (gxy**2)) / (1 + (self.gx**2) + (self.gy**2)) ** 2
        self.display(gauss_curv, "gaussian curvature")
        std_gauss_curv = np.nanstd(gauss_curv)
        mean_gauss_curv = np.nanmean(gauss_curv)
        self.display(gauss_curv, "std gaussian curvature")
        estimated_curv_th = 6.0
        # correction for global roughness, should be improved probably
        if std_gauss_curv > 0.01:
            estimated_curv_th *= 2.0
        if std_gauss_curv > 0.03:
            estimated_curv_th *= 2.0
        if std_gauss_curv > 0.1:
            estimated_curv_th *= 2.0

        # keep NaN values mask
        peaks_dataset = np.ma.masked_greater(gauss_curv, estimated_curv_th)
        # dilatedmask=morpho.binary_dilation( ~peaks_dataset.mask,iterations=3)
        # pylint: disable=E1101
        dilatedmask = peaks_dataset.mask
        self.display(dilatedmask, "Point of interest filtered")
        self._generate_features(dilatedmask)
        self._show_and_wait()

    @staticmethod
    def _show_and_wait():
        """display figure and wait"""
        plt.show(block=True)

    def _get_feature_centers(self, data, filter_max_area=None):
        """return a list of (longitude, latitude) tuples of the feature centers"""
        result = []
        all_labels, n = measure.label(data, return_num=True)
        regions = measure.regionprops(all_labels)
        # objects= measurements.find_objects(all_labels)
        centers = measurements.center_of_mass(data, all_labels, range(1, n + 1))
        # we have center point, compute their positions
        (
            ulx,
            xres,
            xskew,
            uly,
            yskew,
            yres,
        ) = self.input.reference_dataset.GetGeoTransform()
        regionIndex = 0
        for c in centers:
            posX = ulx + c[1] * xres
            posY = uly + c[0] * yres
            if filter_max_area is not None:
                retain = regions[regionIndex].area < filter_max_area
            else:
                retain = True
            if retain:
                result.append((posX, posY))
            regionIndex = regionIndex + 1

        return result

    def _generate_features_json(self, data, name="json", filter_max_area=None):
        """export features in a GeoJSON FeatureCollection"""

        result = {"type": "FeatureCollection", "features": []}
        centers = self._get_feature_centers(data, filter_max_area)
        for c in centers:
            result["features"].append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [c[0], c[1]]},
                    "properties": {},
                }
            )
        filename = self._compute_output_name(name, extension="json", directory=self.output_directory)
        with open(filename, mode="w", encoding=locale.getpreferredencoding()) as jsonFile:
            json.dump(result, jsonFile, indent="\t")
        print("Created " + str(len(centers)) + " region of interest in " + self.output_directory + " directory")

    def _generate_features(self, data, name="outlier", filter_max_area=None):
        """export features in Globe-compatible CSV format"""
        self.logger.info(f"Generate named features with prefix {name}")
        centers = self._get_feature_centers(data, filter_max_area)
        outputCount = 0
        for c in centers:
            self._create_csv_file(name + "_" + str(outputCount), longitude=c[0], latitude=c[1])
            outputCount = outputCount + 1
        self.logger.info(f"Generate {outputCount} files")

    def display(self, data, label: str = None):
        """if display result is enabled open a figure and display data"""
        if self.show_intermediate_plot:
            plt.figure()
            plt.imshow(data)
            if data.dtype != bool:
                plt.colorbar()
            if label:
                plt.title(label)

    def check_gradient_method(self, percentile=80, size=5, threshold=10):
        self.logger.info(f"Starting gradient method detection")

        src_data = self.input.reference_target
        self.display(src_data, "source data")
        # geotiff_filename=self._compute_output_name("source_data", directory=tmp.mkdtemp())
        # self._createGeotiff(
        #     geotiff_filename,
        #     src_data,
        #     self.input.reference_dataset.RasterXSize,
        #     self.input.reference_dataset.RasterYSize,
        #     self.input.reference_dataset.GetGeoTransform(),
        #     self.input.reference_dataset.GetProjection(),
        # )
        self.logger.info(f"Compute gradient")

        # compute gradient for now
        gradient = np.gradient(src_data)
        gradient = np.ma.masked_invalid(gradient)
        # pylint: disable=E1101
        gradient_mask = morpho.binary_erosion(gradient.mask, iterations=1)
        gradient[gradient_mask] = np.nan
        gradient = np.ma.masked_invalid(gradient)

        gradient = np.maximum(np.abs(gradient[0]), np.abs(gradient[1]))
        self.display(gradient, "maximum slopes")
        # TO BE TESTED
        # gdal.DEMProcessing('slope.tif', DEM, 'slope')

        # filter gradient to keep only the ones higher than their neighbors
        self.logger.info(f"Compute percentile filter")
        filtered = percentile_filter(gradient, percentile, size)
        self.display(filtered, "filtered")
        self.logger.info(f"Normalizing gradient")
        peaks_dataset = np.abs(gradient / filtered)
        self.display(peaks_dataset, "gradient / filtered")

        stdev_min = np.nanmin(peaks_dataset)
        stdev_max = np.nanmax(peaks_dataset)
        stdev_mean = np.nanmean(peaks_dataset)
        self.logger.info(f"normalized gradient statistics min {stdev_min}, max {stdev_max}, mean {stdev_mean}")

        # mask 0 data
        # pylint: disable=E1101
        self.logger.info(f"Filter percentile filter")

        peaks_dataset = np.ma.masked_greater(peaks_dataset, threshold)
        self.logger.info(f"Apply binary dilatation")
        dilatedmask = morpho.binary_dilation(peaks_dataset.mask, iterations=3)
        self.display(dilatedmask, "Point of interest filtered")

        self._generate_features(dilatedmask, name="gradient_detection")

        self.logger.info(f"End of gradient method")

    def _load_input(self, input_file: str, mask_files: List[str], output_directory: str):
        """load input data, either dtm or Geotiff"""
        input_dataset: InputData
        if input_file.endswith("mnt") or input_file.endswith("dtm"):
            filename_uri = f"NETCDF:{input_file}:{dtm_legacy_constants.VARIABLE_DEPTH}"
        elif input_file.endswith(".nc"):
            filename_uri = f"NETCDF:{input_file}:{dtm_standard_constants.ELEVATION_NAME}"
        else:
            filename_uri = input_file
        return _extract_data(input_file, filename_uri, mask_files, self.logger, output_directory)

    def check_percent(self, depth_percent=10, kernel_size_for_mean_computation=2):
        """
        algorithm for base on detecting local threshold over a given percent of depth
        :return:
        """
        self.logger.info(f"Starting percent method detection")

        reference_target = self.input.reference_target.copy()
        self.display(reference_target, "source data")
        self.logger.info(f"Compute mean values")

        blurred = self.replace_with_mean(reference_target, kernel_size_for_mean_computation)
        self.display(
            blurred,
            "filtered data all values are replaced by the mean of neighbors with a kernel size = "
            + str(kernel_size_for_mean_computation),
        )
        self.logger.info(f"Compute local difference with mean values")

        peaks_dataset = np.abs(blurred - reference_target)
        self.display(peaks_dataset, "peaks dataset (filtered data - source data)")

        # we need to mask values where difference is higher than the depth tolerance
        self.logger.info(f"Compute threshold values")
        mask_relative_to_depth = np.ma.masked_less(peaks_dataset, np.abs(reference_target * (0.01 * depth_percent)))
        out = np.ma.masked_invalid(mask_relative_to_depth)

        # self.display(out, "peaks filtered, keep values more than " + str(depth_percent) + "% of depth")
        # pylint: disable=E1101
        self.logger.info(f"Compute binary dilataion")
        dilatedmask = morpho.binary_dilation(~out.mask, iterations=2)
        self.display(
            dilatedmask,
            "peaks filtered, keep values more than " + str(depth_percent) + "% of depth",
        )
        self._generate_features(dilatedmask, name="peaks_percent")
        self.logger.info(f"End of percent method detection")

    # pylint: disable=W0613
    def check_std(
        self,
        maximum_allowed_std=130,
        kernel_size_for_mean_computation=2,
        kernel_size_for_stdev=2,
        stdev_factor=3,
    ):
        """
        algorithm for peak detection detect values above a
        :return:
        """
        self.logger.info(f"Starting stdev method detection")

        reference_target = self.input.reference_target.copy()
        self.display(reference_target, "source data")

        self.logger.info(f"compute mean values")
        blurred = self.replace_with_mean(reference_target, kernel_size_for_mean_computation)

        self.display(
            blurred,
            "filtered data all values are replaced by the mean of neighbors with a kernel size = "
            + str(kernel_size_for_mean_computation),
        )
        # plt.show(False)
        self.logger.info(f"compute difference with smoothed dtm")
        peaks_dataset = np.abs(blurred - reference_target)
        self.display(peaks_dataset, "peaks dataset (filtered data - source data)")

        self.logger.info(f"compute local stdev with kernel size {kernel_size_for_stdev}")
        local_stdev = self._local_st_computation(reference_target, kernel_size_for_stdev)
        self.display(local_stdev, "local stdev")

        stdev_min = np.nanmin(local_stdev)
        stdev_max = np.nanmax(local_stdev)
        stdev_mean = np.nanmean(local_stdev)
        self.logger.info(f"stdev statistics min {stdev_min}, max {stdev_max}, mean {stdev_mean}")

        # filter stdev to ensure that no more than maximum_allowed_std is encountered
        # pylint: disable=E1101
        highest_std_map = np.ma.masked_less(peaks_dataset, maximum_allowed_std)
        highest_std_map = np.ma.masked_invalid(highest_std_map)
        self.logger.info(f"compute binary dilation")
        highest_std_mask = morpho.binary_dilation(~highest_std_map.mask, iterations=2)
        self.display(
            highest_std_mask,
            "dilated over maximum allowed values" + str(maximum_allowed_std),
        )
        self._generate_features(highest_std_mask, name="local_stdev_higher_than_" + str(maximum_allowed_std))
        self.logger.info(f"End of stdev method")

    #     self.generate_features(dilatedmask,name="peaks having difference to local mean that is more than 3 stdev")

    # self.createGeotiff(self.compute_output_name("peaks"),peaks_dataset,reference_dataset.RasterXSize,reference_dataset.RasterYSize,reference_dataset.GetGeoTransform(),reference_dataset.GetProjection())

    # OK now try to extract features and their centroid
    # make a dilation of filtered mask

    def _create_csv_file(self, filename, latitude: float, longitude: float):
        """
        Create a cvs file in the Globe format, create in this format one point at immersion clamped to ground
        :param filename: destination file name
        :param latitude: latitude of the point
        :param longitude: longitude of the point
        :return: None
        """

        filename = self._compute_output_name(filename, extension="csv", directory=self.output_directory)
        with open(filename, "w", newline="", encoding=locale.getpreferredencoding()) as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(
                [
                    '"ID"',
                    '"LATITUDE_DEG"',
                    '"LONGITUDE_DEG"',
                    '"LATITUDE_DMD"',
                    '"LONGITUDE_DMD"',
                    '"HEIGHT_ABOVE_SEA_SURFACE"',
                    '"SEA_FLOOR_LAYER"',
                    '"MARKER_COLOR"',
                    '"MARKER_SIZE"',
                    '"MARKER_SHAPE"',
                    '"GROUP"',
                    '"CLASS"',
                    '"COMMENT"',
                ]
            )
            spamwriter.writerow(
                [
                    '"001"',
                    '"' + str(latitude) + '"',
                    '"' + str(longitude) + '"',
                    '""',
                    '""',
                    '"0"',
                    '"Globe default layer"',
                    '"#ff0000ff"',
                    '"50"',
                    '"Sphere"',
                    '""',
                    '""',
                    '""',
                ]
            )

    def _createGeotiff(self, filename, array, sizeX, sizeY, geotransform, projection):
        """
        Create one geotiff file and fill it with the array in parameter
        :param filename: the destination filename
        :param array: the data array
        :param sizeX: X lenght
        :param sizeY: Y lenght
        :param geotransform: the GDAL geotransform associated with the geotiff
        :param projection: the GDAL projection associated with the geotiff
        :return:
        """
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename, sizeX, sizeY, 1, gdal.GDT_Float64)
        dataset.SetGeoTransform(geotransform)
        dataset.SetProjection(projection)
        dataset.GetRasterBand(1).WriteArray(array)
        dataset.FlushCache()  # Write to disk.

    def __call__(self) -> None:
        self.runAll()

    def runAll(self):
        if "clear_output" in self.params:
            self.logger.info(f"Clearing output directory {self.output_directory}")
            # Get a list of all the file paths that ends with .txt from in specified directory
            fileList = glob.glob(f"{self.output_directory}/*.csv")
            # Iterate over the list of filepaths & remove each file.
            for filePath in fileList:
                try:
                    os.remove(filePath)
                except Exception:
                    self.logger.warning(f"Error while deleting file : {filePath}")

        # run methods
        if "use_gradient" in self.params and self.params["use_gradient"]:
            percentile = float(self.params["percentile"])
            size = int(self.params["size"])
            threshold = int(self.params["peak_detection_threshold"])
            self.check_gradient_method(percentile=percentile, size=size, threshold=threshold)

        if "use_percent" in self.params and self.params["use_percent"]:
            percent = int(self.params["percent"])
            percent_kernel = int(self.params["percent_kernel"])
            self.check_percent(depth_percent=percent, kernel_size_for_mean_computation=percent_kernel)

        if "use_stdev" in self.params and self.params["use_stdev"]:
            maximum_allowed_std = float(self.params["maximum_allowed_std"])
            kernel_size_for_mean_computation = int(self.params["kernel_size_for_mean_computation"])
            kernel_size_for_stdev = int(self.params["kernel_size_for_stdev"])
            self.check_std(
                maximum_allowed_std=maximum_allowed_std,
                kernel_size_for_mean_computation=kernel_size_for_mean_computation,
                kernel_size_for_stdev=kernel_size_for_stdev,
            )

        if "use_holes_detection" in self.params and self.params["use_holes_detection"]:
            maximum_hole_area_in_pixel = int(self.params["maximum_hole_area_in_pixel"])
            self.detect_holes(maximum_hole_area_in_pixel=maximum_hole_area_in_pixel)

        if self.show_intermediate_plot:
            self._show_and_wait()
