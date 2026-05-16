import logging
import os
import tempfile as tmp
from pathlib import Path
from typing import Dict, List

import matplotlib.colors as mcolors
import numpy as np
from osgeo import gdal
from pygws.service.progress_monitor import DefaultMonitor

import pyat.dtm.dtm_standard_constants as DtmConstants
from pyat.dtm.export.dtm_to_isobath import Dtm2Isobath
from pyat.utils.gdal_utils import GDALDataset, apply_colormap, gdal_progress_callback


class Dtm2Mbtiles:
    """
    Exports a dtm to mbtiles file.
    """

    def __init__(
        self,
        i_paths: List[str],
        o_paths: List[str],
        overwrite: bool = False,
        azimuth: float = 315,
        altitude: float = 45,
        zfactor: float = 1,
        add_isobath: bool = False,
        isobath_interval: float = 50,
        monitor=DefaultMonitor,
    ):
        """Init method."""
        self.i_paths = i_paths
        self.o_paths = o_paths
        self.overwrite = overwrite
        self.azimuth = azimuth
        self.altitude = altitude
        self.zfactor = zfactor
        self.add_isobath = add_isobath
        self.isobath_interval = isobath_interval
        self.monitor = monitor
        self.resulting_files = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self) -> Dict:
        """
        Main function to export DTMs to MBTiles
        """
        self.monitor.begin_task("Start exporting DTM to MBTiles", len(self.i_paths))

        for i_path, o_path in zip(self.i_paths, self.o_paths):
            if self.monitor.check_cancelled():
                self.logger.warning("Cancel requested. Export aborted")
                break

            if not os.path.exists(o_path) or self.overwrite:
                self.logger.info(f"Creating file {o_path}")
                if self.__process_export(i_path, o_path):
                    self.resulting_files.append(o_path)
            else:
                self.logger.warning(f"{o_path} exists and cannot be overwritten")

            self.monitor.worked(1)
        self.monitor.done()

        return {"outfile": [str(file_path) for file_path in self.resulting_files]}

    def __process_export(self, i_path: str, o_path: str) -> bool:
        """
        Open the DTM, process hillshade and color-relief, blend them, eventualy overlay isobath, and export to MBTiles
        """
        result = False
        src_path = f"NETCDF:{i_path}:{DtmConstants.ELEVATION_NAME}"

        with tmp.TemporaryDirectory(suffix="_pyat_dtm2mbtiles") as tmpdir:
            with GDALDataset(src_path) as dtm:

                # Create temporary hillshade
                hillshade_path = os.path.join(tmpdir, "hillshade.tif")
                if not self.__create_hillshade(dtm, hillshade_path):
                    self.logger.error(f"Failed to create hillshade for {i_path}")
                    return result

                # Create temporary color-relief
                color_relief_path = os.path.join(tmpdir, "color_relief.tif")
                if not apply_colormap(dtm, color_relief_path, monitor=self.monitor.split(1), logger=self.logger):
                    self.logger.error(f"Failed to create color-relief for {i_path}")
                    return result

            # Blend hillshade and color-relief
            blended_path = os.path.join(tmpdir, "blended.tif")
            if not self.__blend_tiffs(color_relief_path, hillshade_path, blended_path):
                self.logger.error(f"Failed to blend hillshade and color-relief for {i_path}")
                return result

            if self.add_isobath:
                # create isobath if requested using Dtm2isobath
                isobath_path = os.path.join(tmpdir, "isobath.gpkg")
                exporter = Dtm2Isobath(
                    i_paths=[str(i_path)],
                    o_paths=[isobath_path],
                    overwrite=False,
                    isobath_interval=self.isobath_interval,
                    monitor=self.monitor.split(1),
                )
                exporter()

                # rasterize isobath
                if not self.__rasterize_isobath(blended_path, isobath_path):
                    self.logger.error(f"Failed to rasterize isobath for {i_path}")
                    return result

                # ovelay isobath raster over blended hillshade+color-relief
                isobath_path = Path(isobath_path).with_suffix(".tif")
                self.__overlay_rasters(blended_path, str(isobath_path))

            # Create MBTiles from blended raster
            if not self.__create_mbtiles(blended_path, o_path, os.path.basename(i_path)):
                self.logger.error(f"Failed to create MBTiles for {i_path}")
                return result

            result = True

        return result

    def __create_hillshade(self, dtm_ds: GDALDataset, hillshade_path: str) -> bool:
        """
        Create hillshade from DTM dataset and save to specified path
        """
        hillshade_options = gdal.DEMProcessingOptions(
            format="GTiff",
            azimuth=self.azimuth,
            altitude=self.altitude,
            zFactor=self.zfactor,
            computeEdges=True,
            callback=gdal_progress_callback,
            callback_data=[0, "hillshading", self.monitor.split(1)],
        )
        try:
            hillshade_ds = gdal.DEMProcessing(
                destName=hillshade_path,
                srcDS=dtm_ds,
                processing="hillshade",
                options=hillshade_options,
            )
            if hillshade_ds is None:
                self.logger.error(f"Failed to create hillshade for {dtm_ds.GetDescription()}")
                return False
            hillshade_ds = None
            return True

        except Exception as e:
            self.logger.error(f"Error creating hillshade: {e}")
            return False

    def __blend_tiffs(self, color_relief_path: str, hillshade_path: str, output_path: str) -> bool:
        """
        Blend RGBA relief and hillshade using HSV color space, preserving the alpha channel from the RGBA relief.
        - rgba_path: path to the input RGBA relief raster (must have 4 bands)
        - shade_path: path to the input hillshade raster (must have 1 band, values normalized between 0 and 255)
        - output_path: path to the output blended RGBA raster
        """
        try:
            with GDALDataset(color_relief_path) as ds_rgba, GDALDataset(hillshade_path) as ds_shade:
                if ds_rgba.RasterCount < 4:
                    raise ValueError("Input raster does not have 4 bands (RGBA).")

                # Read the existing Alpha channel first
                alpha = ds_rgba.GetRasterBand(4).ReadAsArray()

                # Read and Normalize RGB (Bands 1, 2, 3)
                r = ds_rgba.GetRasterBand(1).ReadAsArray().astype(np.float32) / 255.0
                g = ds_rgba.GetRasterBand(2).ReadAsArray().astype(np.float32) / 255.0
                b = ds_rgba.GetRasterBand(3).ReadAsArray().astype(np.float32) / 255.0
                rgb = np.dstack((r, g, b))

                # Read Hillshade (Value channel)
                shade = ds_shade.GetRasterBand(1).ReadAsArray().astype(np.float32) / 255.0

                # Perform HSV Blend
                # Convert RGB to HSV color space
                hsv = mcolors.rgb_to_hsv(rgb)

                # Replace the 'Value' (brightness) with the hillshade intensity
                hsv[..., 2] = shade

                # Convert back to RGB
                final_rgb = mcolors.hsv_to_rgb(hsv) * 255.0
                final_rgb = np.clip(final_rgb, 0, 255).astype(np.uint8)

                # Create 4-band RGBA Output
                driver = gdal.GetDriverByName("GTiff")
                out_ds = driver.Create(output_path, ds_rgba.RasterXSize, ds_rgba.RasterYSize, 4, gdal.GDT_Byte)
                out_ds.SetProjection(ds_rgba.GetProjection())
                out_ds.SetGeoTransform(ds_rgba.GetGeoTransform())

                # Write the new RGB bands
                for i in range(3):
                    out_ds.GetRasterBand(i + 1).WriteArray(final_rgb[..., i])

                # Write the ORIGINAL Alpha band back
                out_ds.GetRasterBand(4).WriteArray(alpha)
                out_ds.FlushCache()
                return True
        except Exception as e:
            self.logger.error(f"Error blending RGBA and hillshade: {e}")
            return False

    def __rasterize_isobath(self, blended_path: str, isobath_path: str) -> bool:
        """
        Rasterize isobath shapefile
        """
        try:
            with GDALDataset(blended_path) as blended_ds:
                width = blended_ds.RasterXSize
                height = blended_ds.RasterYSize
                gt = blended_ds.GetGeoTransform()
                proj = blended_ds.GetProjection()
                band_count = blended_ds.RasterCount

                # Create a transparent 1-band raster for the lines using 'Byte' with a NoData value of 0
                drv = gdal.GetDriverByName("GTiff")
                isobath_raster_path = Path(isobath_path).with_suffix(".tif")
                isobath_ds = drv.Create(isobath_raster_path, width, height, band_count, gdal.GDT_Byte)
                isobath_ds.SetGeoTransform(gt)
                isobath_ds.SetProjection(proj)
                band = isobath_ds.GetRasterBand(1)
                band.SetNoDataValue(0)

                # Rasterize the vector onto this new layer : burn value 255 on all bands (resulting in white lines)
                vector_ds = gdal.OpenEx(isobath_path, gdal.OF_VECTOR)
                layer = vector_ds.GetLayer()
                gdal.RasterizeLayer(
                    isobath_ds,
                    np.arange(1, band_count + 1).tolist(),
                    layer,
                    burn_values=np.repeat(255, band_count).tolist(),
                    callback=gdal_progress_callback,
                    callback_data=[0, "rasterizing isobaths", self.monitor.split(1)],
                )
                vector_ds = None
                isobath_ds = None  # Flush to disk
                return True

        except Exception as e:
            self.logger.error(f"Error rasterizing isobath: {e}")
            return False

    def __overlay_rasters(self, base_raster: str, top_raster: str) -> bool:
        """
        Overlay top rasters over base raster using numpy : non-transparent pixels from top replace base pixels
        - base_raster: path to the base raster (must have 3 or 4 bands)
        - top_raster: path to the top raster (must have 4 bands with alpha channel)
        """
        try:
            with GDALDataset(base_raster, mode="r+") as ds_base, GDALDataset(top_raster) as ds_top:
                if ds_base.RasterXSize != ds_top.RasterXSize or ds_base.RasterYSize != ds_top.RasterYSize:
                    raise ValueError("Input rasters must have the same dimensions.")

                # Read base raster
                base_bands = [ds_base.GetRasterBand(i + 1).ReadAsArray() for i in range(ds_base.RasterCount)]
                base_array = np.dstack(base_bands)

                # Read top raster
                top_bands = [ds_top.GetRasterBand(i + 1).ReadAsArray() for i in range(ds_top.RasterCount)]
                top_array = np.dstack(top_bands)

                # put non-transparent (alpha > 0) pixels from top raster in place of base raster pixels
                overlayed_rgb = np.where(top_array[..., 3:] > 0, top_array[..., :3], base_array[..., :3]).astype(
                    np.uint8
                )

                # update base raster with overlayed RGB (keep original alpha if exists, else set to 255)
                if ds_base.RasterCount == 4:
                    blended_rgba = np.dstack((overlayed_rgb, base_array[..., 3]))
                else:
                    blended_rgba = np.dstack((overlayed_rgb, np.full(overlayed_rgb.shape[0:2], 255, dtype=np.uint8)))

                # Write blended RGB bands
                for i in range(3):
                    ds_base.GetRasterBand(i + 1).WriteArray(blended_rgba[..., i])
                ds_base.FlushCache()
                return True

        except Exception as e:
            self.logger.error(f"Error overlaying ratsers: {e}")
            return False

    def __create_mbtiles(self, input_path: str, output_path: str, orginal_filename: str) -> bool:
        """
        Create MBTiles file from blended raster, eventualy adding isobath as an overlay if provided.
        As described in GDAL MBtiles driver documentation, automatic reprojection of the input dataset to EPSG:3857 (Pseudo-Mercator) will be done, with selection of the appropriate zoom level.
        """
        try:
            mbtiles_driver = gdal.GetDriverByName("MBTiles")
            if mbtiles_driver is None:
                self.logger.error("MBTiles driver not available in GDAL")
                return False

            with GDALDataset(input_path) as i_ds:
                o_ds = mbtiles_driver.CreateCopy(
                    output_path,
                    i_ds,
                    options=[
                        f"DESCRIPTION=generated from '{orginal_filename}' by {self.__class__.__name__} process of pyAT",
                    ],
                    callback=gdal_progress_callback,
                    callback_data=[0, "creating MBTiles", self.monitor.split(1)],
                )

                if o_ds is None:
                    self.logger.error(f"Failed to create MBTiles file {output_path}")
                    return False

                o_ds = None
                return True

        except Exception as e:
            self.logger.error(f"Error creating MBTiles: {e}")
            return False
