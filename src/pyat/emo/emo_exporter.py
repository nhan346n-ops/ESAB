#! /usr/bin/env python3
# coding: utf-8

import datetime
from typing import List

import numpy
from pygws.service.progress_monitor import DefaultMonitor

import pyat.dtm.dtm_driver as dtm_driver
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.dtm.utils.dtm_utils as dtm_utils
import pyat.dtm.utils.process_utils as process_util
import pyat.emo.emo_constants as EmoConstants
import pyat.emo.emo_driver as emo_driver
import pyat.utils.numpy_utils as NumpyUtils
import pyat.utils.pyat_logger as log


class ToDtmExporter:
    """
    Utility class to export an emo file as a dtm (netcdf4 format)
    """

    def __init__(
        self,
        i_paths: list,
        o_paths: list = None,
        overwrite: bool = False,
        monitor=DefaultMonitor,
    ):
        """
        Constructor.
        :param : i_paths : path of the imput file to convert
        :param : o_paths : resulting dtm file path

        Raised exceptions :
            - FileNotFoundError when emoFilePath does not exist
            - PermissionError when emoFilePath is not readable or dtmFilePath is not writable
            - IOError when emoFilePath is not a suitable emo file
        """
        self.overwrite = overwrite
        self.monitor = monitor
        self.logger = log.logging.getLogger(self.__class__.__name__)

        self.emoDriver = emo_driver.EmoDriver()

        self._emo_files = [self.emoDriver.check_path(path) for path in i_paths]

        if o_paths:
            self._dtm_drivers = [dtm_driver.DtmDriver(path) for path in o_paths]
        else:
            # Create output name from the input with the nc extension.
            self._dtm_drivers = [
                dtm_driver.DtmDriver(path[: path.rfind(".")] + DtmConstants.EXTENSION_NC) for path in i_paths
            ]

        if len(self.emo_files) != len(self.dtm_drivers):
            raise AttributeError("Number of Output/Input paths must be the same.")

    @property
    def emo_files(self) -> List[emo_driver.EmoFile]:
        return self._emo_files

    @property
    def dtm_drivers(self) -> List[dtm_driver.DtmDriver]:
        return self._dtm_drivers

    def export(self, emoFile: emo_driver.EmoFile, driver: dtm_driver.DtmDriver, monitor) -> None:
        """
        Launch the export
        :param : progressCallback : monitor the progress of the activity (0.0 = started ... 1.0 = finished)
        Raised exception : IOError when error occurs while parsing the file
        """
        self.logger.info(f"Starting to convert {emoFile.file_path} to {driver.dtm_file.file_path}")
        now = datetime.datetime.now()
        monitor.set_work_remaining(2)

        self.logger.info("Opening emo file, extracting extent...")
        self.emoDriver.readExtent(emoFile, monitor)  # 10% of time to read extent
        self.logger.info(f"Number of lines in the emo file : {emoFile.lineCount}")
        self.logger.info(
            f"Extent of emo file : west={emoFile.west}, east={emoFile.east}, south={emoFile.south}, north={emoFile.north}"
        )

        row_count = dtm_utils.estimate_row(emoFile.north, emoFile.south, emoFile.spatialResolution)
        col_count = dtm_utils.estimate_col(
            right_or_east=emoFile.east, left_or_west=emoFile.west, spatial_resolution=emoFile.spatialResolution
        )
        self.logger.info(f"Initializing Dtm file with {col_count} columns and {row_count} rows")

        self.logger.info("Creating dtm file")
        with driver.create_file(
            col_count,
            emoFile.west,
            emoFile.spatialResolution,
            row_count,
            emoFile.south,
            emoFile.spatialResolution,
            overwrite=self.overwrite,
        ) as dataset:

            # Mapping between Emo column and Dtm layer
            dtmEmoMapping = {
                DtmConstants.ELEVATION_NAME: EmoConstants.COL_MEAN_DEPTH,
                DtmConstants.ELEVATION_MIN: EmoConstants.COL_MAX_DEPTH,  # due to -1 factor we need to reverse min and max
                DtmConstants.ELEVATION_MAX: EmoConstants.COL_MIN_DEPTH,  # due to -1 factor we need to reverse min and max
                DtmConstants.VALUE_COUNT: EmoConstants.COL_NB_OF_SOUNDS,
                DtmConstants.STDEV: EmoConstants.COL_STDEV,
                DtmConstants.CDI_INDEX: EmoConstants.COL_CDIID,
                DtmConstants.ELEVATION_SMOOTHED_NAME: EmoConstants.COL_SMOOTHED_DEPTH,
                DtmConstants.INTERPOLATION_FLAG: EmoConstants.COL_INTERPOLATED_CELL,
            }

            self.logger.info("Initializing dtm layers...")
            rasters = {}
            for dtmLayer in dtmEmoMapping.keys():
                fillValue = dtm_driver.get_missing_value(dtmLayer)
                rasters[dtmLayer] = numpy.full(
                    (row_count, col_count), fill_value=fillValue, dtype=dtm_driver.get_type(dtmLayer)
                )

            self.logger.info("Parsing emo lines...")
            sub_monitor = monitor.split(1)
            with open(emoFile.file_path, mode="rt", encoding="utf8") as openedFile:
                try:
                    # noinspection PyNoneFunctionAssignment
                    textFileReader = self.emoDriver.parse(emoFile)
                    lineCount = 0

                    for chunk in textFileReader:
                        # CDI indexes can be stored in CDIID or DTM_SOURCE columns, so before next steps :
                        # all indexes are put into the CDIID column
                        chunk[EmoConstants.COL_CDIID] = chunk[
                            [EmoConstants.COL_CDIID, EmoConstants.COL_DTM_SOURCE]
                        ].apply(lambda x: x[1] if numpy.isnan(x[0]) else x[0], axis=1)

                        # transform longitude/latitude into column/row
                        npChunk = chunk.to_numpy(copy=False)
                        if lineCount == 0:
                            sub_monitor.set_work_remaining(int(emoFile.lineCount / npChunk.shape[0]))
                        driver.dtm_file.project(npChunk, 0, 1)

                        # aggregates rasters
                        for dtmLayer, emoColumn in dtmEmoMapping.items():
                            if emoColumn in [
                                EmoConstants.COL_MIN_DEPTH,
                                EmoConstants.COL_MAX_DEPTH,
                                EmoConstants.COL_MEAN_DEPTH,
                                EmoConstants.COL_SMOOTHED_DEPTH,
                            ]:
                                factor = -1
                            else:
                                factor = 1

                            colIndex = emo_driver.EmoFile.ColumnNames.index(emoColumn)
                            NumpyUtils.aggregate(
                                npChunk,
                                emo_driver.EmoFile.ColumnNames.index(EmoConstants.COL_LONGITUDE),
                                emo_driver.EmoFile.ColumnNames.index(EmoConstants.COL_LATITUDE),
                                colIndex,
                                rasters[dtmLayer],
                                factor,
                            )

                        # Progression
                        lineCount += npChunk.shape[0]
                        self.logger.info(f"{(lineCount / emoFile.lineCount):.2%} processed")
                        # 85% of time to read extent
                        sub_monitor.worked(1)

                        del npChunk
                        del chunk

                except ValueError as error:
                    raise IOError(f"Bad emo file. {error}") from error

            # Write layers
            self.logger.info("Writing dtm layers...")

            dataset.history = "Convert with Python " + __file__ + "MigrateDtm script from " + emoFile.file_path
            for dtmLayer, raster in rasters.items():
                driver.add_layer(dtmLayer, raster)
                del raster
            del rasters

            # Write layers CDI
            self.logger.info("Writing layer of CDIs...")
            cdis = [cdi for cdiId, cdi in emoFile.cdis.values()]
            driver.create_cdi_reference_variable(cdis=cdis)

        monitor.done()
        self.logger.info(f"End of conversion for {emoFile.file_path} : {datetime.datetime.now() - now} time elapsed\n")

    def __call__(self) -> None:
        """Run method."""
        begin = datetime.datetime.now()
        self.monitor.set_work_remaining(len(self.emo_files))
        an_error_occurred = False
        file_in_error = []
        for ind, emo_file in enumerate(self.emo_files):
            sub_monitor = self.monitor.split(1)
            try:
                self.export(emo_file, self.dtm_drivers[ind], sub_monitor)

            except Exception as error:
                file_in_error.append(emo_file.file_path)
                an_error_occurred = True
                self.logger.error("An exception was thrown!", exc_info=True, stack_info=True)

        process_util.log_result(self.logger, begin, file_in_error)
