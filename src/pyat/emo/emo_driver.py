import os

import numpy
import pandas as pd

import pyat.emo.emo_constants as EMO
import pyat.dtm.dtm_standard_constants as DtmConstants
import pyat.utils.numpy_utils as NumpyUtils
import pyat.common.geo_file as gf
from pyat.utils.exceptions.exception_list import ProcessingError


class EmoFile(gf.GeoFile):
    """
    emo file's properties.
    """

    # Column's type in a emo file
    ColumnDescriptions = {
        EMO.COL_LONGITUDE: float,
        EMO.COL_LATITUDE: float,
        EMO.COL_MIN_DEPTH: float,
        EMO.COL_MAX_DEPTH: float,
        EMO.COL_MEAN_DEPTH: float,
        EMO.COL_STDEV: float,
        EMO.COL_NB_OF_SOUNDS: float,  # int,
        EMO.COL_INTERPOLATED_CELL: float,  # int,
        EMO.COL_SMOOTHED_DEPTH: float,
        EMO.COL_SMOOTHED_MEAN_DIFFERENCE: float,
        EMO.COL_CDIID: str,
        EMO.COL_DTM_SOURCE: str,
    }
    # Column's order in a emo file
    ColumnNames = list(ColumnDescriptions.keys())

    def __init__(self, filePath):
        super().__init__(filePath)
        self.minmax = {
            key: [numpy.nan, numpy.nan] for (key, value) in EmoFile.ColumnDescriptions.items() if value != str
        }
        self.lineCount = 0
        self.spatialResolution = numpy.nan
        self.cdis = {}

    @property
    def lineCount(self):
        return self._lineCount

    @lineCount.setter
    def lineCount(self, lineCount):
        self._lineCount = lineCount

    @property
    def spatialResolution(self):
        return self._spatialResolution

    @spatialResolution.setter
    def spatialResolution(self, spatialResolution: float):
        self._spatialResolution = spatialResolution

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, sources):
        self._sources = sources

    @property
    def cdis(self):
        return self._cdis

    @cdis.setter
    def cdis(self, cdis):
        self._cdis = cdis

    def cdiId(self, cdi: str):
        """
        return the Id of the CDI
        """
        if not cdi:
            return numpy.nan
        result = numpy.nan
        if cdi in self._cdis:
            result = self._cdis[cdi][0]
        else:
            result = len(self._cdis)
            self._cdis[cdi] = [result, "SDN:CDI:LOCAL:" + cdi]
        return result

    def sourceId(self, source: str):
        """
        return the Id of the source
        """
        if not source:
            return numpy.nan
        result = numpy.nan
        if source in self._cdis:
            result = self._cdis[source][0]
        else:
            result = len(self._cdis)
            if source == EMO.INTERPOLATED_CDI_MARKER:
                self._cdis[source] = [result, DtmConstants.INTERPOLATED_CDI]
            else:
                self._cdis[source] = [result, "SDN:CPRD:LOCAL:" + source]
        return result

    def min(self, columnName):
        return self.minmax[columnName][0]

    def max(self, columnName):
        return self.minmax[columnName][1]

    @property
    def minmax(self):
        return self._minmax

    @minmax.setter
    def minmax(self, minmax):
        self._minmax = minmax


class EmoDriver:
    """
    Loader of emo file.
    Return a EmoFile when a emo file is parsed
    """

    def computeMinMax(self, emoArray, emoFile):
        emoMinMax = numpy.full((2, 2), numpy.nan)
        NumpyUtils.minMaxOnFloat(emoArray, emoMinMax)
        for colIndex in range(emoMinMax.shape[1]):
            if not numpy.isnan(emoMinMax[0, colIndex]) and not numpy.isnan(emoMinMax[1, colIndex]):
                column = EmoFile.ColumnNames[colIndex]
                if column in ["minDepth", "maxDepth", "meanDepth", "smoothedDepth"]:
                    emoFile.minmax[column] = [
                        min(emoFile.min(column), -emoMinMax[1, colIndex]),
                        max(emoFile.max(column), -emoMinMax[0, colIndex]),
                    ]
                else:
                    emoFile.minmax[column] = [
                        min(emoFile.min(column), emoMinMax[0, colIndex]),
                        max(emoFile.max(column), emoMinMax[1, colIndex]),
                    ]

    def check_path(self, filePath: str, overwrite: bool = False, mode: str = "r") -> EmoFile:
        """
        Verify that filePath is a suitable emo file

        Raised exception :  OSError when file does not exist, is not readable or is not a suitable emo file
        """
        if mode == "r":
            result = EmoFile(filePath)
            # Evaluating the spatial resolution is usefull to test
            # if the file is not in a wrong format
            self.evaluateSpatialResolution(result)
            return result
        else:
            if not os.path.exists(filePath) or overwrite:
                return EmoFile(filePath)
            else:
                raise FileExistsError(
                    "File already exists and overwrite not allowed (allow overwrite with option : '-ow --overwrite)"
                )

    def readExtent(self, emoFile: EmoFile, progressCallback) -> None:
        """
        Read an emo file to determine the extent
        :param : emoFile : file to analyze
        :param : progressCallback : monitor the progress of the activity (0.0 = started ... 1.0 = finished)
        Raised exceptions :
            - OSError when file is not a suitable emo file
        """
        for column in emoFile.minmax.keys():
            emoFile.minmax[column] = [numpy.inf, -numpy.inf]

        # browse file by chunk of 1 000 000 lines
        fileSize = os.path.getsize(emoFile.file_path)
        with open(emoFile.file_path, mode="rt", encoding="utf8") as openedFile:
            try:
                # Read the 2 first lines
                textFileReader = self.parse(emoFile, lonlatOnly=True)
                for chunk in textFileReader:
                    emoFile.lineCount += chunk.shape[0]
                    npChunk = chunk.to_numpy(copy=False)
                    self.computeMinMax(npChunk, emoFile)

                    del npChunk
                    del chunk

                # Estimate progression
                progressCallback.worked(1)

            except ValueError as err:
                raise OSError(f"Bad emo file. {err}") from err

        emoFile.west = emoFile.min(EMO.COL_LONGITUDE) - 0.5 * emoFile.spatialResolution
        emoFile.east = emoFile.max(EMO.COL_LONGITUDE) + 0.5 * emoFile.spatialResolution
        emoFile.south = emoFile.min(EMO.COL_LATITUDE) - 0.5 * emoFile.spatialResolution
        emoFile.north = emoFile.max(EMO.COL_LATITUDE) + 0.5 * emoFile.spatialResolution

    def evaluateSpatialResolution(self, emoFile: EmoFile) -> None:
        """
        Evaluate the spatial resolution for the given EmoFile
        Raised exception : OSError when file is not a suitable emo file
        """
        with open(emoFile.file_path, mode="rt", encoding="utf8") as openedFile:
            try:
                # Read the 2 first lines
                # textFileReader = self.parse(emoFile, chunksize=2, lonlatOnly=True)
                # lines = textFileReader.get_chunk(2)
                # if lines.shape[0] < 2:
                #     raise OSError("Bad emo file : Not enough row")
                #
                # deltaLon = abs(lines[EMO.COL_LONGITUDE][0] - lines[EMO.COL_LONGITUDE][1])
                # deltaLat = abs(lines[EMO.COL_LATITUDE][0] - lines[EMO.COL_LATITUDE][1])
                # del lines
                #
                # rawcellSize = max(deltaLat, deltaLon)
                # reductionFactor = round(1 / (rawcellSize * 60.0))
                # read a big amount of lines
                textFileReader = self.parse(emoFile, chunksize=10 ** 6, lonlatOnly=True)
                lines = textFileReader.get_chunk(10 ** 6)
                if lines.shape[0] < 2:
                    raise ProcessingError("Bad emo file : Not enough row")

                deltaLon = abs(lines[EMO.COL_LONGITUDE][0:-1].to_numpy() - lines[EMO.COL_LONGITUDE][1:].to_numpy())
                deltaLon[deltaLon == 0] = numpy.nan

                deltaLat = abs(lines[EMO.COL_LATITUDE][0:-1].to_numpy() - lines[EMO.COL_LATITUDE][1:].to_numpy())
                deltaLat[deltaLat == 0] = numpy.nan
                rawcellSize = min(numpy.nanmin(deltaLon), numpy.nanmin(deltaLat))
                reductionFactor = round(1 / (rawcellSize * 60.0))
                if reductionFactor == 0:
                    raise ProcessingError("Cannot estimate spatial resolution")

                emoFile.spatialResolution = 1.0 / (60.0 * reductionFactor)

            except ValueError as err:
                raise OSError(f"Bad emo file. {err}") from err

    def parse(self, emoFile: EmoFile, chunksize: int = 10 ** 6, lonlatOnly=False) -> None:
        """
        Parse the emo file with Pandas and return a TextFileReader
        """

        if lonlatOnly:
            names = [EMO.COL_LONGITUDE, EMO.COL_LATITUDE]
            usecols = names
            dtype = {key: value for (key, value) in EmoFile.ColumnDescriptions.items() if key in usecols}
            converters = None
        else:
            names = EmoFile.ColumnNames
            usecols = names
            dtype = {
                key: value
                for (key, value) in EmoFile.ColumnDescriptions.items()
                if not key in [EMO.COL_CDIID, EMO.COL_DTM_SOURCE]
            }
            converters = {EMO.COL_CDIID: emoFile.cdiId, EMO.COL_DTM_SOURCE: emoFile.sourceId}

        return pd.read_csv(
            emoFile.file_path,
            chunksize=chunksize,
            delimiter=";",
            header=None,
            names=names,
            dtype=dtype,
            usecols=usecols,
            converters=converters,
        )
