#! /usr/bin/env python3
# coding: utf-8


import logging
import sys
from typing import Dict

import flatbuffers
import numpy as np
import pygws.data_model.flatbuffers.protocol_adcp_generated as proto_adcp
from pygws.service import service_executor
from rsocket.helpers import utf8_decode

import gws.drivers.adcp.adcp_model as model
from pyat.utils.application_utils import load_json_file
from pyat.utils.exceptions.exception_list import BadParameter
from pyat.utils.logger import logger
from gws.drivers.adcp.adcp_parser import (parse_adcp_file)

# Maximum number of vectors to be returned
MAX_VECTOR_COUNT = 1000000

# Cache of ADCP file data. File were parsed with Dolfyn and processed with a XxxParser
_ADCP_FILE_CACHE: Dict[str, model.AdcpData] = {}


def _open_adcp_file(adcp_file_path: str) -> model.AdcpData:
    """
    Open an Adcp file.
    If present in the cache, return the model.AdcpData directly.
    Otherwise, parse the file with Dolfyn, complete the cache and return the model.AdcpData
    """

    if adcp_file_path not in _ADCP_FILE_CACHE:
        try:
            logger.info(f"Parsing the ADCP file {adcp_file_path}")
            adcp_data = parse_adcp_file(adcp_file_path)
            _ADCP_FILE_CACHE[adcp_file_path] = adcp_data
        except Exception as e:
            raise IOError(f"Not an ADCP file ({adcp_file_path})") from e
    else:
        logger.info(f"using cache data for file {adcp_file_path}")

    return _ADCP_FILE_CACHE[adcp_file_path]


def _create_adcp_file_info(adcp_data: model.AdcpData) -> bytearray:
    """
    Build an AdcpFileInfo from the model.AdcpData
    """
    builder = flatbuffers.Builder()
    file_path = builder.CreateString(adcp_data.file_path)
    proto_adcp.AdcpFileInfoStart(builder)
    proto_adcp.AdcpFileInfoAddFilePath(builder, file_path)

    current_data = adcp_data.current_data
    proto_adcp.AdcpFileInfoAddVectorCount(builder, current_data.index.size)
    proto_adcp.AdcpFileInfoAddDateIndexCount(builder, np.int64(adcp_data.time.shape[0]))
    proto_adcp.AdcpFileInfoAddDatetimeMin(builder, np.int64(adcp_data.time[0].astype(np.int64) / 1e6))
    proto_adcp.AdcpFileInfoAddDatetimeMax(builder, np.int64(adcp_data.time[-1].astype(np.int64) / 1e6))
    proto_adcp.AdcpFileInfoAddLatitudeMin(builder, adcp_data.latitude.min())
    proto_adcp.AdcpFileInfoAddLatitudeMax(builder, adcp_data.latitude.max())
    proto_adcp.AdcpFileInfoAddLongitudeMin(builder, adcp_data.longitude.min())
    proto_adcp.AdcpFileInfoAddLongitudeMax(builder, adcp_data.longitude.max())
    range_values = current_data[model.RANGE]
    proto_adcp.AdcpFileInfoAddRangeMin(builder, range_values.min())
    proto_adcp.AdcpFileInfoAddRangeMax(builder, range_values.max())
    adcp_file_info = proto_adcp.AdcpFileInfoEnd(builder)
    builder.Finish(adcp_file_info)
    logger.info(f"AdcpFileInfo created")

    return builder.Output()


def _apply_filtering(adcpData: model.AdcpData, request: proto_adcp.AdcpCurrentRequest) -> model.AdcpData:
    """
    Generate a new model.AdcpData by filtering current of the model.AdcpData source
    """
    # Filter the current data based on the range of elevation
    range_min = request.RangeMin()
    range_max = request.RangeMax()
    if range_min is not None or range_max is not None:
        logger.info(f"Filtering vectors, range between {range_min} and {range_max}")
        adcpData = adcpData.apply_range_filter(range_min, range_max)

    # Sample the current data based on the elevation sampling
    range_sampling = request.RangeSampling()
    if range_sampling is not None and range_sampling > 0:
        logger.info(f"Sampling vectors on elevation. Item limit is {range_sampling} ")
        adcpData = adcpData.apply_range_sampling(range_sampling)

    # Filter the current data based on range of time
    date_index_min = request.DateIndexMin()
    date_index_max = request.DateIndexMax()
    if date_index_min is not None or date_index_max is not None:
        logger.info(f"Filtering vectors, date index between {date_index_min} and {date_index_max}")
        adcpData = adcpData.apply_time_filter(date_index_min, date_index_max)

    # Sample the current data based on the time sampling
    time_sampling = request.DateIndexSampling()
    if time_sampling is not None and time_sampling > 0:
        logger.info(f"Sampling vectors on time. Item limit is  {time_sampling} ")
        adcpData = adcpData.apply_time_sampling(time_sampling)

    # Limit the number of vectors to be returned
    max_count = request.MaxVectorCount()
    if max_count is None or max_count > MAX_VECTOR_COUNT:
        max_count = MAX_VECTOR_COUNT
    logger.info(f"Limiting vector count to {max_count}")
    adcpData = adcpData.reduce(max_count)

    return adcpData


class OpenAdcpFileInterpreter:
    """
    Handling an open request
    """

    def get_rsocket_route(self) -> str:
        """Route of payload"""
        return "open_adcp_file"

    async def process_payload(self, payload: bytearray) -> bytearray | None:
        """Process payload"""
        adcp_file_path = utf8_decode(payload)
        logger.info("Opening ADCP file %s", adcp_file_path)
        adcp_data = _open_adcp_file(adcp_file_path)
        return _create_adcp_file_info(adcp_data)


class CloseAdcpFileInterpreter:
    """
    Handling a close request
    """

    def get_rsocket_route(self) -> str:
        """Route of payload"""
        return "close_adcp_file"

    async def process_payload(self, payload: bytearray) -> bytearray | None:
        """Process payload"""
        adcp_file_path = utf8_decode(payload)
        if adcp_file_path in _ADCP_FILE_CACHE:
            logger.info(f"closing ADCP file {adcp_file_path}")
            del _ADCP_FILE_CACHE[adcp_file_path]
        return None


class AdcpCurrentInterpreter:
    """
    Return the current data for the file specified in the payload (expecting a AdcpCurrentRequest)
    """

    def get_rsocket_route(self) -> str:
        """Route of payload"""
        return "get_adcp_current"

    async def process_payload(self, payload: bytearray) -> bytearray | None:
        """Managing the AdcpCurrentRequest payload"""
        request = proto_adcp.AdcpCurrentRequest.GetRootAs(payload)
        adcp_file_path = utf8_decode(request.FilePath())
        adcp_data = _open_adcp_file(adcp_file_path)

        adcp_data = _apply_filtering(adcp_data, request)

        builder = flatbuffers.Builder()
        file_path = builder.CreateString(adcp_file_path)
        time_values = builder.CreateNumpyVector(adcp_data.time.astype(np.int64))
        latitude_values = builder.CreateNumpyVector(adcp_data.latitude)
        longitude_values = builder.CreateNumpyVector(adcp_data.longitude)

        current_data = adcp_data.current_data
        range_values = builder.CreateNumpyVector(current_data[model.RANGE].to_numpy())
        time_index_values = builder.CreateNumpyVector(current_data[model.TIME_INDEX].to_numpy())
        eastward_velocity_values = builder.CreateNumpyVector(current_data[model.EASTWARD_VELOCITY].to_numpy())
        northward_velocity_values = builder.CreateNumpyVector(current_data[model.NORTHWARD_VELOCITY].to_numpy())
        downward_velocity_values = builder.CreateNumpyVector(current_data[model.DOWNWARD_VELOCITY].to_numpy())

        proto_adcp.AdcpCurrentStart(builder)
        proto_adcp.AdcpCurrentAddFilePath(builder, file_path)
        proto_adcp.AdcpCurrentAddVectorCount(builder, current_data.index.size)
        proto_adcp.AdcpCurrentAddTime(builder, time_values)
        proto_adcp.AdcpCurrentAddLatitude(builder, latitude_values)
        proto_adcp.AdcpCurrentAddLongitude(builder, longitude_values)
        proto_adcp.AdcpCurrentAddRange(builder, range_values)
        proto_adcp.AdcpCurrentAddTimeIndex(builder, time_index_values)
        proto_adcp.AdcpCurrentAddEastwardVelocity(builder, eastward_velocity_values)
        proto_adcp.AdcpCurrentAddNorthwardVelocity(builder, northward_velocity_values)
        proto_adcp.AdcpCurrentAddDownwardVelocity(builder, downward_velocity_values)
        builder.Finish(proto_adcp.AdcpCurrentEnd(builder))
        logger.info(f"AdcpCurrent created with {current_data.index.size} vectors")

        return builder.Output()


if __name__ == "__main__":
    logging.basicConfig(filename="logs/adcp_driver.log", level=logging.INFO, force=True)
    logger.info("Starting ADCP driver")
    # Expecting the JSON configuration file.
    if len(sys.argv) == 2:
        # pylint: disable=unbalanced-tuple-unpacking
        arguments = load_json_file(sys.argv[1])
        if "rsocket_port" not in arguments:
            raise BadParameter("Socket port not in configuration file. Execution aborted")

        logger.info("TCP socket port is %d", arguments["rsocket_port"])
        payload_interpreters = [OpenAdcpFileInterpreter(), AdcpCurrentInterpreter(), CloseAdcpFileInterpreter()]
        service_executor.execute(socket_port=arguments["rsocket_port"], payload_interpreters=payload_interpreters)
    else:
        raise BadParameter("Bad number of argument for globe - python process")
