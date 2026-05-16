#! /usr/bin/env python3
# coding: utf-8

import logging
import sys

import flatbuffers
import numpy as np
import pygws.data_model.flatbuffers.protocol_xsf_generated as proto_xsf
from pygws.service import service_executor
from rsocket.helpers import utf8_decode

from pyat.utils.application_utils import load_json_file
from pyat.sonarscope.model.sonar_factories import ModeComputerFactory
from pyat.sonarscope.model.sounder_lib import SounderType
from pyat.sonarscope.model.sounder_mode.sounder_modes import KeyMode
from pyat.utils.exceptions.exception_list import BadParameter
from pyat.utils.logger import logger
from pyat.xsf import xsf_driver


class XsfAcquisitionModeInterpreter:
    """
    Return the acquisition modes for the file specified in the payload (expecting a XsfAcquisitionModeRequest)
    """

    def get_rsocket_route(self) -> str:
        """Route of payload"""
        return "get_xsf_acquisition_mode"

    async def process_payload(self, payload: bytearray) -> bytearray | None:
        """Managing the XsfSyntheticModeRequest payload"""
        request = proto_xsf.XsfAcquisitionModeRequest.GetRootAs(payload)
        xsf_file_path = utf8_decode(request.FilePath())
        total_keys = {}  # dictionary containing KeyMode and their id
        mode_values = None
        with xsf_driver.open_xsf(file_path=xsf_file_path) as xsf_file:
            sounder_type = SounderType.from_dataset(xsf_dataset=xsf_file)
            mode_computer = ModeComputerFactory.create_mode_computer(sounder_type)
            total_keys, mode_values = mode_computer.compute_xsf(xsf=xsf_file, global_keys=total_keys)

        # order keys by id
        keys_array = np.ndarray(shape=(len(total_keys)), dtype=KeyMode)
        for key, index in total_keys.items():
            keys_array[index] = key

        builder = flatbuffers.Builder()
        mode_indices = builder.CreateNumpyVector(mode_values.astype(np.int32).ravel())

        # flatbuffer vectors are filled in reverse order
        mode_s = [builder.CreateString(str(mode)) for mode in keys_array[::-1]]
        proto_xsf.XsfAcquisitionModeStartModeNameVector(builder, len(keys_array))
        for mode in mode_s:
            builder.PrependUOffsetTRelative(mode)
        mode_names = builder.EndVector(len(keys_array))

        proto_xsf.XsfAcquisitionModeStart(builder)
        proto_xsf.XsfAcquisitionModeAddModeCount(builder, len(keys_array))
        proto_xsf.XsfAcquisitionModeAddModeName(builder, mode_names)
        proto_xsf.XsfAcquisitionModeAddPingCount(builder, len(mode_values))
        proto_xsf.XsfAcquisitionModeAddModeIndex(builder, mode_indices)
        builder.Finish(proto_xsf.XsfAcquisitionModeEnd(builder))
        logger.info(
            f"XsfAcquisitionMode created with {len(keys_array)} modes and {len(mode_values)} pings and {len(np.unique(mode_values))} values"
        )
        return builder.Output()


if __name__ == "__main__":
    logging.basicConfig(filename="logs/xsf_driver.log", level=logging.DEBUG, force=True)
    logger.info("Starting XSF driver")
    if len(sys.argv) != 2:
        raise BadParameter("Bad number of argument for globe - python process")
    # pylint: disable=unbalanced-tuple-unpacking
    arguments = load_json_file(sys.argv[1])
    if "rsocket_port" not in arguments:
        raise BadParameter("Socket port not in configuration file. Execution aborted")

    logger.info("TCP socket port is %d", arguments["rsocket_port"])
    payload_interpreters = [
        XsfAcquisitionModeInterpreter(),
    ]
    service_executor.execute(socket_port=arguments["rsocket_port"], payload_interpreters=payload_interpreters)
