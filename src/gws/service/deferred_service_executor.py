#! /usr/bin/env python3
# coding: utf-8

import logging
import os
import sys
from typing import Dict

import pygws.service.deferred_service_executor as pygws_deferred_service_executor
import pygws.service.execution_context as exec_ctx

import pyat.utils.application_utils as app_util
from pyat.utils.logger import logger

## Set nb of threads used by numpy to avoid warning in log
os.environ["NUMEXPR_MAX_THREADS"] = "16"


def _process_json_request(arguments: Dict) -> None:
    try:
        if "configuration_file" not in arguments:
            raise ValueError("No argument configuration_file found in arguments")

        json_configuration_file = arguments["configuration_file"]
        del arguments["configuration_file"]

        monitor = exec_ctx.get_root_progress_monitor()
        service = app_util._extract_function(json_configuration_file)

        # Running the service by
        logger.debug("Running service '%s'", service)
        result = app_util.run(arguments, monitor, logger, service)
        # Sending the resulting output files
        if result is not None and "outfile" in result:
            rsocket_msg_emitter = exec_ctx.get_rsocket_msg_emitter()
            if rsocket_msg_emitter is not None:
                rsocket_msg_emitter.emit_files(result["outfile"])

        logger.debug("End of the service '%s'", service)
    except Exception as ex:
        logger.error("Error during service execution: %s", str(ex))
        raise


if __name__ == "__main__":
    logging.basicConfig(filename="logs/pooled_service.log", level=logging.INFO, force=True)

    # Expecting the RSocket port.
    if len(sys.argv) == 2:
        try:
            rsocket_port = int(sys.argv[1])
            logger.info("Preparing a service execution on port %d", rsocket_port)

            pygws_deferred_service_executor.start_and_wait_signal(
                socket_port=rsocket_port, service=_process_json_request
            )

        except ValueError as e:
            raise ValueError("Expecting an integer in argument for the RSocket port") from e

    else:
        raise ValueError("Bad number of argument. Expecting the RSocket port")
