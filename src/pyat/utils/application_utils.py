import inspect
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from importlib import import_module
from typing import Callable, Dict

from dateutil import parser
from pygws.service.progress_monitor import DefaultMonitor

import pyat.utils.argument_utils as arg_util
from pyat.utils.exceptions.exception_list import BadParameter


def launch_application(json_conf_file_path: str, callable_process: Callable[..., Dict | None] = None) -> None:
    """
    Launch an application wrapping a callable process.
    :param json_conf_file_path: the path of the application (*_app.py)
    :param callable_process: a class previously imported. eg:
        'from pyat.pyat.dtm.peak_detector import PeakFinder'
        -> pass PeakFinder as a parameter

    When called through a command line:
    Parameters of the application are read in the json file when present as the first argument of the command line.
    If this json file is missing, all the command line is considered as the parameters of the application

    Then the application is instantiated with all the parameters and finally called
    """
    # if callable not provided, find it from configuration file
    if not callable_process:
        callable_process = _extract_function(json_conf_file_path)

    # set the logger and the monitor depending on what called the application
    monitor = DefaultMonitor
    logger = _init_logger(callable_process)

    if callable_process is not None:
        logger.debug(f"Launching application '{callable_process.__name__}'")

    if len(sys.argv) > 1:
        # Get arguments
        arguments: Dict = {}
        system_args = sys.argv[1:]
        if system_args[0].endswith("json"):
            # a json parameter file has been passed in system arguments
            arguments = load_json_file(system_args[0])
        else:
            # get parameters from command line. First one is the name of the processed file
            arg_parser = arg_util.create_argv_parser(callable_process.__name__, json_conf_file_path)
            arguments = vars(arg_parser.parse_args(system_args))

        # Globe provides the configuration file among the arguments. Not needed here
        if "configuration_file" in arguments:
            del arguments["configuration_file"]

        run(arguments, monitor, logger, callable_process)

    else:
        logger.error(f"Error while launching '{callable_process.__name__}' : arguments missing")
        logger.error("At the very least, provide a json file as the expected process arguments.")


def run(arguments, monitor, logger, callable_process) -> Dict:
    """
    :param arguments: dictionary containing parameters (previously parsed from json)
    :param monitor: from init_logger
    :param logger: from init_logger
    :param callable_process: a class previously imported cf. launch_application
    :return: a dictionary containing the report of the process
    """
    if arguments:
        try:
            monitor.begin_task(str(callable_process), 100)

            # add monitor if possible
            if "monitor" in inspect.signature(callable_process).parameters:
                arguments["monitor"] = monitor

            # setup temp directory if possible
            if "temp_dir" in arguments:
                tempfile.tempdir = arguments["temp_dir"]

            is_function: bool = inspect.isfunction(callable_process)

            # if necessary, cast arguments to match with callable parameters types
            callable_params = inspect.signature(
                callable_process if is_function else callable_process.__call__
            ).parameters
            for arg in arguments:
                if arg in callable_params and callable_params[arg].annotation is datetime:
                    arguments[arg] = parser.parse(arguments[arg])  # cast "datetime" arguments

            # call function or run class
            report = callable_process(**arguments) if is_function else callable_process(**arguments)()

            # Reporting
            monitor.done()
            return _format_result_of_process(arguments, report)
        except ValueError as e:
            logger.exception(str(e))
            return _format_result_of_process(None, {"error": str(e)})
        except LookupError as e:
            # Error occurs when app try to access to an empty context variable (pygws.service.execution_context)
            logger.exception("This process need a TCP socket. Activates this option and try again.")
            return _format_result_of_process(None, {"error": str(e)})
        except Exception as e:
            logger.exception(f"An exception was thrown : {str(e)}")
            return _format_result_of_process(None, {"error": str(e)})
    else:
        error = """ Useless process without input(s) and parameter(s). Stop the program.
        Please enter input with the option -i I_PATHS [I_PATHS ...], --i_paths I_PATHS [I_PATHS ...].
        """
        logger.error(error)
        return {"error": error}


def _init_logger(callable_process: Callable) -> logging.Logger:
    """
    :param callable_process: a class previously imported cf. launch_application
    :return logger: get previously by init_logger
    """

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s - %(levelname)s - %(name)s : %(message)s",
        force=True,
    )

    return logging.getLogger(callable_process.__name__)


def get_json_configuration_file(application_file_path: str) -> str:
    """
    Compute the json configuration file of the an application.
    :param application_file_path -- the path of the application (*_app.py).
    :return [str] -- the json configuration file
    """
    bn = os.path.basename(application_file_path)
    dir_path = os.path.dirname(application_file_path)
    return os.path.join(dir_path, "conf", bn[: bn.rfind("_")] + ".json")


def get_json_configuration_file_from_argument_file(json_arguments_path: str) -> str:
    """
    Extract the json configuration file of the an application from the json argument file.
    :param application_file_path -- the path of the json argument file.
    :return [str] -- the json configuration file
    """
    arguments = load_json_file(json_arguments_path)
    if "configuration_file" in arguments:
        return arguments["configuration_file"]
    raise BadParameter(f"No argument configuration_file found in {json_arguments_path} - python process")


def load_json_file(file_path: str):
    """
    Loads json file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _extract_function(json_conf_file_path: str):
    """
    Extracts from the conguration file, the property function
    It represents the python entry point to be invoked to start the service
    """
    conf = load_json_file(json_conf_file_path)
    module_path, function = conf["function"].rsplit(".", 1)
    mod = import_module(module_path)
    return getattr(mod, function)


def _format_result_of_process(arguments, report) -> Dict:
    """
    If no report is provided, it will be created from the output files
    Otherwise, it will be returned as is.
    : param arguments: the arguments of the process
    : param report: dictionary of results generated by the process
    """
    if report is None:
        o_paths = "o_paths" if "o_paths" in arguments else "o_path"
        if o_paths in arguments:
            out_file_paths = arguments[o_paths]
            if isinstance(out_file_paths, str):
                out_file_paths = [out_file_paths]
            report = {"outfile": [file_path for file_path in out_file_paths if os.path.exists(file_path)]}

    return report
