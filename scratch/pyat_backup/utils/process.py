from typing import Optional

import logging
import subprocess
from subprocess import Popen, PIPE, STDOUT, DEVNULL


def run(command_line_args: str, logger: Optional[logging.Logger] = None) -> int:
    """Run the command line and return result, is a logger is specified use it"""
    try:
        if logger is not None:
            ret = run_and_log(command_line_args, logger)
        else:
            ret = subprocess.run(command_line_args, stdout=DEVNULL, check=False)
            ret = ret.returncode
        return ret
    except BaseException as e:
        logging.error(e, exc_info=True)
    return -1


def run_and_log(command_line_args: str, logger) -> int:
    """Basic utility to run a command line and write output to the logger.
    Some restriction might apply, like the possibility to interleave errors and warning, and all log are stored in memory
    """
    if logger is None:
        raise NotImplementedError("Call run without logger, use run command instead ")
    try:
        with Popen(command_line_args, stdout=PIPE, stderr=STDOUT, text=True) as process:
            with process.stdout:
                for line in process.stdout:
                    line = line.strip()
                    if line != '':
                        logger.info(line.strip())
            exitcode = process.wait()  # 0 means success
            return exitcode
    except BaseException as e:
        logging.error(e, exc_info=True)
    return -1
