import sys

from pyat.utils import pyat_logger, process


def test_process_with_exception():

    logger = pyat_logger.logging.getLogger(__name__)
    value = process.run_and_log([r"C:\Windows\System32\whoami.exe", "-e"], logger)
    assert value != 0  # 0 means success


def test_process_without_exception():
    logger = pyat_logger.logging.getLogger(__name__)
    if sys.platform == "win32":
        value = process.run_and_log([r"C:\Windows\System32\whoami.exe"], logger)
    else:
        value = process.run_and_log([r"/usr/bin/ls"], logger)
    assert value == 0  # 0 means success
