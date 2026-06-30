import logging


class Error(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message):
        super().__init__()
        self.message = message
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.error(f"Exception occured :{self.message}")


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """

class UnexpectedError(Exception):
    """Implementation error"""


class BadParameter(Error):
    """Exception raised for errors during processing.

    Attributes:
        message -- explanation of the error
    """


class UnknownLayer(BadParameter):
    """Exception when a unknown key is queries.

    Attributes:
        message -- explanation of the error
    """

class BadAssumption(Error):
    """
    Exception raised for assumption errors during processing.

    Attributes:
        message -- explanation of the error
    """


class ProcessingError(Error):
    """Exception raised for errors during processing.

    Attributes:
        message -- explanation of the error
    """


class UnsupportedProjectionError(Error):
    """Exception raised when Projection does not meet what is expected.

    Attributes:
        message -- explanation of the error
    """
