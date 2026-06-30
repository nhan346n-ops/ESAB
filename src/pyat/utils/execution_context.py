import sys

gettrace = sys.gettrace()

# For debugging
debug_status = bool(gettrace)

def is_debug() -> bool:
    """Return true if the process is started in debug mode"""
    return debug_status



# pylint: disable=import-outside-toplevel
def is_running_from_ipython():
    # pylint:disable = import-error
    from IPython import get_ipython
    return get_ipython() is not None
