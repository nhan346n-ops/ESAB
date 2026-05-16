import os
import tempfile

session_temp_directory = tempfile.mkdtemp(suffix="_pyat")
print(f"Using {tempfile} as output directory")


def get_input_directory():
    """retrieve a directory where input test data are stored"""
    return os.path.abspath("./data/raw")


def get_output_directory():
    """retrieve a directory where output test data should be stored"""
    return session_temp_directory


def get_test_directory():
    """retrieve a temporary directory where to put generated data"""
    return os.path.abspath("./data")
