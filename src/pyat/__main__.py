import sys
from importlib.metadata import version

from pyat.utils.application_utils import get_json_configuration_file_from_argument_file, launch_application


def __launch__(json_arguments_path: str):
    # Extract configuration file from arguments file
    json_configuration_file = get_json_configuration_file_from_argument_file(json_arguments_path)
    return launch_application(json_configuration_file)


if __name__ == "__main__":
    # Default process : the function to run will be determined from the JSON configuration file.
    if len(sys.argv) == 2:
        # pylint: disable=unbalanced-tuple-unpacking
        __launch__(sys.argv[1])
    else:
        # Display pyat version and usage message if no .json parameter file is provided
        pyat_version = version("pyat")
        print(f"pyat version: {pyat_version}")
        print("A parameters file in .json format may be passed as an argument.")
        print("Usage: python -m pyat <parameters_file.json>")
