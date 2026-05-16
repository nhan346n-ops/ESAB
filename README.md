# __Py__thon __A__coustic __T__oolBox

The python acoustic toolbox gather a set of the French Oceanographic Fleet processing algorithms.
The set of algorithm focus mainly but not exclusively on data processing of single and multibeam echo sounders

## Project Organization

Project organization is inspired by
https://drivendata.github.io/cookiecutter-data-science/#cookiecutter-data-science

```
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs                <- Documentation files (antora based documentation).
|
├── gws                 <- GWS (Globe Web Services) specific files.
|
├── licenses            <- This directory holds license and credit information for works astropy is derived from or distributes, and/or datasets.
│
├── src                 <- Source code for use in this project.
│   ├── pyat        <- PyAT main package
│   ├── gws         <- GWS (Globe Web Services) module (used by GWS serer to build services from pyat modules)
│
│── tests               <- Unit tests
|
├── requirements   <- The requirements files for reproducing the analysis environment,
|                     use create_anaconda_environment.py script to create them
│
├── pyproject.toml      <- The build system configuration file.
├── LICENSE             <- The PyAT license file (LGPL v3).
├── README.md           <- The top-level README for developers using this project.
```

Note : PyAT project use conda environment to manage dependencies.

## Install and use PyAT

### How to install

PyAT is installable as a pip package in a conda environment. To create a conda environment with PyAT, use a conda requirement file like
`gws/requirements/requirements_gws_runtime.yml`. This file contains a reference to PyAT package, and all required dependencies.

To create a conda environment with this file, use the following command :
`conda env create -f gws/requirements/requirements_gws_runtime.yml -n env_pyat_runtime`

Once the environment is created and activated, check if PyAT is available with the following command : `python -m pyat`. Should display PyAT version
like :

```
pyat version: 0.1.45
```

PyAT and its submodules are now available in the conda environment, and can be imported in python code like this : `import pyat` or `from pyat.modules import some_module`.

### How to launch a process with a JSON file as argument

PyAT module cas be called with a JSON file as argument, containing the process to execute and its arguments.
For example, launch PyAt with the following command : `python -m pyat path/to/file.json`

With `path/to/file.json` content :

```
{
  "i_paths" : [ "F:\\01_format_dtm\\1_16_8200111182_SEADOME.dtm.nc", "F:\\01_format_dtm\\1_16_9000011182_MESEAI-cdi.dtm.nc" ],
  "o_path" : "C:\\TMP\\merged_simple.dtm.nc",
  "layers" : {
    "elevation" : true,
    "elevation_min" : true,
    "elevation_max" : true,
    "cdi_index" : true,
    "value_count" : true,
    "stdev" : true
  },
  "coord" : {
    "north" : 44.08333348197242,
    "south" : 41.06666666666667,
    "west" : 4.05,
    "east" : 8.883333496749401
  },
  "configuration_file" : "C:\\dev\\pyat\\src\\gws\\conf\\dtm\\merge\\merge_simple.json"
}
```
Note : `configuration_file` is a JSON file containing the configuration of the process to execute. This file is used by PyAT to know which module and function to call, and how to map JSON arguments to function arguments. Configuration files are stored in `src/gws/conf` directory.

## Developer guide

### Install conda environment

To create conda environment, use the following command :
`python ./requirements/create_anaconda_environments.py -target pyat_dev`

Note : first time, this command will install `conda_mgr` package, which is required to create conda environments.

### Install project in editable mode

To install project in editable mode, use the following command from project root directory :
`python -m pip install -e . --no-cache-dir`

### Run a process in debug mode

To run a process, use the following command from project root directory :
`python -m pyat PATH_TO_FILE_WITH_ARGS.json`

Launch this command in PyCharm Run/Debug configuration to be able to set breakpoints in the code.

### How to use local project as dependency

To use local version of dependency projects (like PyTechsas, PyNvi...); use this command :

`python -m pip install -e path/to/SomeLocalProject --no-cache-dir`

See : https://pip.pypa.io/en/stable/topics/local-project-installs/

Note : subprojects can be open in the same PyCharm window than PyAt : `File > Open...` ; choose the local project and select `Attach`.

### Execute unit tests

To execute unit tests, use the following command from project root directory : `pytest`
To execute a specific test file : `pytest path/to/specific_test_file.py`

### Run pylint

To run pylint, use the following command from project root directory : `pylint pyat`

### Deploy new release

Pyat project can be pip installable. Use the following commands :

```
python -m build
python -m twine upload --repository gitlab-pyat dist/*
```

Note : id of server is defined in ~/.pypirc.

