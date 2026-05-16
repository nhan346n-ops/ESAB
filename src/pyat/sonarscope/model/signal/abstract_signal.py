from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Dict, List, Set, Hashable, Callable, Mapping, Any

import netCDF4 as nc
import numpy as np
import xarray as xr

from pyat.utils import pyat_logger
from pyat.xsf.xsf_driver import XsfDriver


class PDimension:
    def __init__(self, name: str, dimension: nc.Dimension):
        self.name = name
        self.dimension = dimension


class PVariable:
    """A xarray variable like class, temporary usage and internal use only"""

    def __init__(self, data, dimensions: [str], attributes: Dict[str, object] = None, coordinates: [str] = None):
        self.data = data
        self.dimensions = dimensions
        self.attributes = attributes
        self.coordinates = coordinates


class BaseSignal(ABC):
    def __init__(self, xsf_dataset: XsfDriver):
        self.xsf_dataset = xsf_dataset  # the netcdf dataset
        self.xr_dataset: xr.Dataset = None  # final xarray dataset holding all queried variables
        self.logger = pyat_logger.logging.getLogger(__name__)

    @abstractmethod
    def get_mandatory_variable(self) -> List[str]:
        """return the list of key of mandatory variables that will always be added"""

    def _compute_variable_list(self, keys: Iterable[str]) -> Tuple[List[str], Set[Hashable]]:
        """Compute the list of variable to load given the list of variables to load, and knowing the already loaded variables"""

        known_variables = []
        if self.xr_dataset is not None:
            known_variables = self.variables.keys()

        keys = list(keys) + self.get_mandatory_variable()

        keys = set(keys)  # remove double entries

        keys = [k for k in keys if k not in known_variables]
        return keys, set(known_variables)

    @abstractmethod
    def read_new_variables(
        self, variable_to_load: List[str], reductor_function: Callable, ignore_unknown_variables: bool
    ):
        """really perform the reading of new variables, interpolations and so on"""

    @abstractmethod
    def read_coordinates(self) -> Mapping[Any, Any]:
        """Return a mapping with coordinates variables"""

    def read(self, keys: Iterable[str], reductor_function=np.nanmean, ignore_unknown_variables=False):
        """
        Read the new set of queried variables, check for their associated dimensions and coordinates and append everything to the existing dataset
        Note that a new xr_dataset will be created

        Args:
            ignore_unknown_variables: if False raise exception if a variable is not found in file or cannot be computed,
             otherwise a variable with default values is created
            keys: the list of variable to load
            reductor_function : a numpy reduction function to apply to data if dimension is higher than one.
                Function can be np.nanmean, np.nanmin, etc

        Returns:

        """

        # Compute the real list of variable to
        # keep only key values from Iterable[Key]
        variable_to_load, known_variables = self._compute_variable_list(keys)

        # load or retrieve dimension values to prevent to read it multiple times

        new_variables = self.read_new_variables(
            variable_to_load=variable_to_load,
            reductor_function=reductor_function,
            ignore_unknown_variables=ignore_unknown_variables,
        )

        # Merge with existing dataset
        coordinates = None
        variables = {}

        # If not already exists, create an empty dataset
        if self.xr_dataset is None:
            self.xr_dataset = xr.Dataset()
            # read coordinates only in case of new dataset
            coordinates = self.read_coordinates()

        # Retrieve existing variables (coordinates included)
        for name, value in self.xr_dataset.variables.items():
            variables[name] = value

        for name, value in new_variables.items():
            variables[name] = xr.Variable(data=value.data, dims=value.dimensions, attrs=value.attributes)

        # now we can create DataArrays
        dataset = xr.Dataset(data_vars=variables, attrs=None, coords=coordinates)

        # decode time units if needed
        dataset = xr.decode_cf(dataset, decode_timedelta=False)

        self.xr_dataset = dataset

    @property
    def variables(self):
        """return the name and values of all known variables"""
        return {} if self.xr_dataset is None else self.xr_dataset.variables

    @property
    def dims(self):
        """return the name and values of all known dimension"""
        return {} if self.xr_dataset is None else self.xr_dataset.dims

    @property
    def coordinates(self):
        """return the name and values of all known coordinates"""
        return {} if self.xr_dataset is None else self.xr_dataset.coords

    def get_dataset(self) -> xr.Dataset:
        """Return the xarray nc dataset"""
        return self.xr_dataset

    def __contains__(self, key: object) -> bool:
        """The 'in' operator will return true or false depending on whether
        'key' is an array in the dataset or not.
        """
        return False if self.xr_dataset is None else key in self.xr_dataset
