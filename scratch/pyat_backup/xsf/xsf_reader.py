import numpy as np
import sonar_netcdf.sonar_groups as constants

from pyat.utils.nc_encoding import open_nc_file


class XSFReader:
    """
    Class used for reading a xsf file
    """

    def __retrieve_known_variables__(self):
        self.beamGroup = self.dataset[constants.BeamGroup1Grp.get_group_path("Beam_group1")]
        ping_count = self.beamGroup.dimensions[constants.BeamGroup1Grp.PING_TIME_DIM_NAME]
        self.ping_count = ping_count.size

        beam_count = self.beamGroup.dimensions[constants.BeamGroup1Grp.BEAM_DIM_NAME]
        self.beam_count = beam_count.size

        # get WC datasets
        # pylint: disable=W0702
        try:
            vendorGroup = self.dataset[constants.BeamGroup1VendorSpecificGrp.get_group_path("Beam_group1")]
            vList = vendorGroup.variables
            tvg_name = constants.BeamGroup1VendorSpecificGrp.get_group_path() + "tvg_offset"
            if "tvg_offset" in vList:
                self.tvg_offset = self.dataset[tvg_name]
        except:
            print("An error occurred")

    def __init__(self, filename):
        # open the file
        self.dataset = None
        self.file_name = filename
        self.dataset = open_nc_file(self.file_name)
        self.__retrieve_known_variables__()
        self.ignore_variable = []

    def __del__(self):
        # close the file
        if self.dataset is not None:
            self.dataset.close()

    def get_variable_histogram(self, path):
        """
        Compute histogram data for a given variable
        """
        variablePath = self.dataset[path]
        hist, bin_edges = np.histogram(np.array(variablePath), bins=256)
        return hist, bin_edges

    # pylint: disable=W0613
    def get_variable_data(self, variable_path, index=0, substract_tvg_offset=False):
        """
        Return the variable as a np array matrix (either 1D or 2D)
        :param variable_path: path to the variable name
        :param substract_tvg_offset: do we substract TVG offset trying to something close to an absolute level value for backscatter
        :param index: the current swath id, for WC data it will allow to load the current swath data only
        :return: a np array
        """
        variablePath = self.dataset[variable_path]
        if self.is_vlen(variable_path):
            return self.get_vlen_variable(variable_path, index)
        return np.array(variablePath).transpose()

    def get_variable(self, variable_path):
        return self.dataset[variable_path]

    def is_vlen(self, variable_name):
        """
        Check if the given variable is a variable length variable (in xsf variable length definition)
        """
        v = self.dataset[variable_name]
        # noinspection PyProtectedMember
        return v._isvlen

    def is_ignored(self, variable_name):
        """
        Tell is a variable is masked variable (ancillary variable for example)
        """
        return variable_name in self.ignore_variable

    def get_vlen_variable(self, variable_path, slice_index):
        """
        retrieve a matrix containing a slice (matrix) of all vlen data, filled with NaN values
        """
        vlen_variable = self.dataset[variable_path]
        shape = vlen_variable.shape
        # retrieve max size
        max_samples = 0
        ping_values = vlen_variable[slice_index]
        if len(shape) == 1:
            # this is a considered to be a ping indexed one dimensionnal variable
            return ping_values
        elif len(shape) == 2:
            # initialize ping (with nans)\n",
            for sub_array in ping_values:
                max_samples = max(max_samples, len(sub_array))

            # fill ping with data\n",
            matrix = np.full((shape[1], max_samples), dtype="float32", fill_value=float(np.nan))

            for bnr in range(shape[1]):
                # Warning, auto scale if set by default ping[bnr][:count] = sample_amplitude[start:stop]
                count = len(ping_values[bnr])
                matrix[bnr][:count] = ping_values[bnr]
            return matrix.transpose()
        elif len(shape) == 3 and shape[2] == 1:
            # initialize ping (with nans)\n",
            for sub_array in ping_values:
                max_samples = max(max_samples, len(sub_array[:][0]))

            # fill ping with data\n",
            matrix = np.full((shape[1], max_samples), dtype="float32", fill_value=float(np.nan))

            for bnr in range(self.beam_count):
                # Warning, auto scale if set by default ping[bnr][:count] = sample_amplitude[start:stop]
                count = len(ping_values[bnr][0])
                matrix[bnr][:count] = ping_values[bnr][0]
            return matrix.transpose()
        else:
            raise NotImplementedError("Not supported vlen variable")
