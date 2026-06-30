from pathlib import Path

import netCDF4 as nc
import numpy as np
import sonar_netcdf.sonar_groups as sonar

from pyat.utils.nc_encoding import open_nc_file


class TimeBoundingBox:
    """Extract time bounding boxe from sonar file"""

    def __init__(self):
        pass

    def __call__(self, input_files: str):
        timed_dict = {}
        for input_file in input_files:
            first_value = None
            last_value = None
            with open_nc_file(
                input_file,
                mode="r",
            ) as file_in:
                for beam_group_name in file_in[sonar.SonarGrp.get_group_path()].groups:
                    # retrieve first ping time
                    first_ping_time = file_in[sonar.BeamGroup1Grp.PING_TIME(beam_group_name)]
                    if first_value is None:
                        first_value = first_ping_time[0]
                    first_value = min(first_value, first_ping_time[0])
                    if last_value is None:
                        last_value = first_ping_time[-1]
                    last_value = max(first_ping_time[-1], last_value)
            timed_dict[input_file] = (first_value.astype("datetime64[ns]"), last_value.astype("datetime64[ns]"))
        return timed_dict


class Merger:
    """
    Merger will merge several sonar-netcdf file along
    """

    def __init__(self):
        pass

    def __sort_inputs(self, inputs):
        timed_values = {}
        _bb = TimeBoundingBox()
        time_inputs = _bb(inputs)
        # pylint:disable = consider-using-dict-items
        for input_file in time_inputs:
            (start_date, stop_date) = time_inputs[input_file]
            print(f"file {input_file} starting date {start_date}")
            timed_values[input_file] = start_date
        sorted_list = sorted(inputs, key=lambda x: timed_values[x])
        return sorted_list

    def _overridde_data(self, input_variable: nc.Variable, output_variable: nc.Variable):
        """copy all data from input dataset to output dataset
        dimensions are expected to match
        """
        output_variable[:] = input_variable[:]

    def _append_data(
        self,
        input_variable: nc.Variable,
        output_variable: nc.Variable,
        unlimited_dim: dict,
    ):
        """
        Append data along unlimited dimensions, do nothing is no dimension is unlimited
        :return:
        """
        # retrieve variable dimension :
        appendable_dimensions = {key: unlimited_dim[key] for key in output_variable.dimensions if key in unlimited_dim}
        # if no dimension is appendable, we skip data
        if len(appendable_dimensions) == 0:
            return
        if len(appendable_dimensions) > 1:
            print(f"WARNING for variable {output_variable.name} appending in more than 1 vlen variable dimension")
        # construct slice
        slicing = ()
        for dim_name in input_variable.dimensions:
            if dim_name in appendable_dimensions:
                starting = appendable_dimensions[dim_name]
                slicing += (slice(starting, None, 1),)  # we select
            else:
                slicing += (slice(None),)  # we select all data
        # print(f"append data for  {input_variable.name} with slice {slicing}")
        output_variable[slicing] = input_variable[:]

    # def _duplicate_dataset(self, input_ds :nc.Dataset, output_ds :nc.Dataset)->None :
    #     """ copy one dataset from input to another dataset"""
    #
    #
    #     # copy group attributes
    #     output_ds.setncatts(input_ds.ncattrs())
    #
    #     # copy cmptypes declaration
    #     for name, cmp in input_ds.cmptypes.items():
    #         output_ds.createCompoundType(datatype=cmp.dtype, datatype_name=name)
    #
    #     #copy enum types declaration
    #     for name,enum in input_ds.enumtypes.items():
    #         output_ds.createEnumType(datatype=enum.dtype, datatype_name=name, enum_dict=enum.enum_dict)
    #
    #     # copy vlen types declaration
    #     for name, vl in input_ds.vltypes.items():
    #         output_ds.createVLType(datatype=vl.dtype, datatype_name=name)
    #
    #     # Copy dimensions
    #     for dname, the_dim in input_ds.dimensions.items():
    #         output_ds.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)
    #     # Copy variables
    #     """Copy variables based on the input file."""
    #     for name, variable in input_ds.variables.items():
    #         v = output_ds.createVariable(
    #             name,
    #             variable.datatype,
    #             variable.dimensions,
    #             fill_value=variable._FillValue if hasattr(variable, "_FillValue") else None,
    #         )
    #         # copy attributes
    #         v.setncatts(variable.__dict__)
    #         # copy variable
    #         v[:] = variable[:]
    #
    #     #copy subgroups
    #     for name in input_ds.groups:
    #         out_subgroup=output_ds.createGroup(name)
    #         self._duplicate_dataset(out_subgroup, input_ds.groups[name])
    #

    def __merge_group(self, dataset_out: nc.Dataset, dataset_in: nc.Dataset, unlimited_dimensions):

        #########
        # ATTRIBUTES
        #########
        output_attributes = dataset_out.ncattrs()
        # copy attributes if not already existing
        for att in dataset_in.ncattrs():
            if att not in output_attributes:
                dataset_out.setncattr(att, dataset_in.getncattr(att))
        # copy cmptypes declaration
        for name, cmp in dataset_in.cmptypes.items():
            if name not in dataset_out.cmptypes:
                dataset_out.createCompoundType(datatype=cmp.dtype, datatype_name=name)

        # copy enum types declaration
        for name, enum_value in dataset_in.enumtypes.items():
            if name not in dataset_out.enumtypes:
                dataset_out.createEnumType(
                    datatype=enum_value.dtype, datatype_name=name, enum_dict=enum_value.enum_dict
                )

        # copy vlen types declaration
        for name, vl in dataset_in.vltypes.items():
            if name not in dataset_out.vltypes:
                dataset_out.createVLType(datatype=vl.dtype, datatype_name=name)

        #########
        # DIMENSIONS
        #########
        # copy dimensions if not already existing, if existing behaviour depends on their types
        output_dimensions = dataset_out.dimensions
        for dimension_name in dataset_in.dimensions:
            input_dim = dataset_in.dimensions[dimension_name]
            if dimension_name not in output_dimensions:
                if input_dim.isunlimited():
                    dataset_out.createDimension(dimension_name, None)
                else:
                    dataset_out.createDimension(dimension_name, input_dim.size)
            else:  # input dim already exists, check compliance
                out_put_dim = dataset_out.dimensions[dimension_name]
                if input_dim.isunlimited():
                    if not out_put_dim.isunlimited():
                        raise Exception(
                            f"Dimension {dimension_name} input dimension is unlimited but not output dimension)"
                        )
                    # dimension are compliant : they can be merged
                elif out_put_dim.size != input_dim.size:
                    raise Exception(
                        f"Dimension {dimension_name} differs between input files ({out_put_dim.size} vs {input_dim.size})"
                    )
        # retrieve variable dimension :

        # construct a dictionnary of unlimited dimension and initial values before any merge operation
        my_group_unlimited_dims = {
            key: len(value) for key, value in dataset_out.dimensions.items() if value.isunlimited()
        }
        # merge two dictionaries
        unlimited_dimensions = {**unlimited_dimensions, **my_group_unlimited_dims}

        # Variables
        # Copy variables based on the input file.
        for name, input_variable in dataset_in.variables.items():
            # create variable if not already existing
            if name not in dataset_out.variables:
                # retrieve compression parameters
                zlib = None
                complevel = 0
                chunksizes = None
                if input_variable.filters() is not None and not input_variable._isvlen:
                    zlib = (
                        sonar.DEFAULT_COMPRESSION_LIB
                        if input_variable.filters().get(sonar.DEFAULT_COMPRESSION_LIB, False)
                        else None
                    )
                    complevel = input_variable.filters().get("complevel", sonar.DEFAULT_COMPRESSION_LEVEL)
                if input_variable.chunking() is not None:
                    chunksizes = (
                        tuple(input_variable.chunking())
                        if (
                            input_variable.chunking() != "contiguous"
                            and tuple(input_variable.chunking()) != input_variable.shape
                        )
                        else None
                    )

                v = dataset_out.createVariable(
                    name,
                    input_variable.datatype,
                    input_variable.dimensions,
                    fill_value=input_variable._FillValue if hasattr(input_variable, "_FillValue") else None,
                    compression=zlib,
                    complevel=complevel,
                    chunksizes=chunksizes,
                )
                # copy attributes
                v.setncatts(input_variable.__dict__)
                # copy data only if they do not already exists
                self._overridde_data(input_variable, v)
            else:
                output_variable = dataset_out.variables[name]
                # copy variable data
                # we copy variables data only if one dimensions of data is appendable, otherwise

                # if at least one dimension is considered as appendable,,we will append data along this dimension
                # v[:] = variable[:]
                self._append_data(
                    input_variable=input_variable,
                    output_variable=output_variable,
                    unlimited_dim=unlimited_dimensions,
                )

        # merge sub groups
        for subgroup_name in dataset_in.groups:
            # create subgroup if not exists
            if subgroup_name not in dataset_out.groups:
                subgroup_out = dataset_out.createGroup(subgroup_name)
            self.__merge_group(
                dataset_out=dataset_out.groups[subgroup_name],
                dataset_in=dataset_in.groups[subgroup_name],
                unlimited_dimensions=unlimited_dimensions,
            )

    def merge(self, inputs, output_file_name: Path):
        """

        :param self:
        :param inputs: list of files to merger
        :return:
        """

        # first sort input_file files
        sorted_files = self.__sort_inputs(inputs)

        with open_nc_file(
            output_file_name,
            mode="w",
        ) as output:
            for input_file in sorted_files:
                # open a file
                print(f"Merging {input_file} file")
                with open_nc_file(input_file) as file_in:
                    # for each group in input_file file
                    self.__merge_group(dataset_out=output, dataset_in=file_in, unlimited_dimensions={})


def process_run(run_name: str):
    input_directory = f"X:/Campagnes/GAZCOGNE3_Thalassa/HERMES/DONNEES/{run_name}"
    print(f"starting sonar netcdf WC converter for {input_directory}")

    path = Path(input_directory)
    files = list(path.glob("GAZCOGNE3*.xsf.nc"))
    #
    bb = TimeBoundingBox()
    times = bb(files)
    cumul_time = np.timedelta64(0, "m")
    file_list_to_merge = []
    current_set = []
    keys = sorted(times.keys())
    for file in keys:
        start, stop = times[file]
        print(f"processing file{file} {start} to {stop}")
        delta = stop - start
        delta = np.abs(delta)
        delta = np.timedelta64(delta, "m")
        if cumul_time + delta > np.timedelta64(60, "m"):
            # With this file will be more than an hour, we create a new set of file to merge
            file_list_to_merge.append(current_set)
            current_set = [file]
            cumul_time = delta
        else:
            # we add this file to the list of file to merge and go on
            current_set.append(file)
            cumul_time += delta
    # do not forget to append the last set of files
    file_list_to_merge.append(current_set)

    file_count = np.sum([len(e) for e in file_list_to_merge])
    assert file_count == len(files)

    merger = Merger()

    for merge_input in file_list_to_merge:
        first_file = merge_input[0].stem
        last_file = merge_input[-1].stem
        if first_file == last_file:
            out_file_name = first_file
        else:
            out_file_name = f"MERGED_{first_file}_TO_{last_file}.nc"
        print(f"Starting merge operation create file {out_file_name} from files {[str(f.name) for f in merge_input]}")
        merger.merge(inputs=merge_input, output_file_name=path.joinpath(out_file_name))


if __name__ == "__main__":
    #    merger.merge_with_MFDataset(output_file_name=path.joinpath("out.nc"),input_files=files)
    for i in range(1, 7):
        if i != 3:  # already done
            process_run(f"RUN00{i}")
