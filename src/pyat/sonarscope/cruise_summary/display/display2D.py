import os.path

# disable pylint for bokeh files
# pylint: disable=import-error
from bokeh.io import output_file
from bokeh.plotting import figure, show
from sonar_netcdf.utils.print_color import warning

from pyat.sonarscope.cruise_summary.display.display import BasePlotter
from pyat.sonarscope.cruise_summary.global_data import GlobalDataModel
from pyat.sonarscope.model.constants import VariableDim as DimDef
from pyat.utils.execution_context import is_running_from_ipython

# pylint: enable=import-error


class Plotter2(BasePlotter):
    def __init__(self, workdir: str, data: GlobalDataModel):
        super().__init__(workdir=workdir, data=data)

    def __get_output_graph_path(self, key_name: str):
        """Compute the file name of the bokeh graph use to save and retrieve charts"""
        return os.path.join(self.workdir, f"{key_name}_dd.html")

    def plot_ping_time_variable(self, variable_name, title):
        """
        Plot ping time variable, variable is 2D, and is plot along ping time axis, given the assumption that ping_time is the first dimension
        Data is appended along the ping direction, and maximum length along the x direction is used
        """
        if len(self.data.file_data) <= 0:
            warning(f"No file to display")
            raise FileNotFoundError(f"No file to display")
        if not is_running_from_ipython():
            output_file(self.__get_output_graph_path(variable_name + f"{DimDef.PING_DIM}_{DimDef.DETECTION_DIM}"))

        # hoovertool= HoverTool(tooltips=[("Name", "$name"), ("(datetime,value)", "(@x{%Y-%m-%d %H:%M:%S.%3Ns}, @y)")], formatters={'@x':'datetime'})
        # _figure.add_tools(hoovertool)
        x_length = 0
        y_max = 0
        arrays = []
        for f, v in self.data.file_data.items():
            ds = v.ping_detection_dataset.xr_dataset
            values = ds[variable_name]
            arrays.append(values.to_numpy().T)
            x_length += values.shape[0]
            y_max = max(y_max, values.shape[1])

        _figure = figure(
            title=title,
            x_axis_label=DimDef.PING_DIM,
            # x_axis_type="ping",
            y_axis_label=DimDef.DETECTION_DIM,
            sizing_mode="stretch_width",
            height=300,
            x_range=(0, x_length),
            y_range=(0, y_max),
        )

        _figure.image(image=arrays, x=0, y=0, dw=x_length, dh=y_max, palette="Greys256")
        show(_figure)
        return _figure
