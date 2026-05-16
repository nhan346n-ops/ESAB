import os.path

import matplotlib.pyplot as plt
import numpy as np
import sonar_netcdf.sonar_groups as sg

# pylint: disable=import-error
from bokeh.io import output_file, output_notebook
from bokeh.models import Circle, ColorBar, ColumnDataSource, Legend, LegendItem, Range1d
from bokeh.models.tools import HoverTool
from bokeh.palettes import Category10_4, Category10_10
from bokeh.plotting import figure, show
from bokeh.transform import linear_cmap
from sonar_netcdf.utils.print_color import info, warning

from pyat.sonarscope.cruise_summary.global_data import GlobalDataModel
from pyat.utils import numpy_utils
from pyat.utils.execution_context import is_debug, is_running_from_ipython

# pylint: enable=import-error


plt.rcParams["figure.dpi"] = 200


def init_context():
    """Init context for plotting"""
    if is_running_from_ipython():
        output_notebook()
        # disable numpy warning
    if not is_debug():
        numpy_utils.disable_warning()
    plt.rcParams["figure.dpi"] = 200


# static initialisation of plotter context
init_context()


class BasePlotter:
    def __init__(self, workdir: str, data: GlobalDataModel):
        self.workdir = workdir
        self.data = data
        # compute per file color index for display
        color_index = 0
        self.color_dict = {}
        max_color = len(Category10_10)
        for f in data.file_data.keys():
            color_index = color_index % max_color
            self.color_dict[f] = Category10_10[color_index]
            color_index += 1


class Plotter(BasePlotter):
    def __get_output_graph_path(self, key_name: str):
        """Compute the file name of the bokeh graph use to save and retrieve charts"""
        return os.path.join(self.workdir, f"{key_name}.html")

    def __plot_geovariable_abstract(
        self,
        title: str,
        output_filename,
        color_mapper,
        value_retriever_func,
        use_finest_level=False,
    ):
        # write to file if not running in notebook
        if not is_running_from_ipython() and output_filename is not None:
            output_file(output_filename)
        longitude_variable = sg.BeamGroup1Grp.PLATFORM_LONGITUDE_VNAME
        latitude_variable = sg.BeamGroup1Grp.PLATFORM_LATITUDE_VNAME
        pingtime_variable = sg.BeamGroup1Grp.PING_TIME_VNAME
        _figure = figure(
            title=title,
            x_axis_label=longitude_variable,
            y_axis_label=latitude_variable,
            sizing_mode="stretch_width",
            height=400,
        )
        _figure.add_tools(
            HoverTool(
                tooltips=[
                    ("Name", "$name"),
                    ("(longitude,latitude)", "(@x, @y)"),
                    ("value", "@value"),
                    ("datetime", "@datetime{%Y-%m-%d %H:%M:%S.%3Ns}"),
                ],
                formatters={"@datetime": "datetime"},
            )
        )

        for f, v in self.data.file_data.items():
            color = self.color_dict[f]
            ds = v.ping_time_dataset.xr_dataset if use_finest_level else v.decimated_dataset
            _figure.xaxis.axis_label = ds[longitude_variable].long_name
            _figure.yaxis.axis_label = ds[latitude_variable].long_name
            x = np.asarray(ds[longitude_variable])
            y = np.asarray(ds[latitude_variable])
            pingtime = np.asarray(ds[pingtime_variable])
            values = value_retriever_func(ds)
            if values is not None:
                source = ColumnDataSource(data={"x": x, "y": y, "value": values, "datetime": pingtime})
                _figure.scatter(source=source, x="x", y="y", color=color_mapper, name=os.path.basename(f))

        return _figure

    def plot_geovariable_discrete(
        self,
        title: str,
        variable_name: str,
        graphic_name: str = None,
        labels=None,
        color_palette=Category10_4,
        use_finest_level=False,
    ):
        if graphic_name is None:
            graphic_name = variable_name

        def retriever_value(dataset):
            values = np.asarray(dataset[variable_name])
            return values

        self._plot_geovariable_discrete(
            graphic_name=graphic_name,
            title=title,
            labels=labels,
            color_palette=color_palette,
            value_computer_func=retriever_value,
            use_finest_level=use_finest_level,
        )

    def _plot_geovariable_discrete(
        self,
        title: str,
        graphic_name: str,
        value_computer_func,
        labels=None,
        color_palette=Category10_4,
        use_finest_level=False,
    ):
        # initialize label and color palette
        if not labels:
            labels = {"Unknown": 0}
        min_value = min(labels.values())
        max_value = max(labels.values())
        if max_value - min_value + 1 > len(color_palette):
            warning(f"For plot {graphic_name} color palette does not have enough value to match all cases ")

        color_mapper = linear_cmap(field_name="value", palette=color_palette, low=min_value, high=max_value)

        _figure = self.__plot_geovariable_abstract(
            title=title,
            output_filename=self.__get_output_graph_path(graphic_name),
            color_mapper=color_mapper,
            value_retriever_func=value_computer_func,
            use_finest_level=use_finest_level,
        )

        # create discrete legend
        items = []

        for (k, v), color in zip(labels.items(), color_palette):
            # find color for this value
            circle = Circle(size=6, fill_color=color)
            renderer = _figure.add_glyph(circle)
            itm = LegendItem(label=k, renderers=[renderer])
            items.append(itm)
        _figure.add_layout(Legend(items=items))

        show(_figure)

    def plot_geovariable_contigous(
        self,
        title: str,
        variable_name: str,
        graphic_name: str = None,
        min_value=None,
        max_value=None,
        color_palette="Viridis256",  # Magma, Inferno, Plasma, and Viridis
        use_finest_level=False,
    ):
        if graphic_name is None:
            graphic_name = variable_name
        # set min and max values for colorbar
        if min_value is None:
            min_value = self.data.metadata.variable_metadata[variable_name].min_value
        if max_value is None:
            max_value = self.data.metadata.variable_metadata[variable_name].max_value

        def retriever_value(dataset):
            values = np.asarray(dataset[variable_name])
            return values

        self._plot_geovariable_contigous(
            title=title,
            graphic_name=graphic_name,
            value_computer_func=retriever_value,
            min_value=min_value,
            max_value=max_value,
            color_palette=color_palette,
            use_finest_level=use_finest_level,
        )

    def _plot_geovariable_contigous(
        self,
        title: str,
        graphic_name: str,
        value_computer_func,
        min_value,
        max_value,
        color_palette="Viridis256",  # Magma, Inferno, Plasma, and Viridis
        use_finest_level=False,
    ):
        """Plot a georeferenced variable along platform latitude and longitude
        Values are supposed to be contiguous and a color bar is added between min and max value
        """
        # initialize label and color palette
        color_mapper = linear_cmap(field_name="value", palette=color_palette, low=min_value, high=max_value)
        _figure = self.__plot_geovariable_abstract(
            title=title,
            output_filename=self.__get_output_graph_path(graphic_name),
            color_mapper=color_mapper,
            value_retriever_func=value_computer_func,
            use_finest_level=use_finest_level,
        )

        color_bar = ColorBar(color_mapper=color_mapper["transform"])
        _figure.add_layout(color_bar, "right")
        show(_figure)

    def plot_ping_time_variable(self, variable_name, title, use_finest_level=False):
        """Plot ping time variable, color is by file"""
        if len(self.data.file_data) <= 0:
            warning(f"No file to display")
            raise FileNotFoundError(f"No file to display")
        if not is_running_from_ipython():
            output_file(self.__get_output_graph_path(variable_name))

        first_data = list(self.data.file_data.values())[0]
        ds = first_data.ping_time_dataset.xr_dataset if use_finest_level else first_data.decimated_dataset
        first_values = ds[variable_name]
        _figure = figure(
            title=title,
            x_axis_label=first_values.ping_time.long_name,
            x_axis_type="datetime",
            y_axis_label=first_values.long_name,
            sizing_mode="stretch_width",
            height=400,
        )
        hoovertool = HoverTool(
            tooltips=[("Name", "$name"), ("(datetime,value)", "(@x{%Y-%m-%d %H:%M:%S.%3Ns}, @y)")],
            formatters={"@x": "datetime"},
        )
        _figure.add_tools(hoovertool)
        for f, v in self.data.file_data.items():
            color = self.color_dict[f]
            ds = v.ping_time_dataset.xr_dataset if use_finest_level else v.decimated_dataset
            values = ds[variable_name]
            values_time = values.ping_time
            _figure.line(x=values_time.to_numpy(), y=values.to_numpy(), color=color, name=os.path.basename(f))
        show(_figure)
        return _figure

    def plot_modes(self, title, variable_name: str, values_dict, labels):
        """Plot all modes for the set of files,"""
        if len(self.data.file_data) <= 0:
            warning(f"No file to display")
            raise FileNotFoundError(f"No file to display")
        if not is_running_from_ipython():
            output_file(self.__get_output_graph_path(variable_name))

        _figure = figure(
            title=title,
            sizing_mode="stretch_width",
            height=400,
        )
        hoovertool = HoverTool(
            tooltips=[("Name", "$name"), ("(x,value)", "(@x, @y)"), ("(ping,value)", "(@ping_index, @y)")],
        )
        _figure.add_tools(hoovertool)
        offset = 0  # offset to have a linear representation of ping modes
        for f, v in values_dict.items():
            color = self.color_dict[f]
            x_arr = np.arange(offset, offset + len(v))
            ping_index = np.arange(0, len(v))
            offset = offset + len(v)
            source = ColumnDataSource(data={"x": x_arr, "y": v, "ping_index": ping_index})

            _figure.line(source=source, color=color, name=os.path.basename(f))

        # create discrete legend
        # items = []
        # for k, v in labels.items():
        #    itm = LegendItem(label=f"{v}: {k})")
        #    items.append(itm)
        # _figure.add_layout(Legend(items=items), place="below")

        show(_figure)

        # simply print legend after figure to avoid overlapping
        for k, v in labels.items():
            info(f"{v}: {k})")

        return _figure

    def plot_sound_speed_profiles(self, max_depth):
        """retrieve and plot all sound speed profile
        If repeated they will be plotted several times
        """
        _figure = figure(
            title="Sound speed profiles (as found in files)",
            x_axis_label="sound speed (m/s)",
            y_axis_label="depth of sample below surface (m)",
            sizing_mode="stretch_width",
            height=400,
        )
        _figure.add_tools(HoverTool(tooltips=[("Name", "$name"), ("(speed,depth)", "($x, $y)")]))
        min_sound_speed = None
        max_sound_speed = None
        counter = 0

        for f, v in self.data.file_data.items():
            depth_values = v.svp_dataset.xr_dataset[sg.SoundSpeedProfileGrp.SAMPLE_DEPTH_VNAME].values
            depth_values *= -1
            speed_values = v.svp_dataset.xr_dataset[sg.SoundSpeedProfileGrp.SOUND_SPEED_VNAME].values
            color = self.color_dict[f]
            # loop over each profile to get the list of samples

            for i in range(v.svp_dataset.xr_dataset.dims[sg.SoundSpeedProfileGrp.PROFILE_TIME_VNAME]):
                # we compute the min/max speed values for data that are less that at depth value
                if not np.isnan(max_depth):
                    value_over_depth_limit = speed_values[i][depth_values[i] > max_depth]
                    file_min_ssp = np.nanmin(value_over_depth_limit)
                    file_max_ssp = np.nanmax(value_over_depth_limit)
                    if min_sound_speed is None:
                        min_sound_speed = file_min_ssp
                    else:
                        min_sound_speed = np.nanmin(file_min_ssp)
                    if max_sound_speed is None:
                        max_sound_speed = file_max_ssp
                    else:
                        max_sound_speed = np.nanmin(file_max_ssp)
                _figure.line(
                    x=speed_values[i],
                    y=depth_values[i],
                    color=color,
                    name=f,
                )
                _figure.circle(
                    x=speed_values[i],
                    y=depth_values[i],
                    color=color,
                    name=f,
                )
                counter += 1
        if not np.isnan(max_depth):
            # set zoom from 0 to max depth
            _figure.y_range = Range1d(max_depth, 0)
            # update x zoom, keep x% of free space
            space = np.abs(max_sound_speed - min_sound_speed) * 0.05
            _figure.x_range = Range1d(min_sound_speed - space, max_sound_speed + space)
        info(f"Sound speed profile display {counter} profiles (that may be identical)")
        show(_figure)

    # def debug(self):
    #     first_data = list(self.data.file_data.values())[0]
    #     first_values = first_data.decimated_dataset
    #     longitude_variable = "platform_longitude"
    #     latitude_variable = "platform_latitude"
    #     output_filename = self.__get_output_graph_path(f"navigation")
    #     output_file(output_filename)
    #
    #     _figure = figure(
    #         title="Navigation (latitudes, longitudes)",
    #         x_axis_label=first_values[longitude_variable].long_name,
    #         y_axis_label=first_values[latitude_variable].long_name,
    #         sizing_mode="stretch_width",
    #         height=400,
    #     )
    #     x = first_values[longitude_variable].to_numpy(),
    #     y = first_values[latitude_variable].to_numpy(),
    #     _figure.add_tools(HoverTool(tooltips=[("Name", "$name"), ("(x,y)", "($x, $y)")]))
    #     source = ColumnDataSource(data=dict(x=x[0], y=y[0], value=y[0]))
    #
    #     _figure.scatter(
    #
    #         x='x',
    #         y='y',
    #         source=source,
    #         color='#1f77b4',
    #         name="dflskfmls",
    #     )
    #     show(_figure)
    #
    #     print("done")
    def plot_navigation(self, output_filename=None, use_finest_level=False, show_figure=True):
        """Plot decimated navigation with interactive graph"""
        if output_filename is None:
            output_filename = self.__get_output_graph_path(f"navigation")
        if len(self.data.file_data) <= 0:
            warning(f"No file to display")
            raise FileNotFoundError(f"No file to display")

        if not is_running_from_ipython():
            output_file(output_filename)

        first_data = list(self.data.file_data.values())[0]
        first_values = first_data.ping_time_dataset if use_finest_level else first_data.decimated_dataset
        longitude_variable = "platform_longitude"
        latitude_variable = "platform_latitude"
        _figure = figure(
            title="Navigation (latitudes, longitudes)",
            x_axis_label=first_values[longitude_variable].long_name,
            y_axis_label=first_values[latitude_variable].long_name,
            sizing_mode="stretch_width",
            height=400,
        )
        _figure.add_tools(HoverTool(tooltips=[("Name", "$name"), ("(longitude,latitude)", "($x, $y)")]))
        for f, v in self.data.file_data.items():
            color = self.color_dict[f]
            ds = v.ping_time_dataset.xr_dataset if use_finest_level else v.decimated_dataset
            _figure.line(
                x=ds[longitude_variable].to_numpy(),
                y=ds[latitude_variable].to_numpy(),
                color=color,
                name=os.path.basename(f),
            )
        if show_figure:
            show(_figure)
        return _figure
