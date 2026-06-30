# pylint: disable=import-error
import hvplot
import hvplot.interactive
import hvplot.xarray
import panel as pn

from pyat.sonarscope.bs_correction.mean_bs_model import BackscatterCurve, MeanBSModel

# pylint: enable=import-error


class BsarEditor:
    def __init__(self, mean_bs_model: MeanBSModel):
        self.mean_bs_model = mean_bs_model
        self.key_dict = {str(key): key for key in self.mean_bs_model.model.keys()}

    def incidence_ds(self, key):
        return self.mean_bs_model.model[key][0].ds

    def transmission_ds(self, key):
        return self.mean_bs_model.model[key][1].ds

    def edit(self, show_raw_data: bool = False):

        # Set up widgets
        self.mode_widget = pn.widgets.Select(
            name="mode",
            options=dict(sorted({str(key): key for key in self.mean_bs_model.model.keys()}.items())),
            width_policy="fit",
        )

        self.bound_incidence_ds = hvplot.bind(
            self.incidence_ds,
            self.mode_widget,
        ).interactive()
        self.bound_transmission_ds = hvplot.bind(
            self.transmission_ds,
            self.mode_widget,
        ).interactive()

        self.curve_plot = pn.Card(
            (
                self.bound_incidence_ds.hvplot.line(
                    x=BackscatterCurve.ANGLE,
                    y=(
                        [BackscatterCurve.MEAN_BS, BackscatterCurve.RAW_MEAN_BS]
                        if show_raw_data
                        else BackscatterCurve.MEAN_BS
                    ),
                    framewise=False,
                    grid=True,
                )
                .output()
                .opts(show_legend=False)
                * self.bound_transmission_ds.hvplot.line(
                    x=BackscatterCurve.ANGLE,
                    y=(
                        [BackscatterCurve.MEAN_RESIDUAL_BS, BackscatterCurve.RAW_MEAN_RESIDUAL_BS]
                        if show_raw_data
                        else BackscatterCurve.MEAN_RESIDUAL_BS
                    ),
                    framewise=False,
                    grid=True,
                )
                .overlay()
                .output()
            ).opts(show_legend=False)
        )

        self.counts_plot = pn.Card(
            (
                self.bound_incidence_ds.hvplot.step(
                    y=BackscatterCurve.VALUE_COUNT,
                    framewise=False,
                    grid=True,
                ).output()
                * self.bound_transmission_ds.hvplot.step(
                    x=BackscatterCurve.ANGLE,
                    y=BackscatterCurve.VALUE_COUNT,
                    framewise=False,
                    grid=True,
                )
                .overlay()
                .output()
            ).opts(show_legend=False),
        )

        return pn.Column(
            self.mode_widget,
            self.curve_plot,
            self.counts_plot,
        ).servable()
