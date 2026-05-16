# pylint: disable=import-error
import hvplot.interactive
import hvplot.xarray
import numpy as np
import panel as pn
import xarray as xr

from pyat.sonarscope.bs_correction.gsab_model import GsabDataCoefficients, GsabDataModel

# pylint: enable=import-error


class GsabEditor:
    def __init__(self, gsab_model: GsabDataModel):
        self.gsab_model = gsab_model

    def gsab_ds(self, a, b, c, d, e, f):
        self.gsab_model.coeffs = GsabDataCoefficients([a, b, c, d, e, f])
        linear_coeffs = self.gsab_model.coeffs.linear_coeffs()

        return xr.Dataset(
            data_vars={
                "bs_values": self.gsab_model.y,
                "gsab_fit": self.gsab_model.apply_gsab_func(linear_coeffs=linear_coeffs),
                "ab_comp": self.gsab_model.apply_gsab_func(linear_coeffs=linear_coeffs.ab_coeffs()),
                "cd_comp": self.gsab_model.apply_gsab_func(linear_coeffs=linear_coeffs.cd_coeffs()),
                "ef_comp": self.gsab_model.apply_gsab_func(linear_coeffs=linear_coeffs.ef_coeffs()),
            },
            coords={
                "angle": self.gsab_model.x,
            },
        )

    def refit(self, _):
        # parameter Event not used, but required by the button callback
        coeffs = self.gsab_model.fit_gsab()
        # Use a local copy of coeffs to managed reactive updates
        self.a_widget.value = coeffs.a
        self.b_widget.value = coeffs.b
        self.c_widget.value = coeffs.c
        self.d_widget.value = coeffs.d
        self.e_widget.value = coeffs.e
        self.f_widget.value = coeffs.f

    def edit(self):
        # first fit
        self.gsab_model.fit_gsab()

        fmt = "1[.]000"
        step = 0.001
        # Set up widgets
        self.a_widget = pn.widgets.EditableFloatSlider(
            name="a (dB)",
            format=fmt,
            value=self.gsab_model.coeffs.a,
            start=self.gsab_model.coeffs_min.a,
            end=self.gsab_model.coeffs_max.a,
            step=step,
        )
        self.b_widget = pn.widgets.EditableFloatSlider(
            name="b (deg)",
            format=fmt,
            value=self.gsab_model.coeffs.b,
            start=self.gsab_model.coeffs_min.b,
            end=self.gsab_model.coeffs_max.b,
            step=step,
        )
        self.c_widget = pn.widgets.EditableFloatSlider(
            name="c (dB)",
            format=fmt,
            value=self.gsab_model.coeffs.c,
            start=self.gsab_model.coeffs_min.c,
            end=self.gsab_model.coeffs_max.c,
            step=step,
        )
        self.d_widget = pn.widgets.EditableFloatSlider(
            name="d (deg)",
            format=fmt,
            value=self.gsab_model.coeffs.d,
            start=self.gsab_model.coeffs_min.d,
            end=self.gsab_model.coeffs_max.d,
            step=step,
        )
        self.e_widget = pn.widgets.EditableFloatSlider(
            name="e (dB)",
            format=fmt,
            value=self.gsab_model.coeffs.e,
            start=self.gsab_model.coeffs_min.e,
            end=self.gsab_model.coeffs_max.e,
            step=step,
        )
        self.f_widget = pn.widgets.EditableFloatSlider(
            name="f (deg)",
            format=fmt,
            value=self.gsab_model.coeffs.f,
            start=self.gsab_model.coeffs_min.f,
            end=self.gsab_model.coeffs_max.f,
            step=step,
        )

        self.refit_btn = pn.widgets.Button(name="Refit")
        self.refit_btn.on_click(self.refit)

        # Set up callbacks
        self.bound_ds = hvplot.bind(
            self.gsab_ds,
            a=self.a_widget,
            b=self.b_widget,
            c=self.c_widget,
            d=self.d_widget,
            e=self.e_widget,
            f=self.f_widget,
        ).interactive(loc="left")

        return pn.Column(
            self.bound_ds.hvplot(
                y=["bs_values", "gsab_fit", "ab_comp", "cd_comp", "ef_comp"],
                ylim=[float(np.nanmin(self.gsab_model.y)) - 3.0, float(np.nanmax(self.gsab_model.y)) + 3.0],
                line_dash=["solid", "solid", "dashed", "dashed", "dashed"],
                framewise=False,
                legend="top_right",
                grid=True,
            ),
            self.refit_btn,
        )
