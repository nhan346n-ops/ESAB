"""Display utilities for debugging purpose mainly"""

import matplotlib.pyplot as plt

from pyat.sonarscope.bs_correction.stats_computer import BackscatterCurve, MeanBSModel
from pyat.utils.exceptions.exception_list import BadParameter


def plot(mean_model:MeanBSModel, curve_per_file=None, display_count=False):
    """plot tools for display of curves computed with stats_computer.compute
        plot the main curve, and if available the curves per file
    """
    for mode in mean_model.model.keys():
        if curve_per_file:
            if mode not in curve_per_file.keys():
                raise BadParameter(
                    f"Unexpected case, mode {mode} in not available in per_file curves, but availaible in main_curve")

        mean_fig, count_fig = None, None
        if display_count:
            fig, (mean_fig, count_fig) = plt.subplots(2, 1)
            count_fig.set_title('value count')
            count_fig.grid()
            count_fig.set_ylabel("value count")
            count_fig.set_xlabel("angle (degree)")
        else:
            fig, mean_fig = plt.subplots(1, 1)
        fig.suptitle(f"mode : {mode}")
        mean_fig.set_title('mean values')
        mean_fig.grid()
        mean_fig.set_ylabel("bs (dB)")
        mean_fig.set_xlabel("angle (degree)")

        incidence_bs_curve, transmission_bs_curve = mean_model.model[mode]
        ds_i = incidence_bs_curve.ds
        mean_fig.plot(ds_i[BackscatterCurve.ANGLE], ds_i[BackscatterCurve.MEAN_BS], marker='.', color="blue")
        if display_count:
            count_fig.plot(ds_i[BackscatterCurve.ANGLE], ds_i[BackscatterCurve.VALUE_COUNT], marker='.', color="blue")

        if transmission_bs_curve:
            ds_t = transmission_bs_curve.ds
            dim_rx = ds_t[BackscatterCurve.MEAN_RESIDUAL_BS].shape[0]
            dim_tx = ds_t[BackscatterCurve.MEAN_RESIDUAL_BS].shape[1]
            for rx in range(dim_rx):
                for tx in range(dim_tx):
                    mean_fig.plot(ds_t[BackscatterCurve.ANGLE], ds_t[BackscatterCurve.MEAN_RESIDUAL_BS][rx][tx], marker='.', color="red")
                    mean_fig.plot(ds_t[BackscatterCurve.ANGLE], ds_t[BackscatterCurve.MEAN_BS][rx][tx], marker='.', color="orange")

        # if curve_per_file is not None:
        #     for sub_curve in curve_per_file.get(mode):
        #         ds = sub_curve.ds
        #         label = sub_curve.origin
        #         label = os.path.basename(label)
        #         mean_fig.plot(ds[BackscatterCurve.ANGLE], ds[BackscatterCurve.MEAN_BS], linestyle=':',
        #                       label=label)
        #         if display_count:
        #             count_fig.plot(ds[BackscatterCurve.ANGLE], ds[BackscatterCurve.VALUE_COUNT],
        #                            label=label)
        mean_fig.legend()
