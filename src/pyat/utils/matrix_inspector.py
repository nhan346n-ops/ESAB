import matplotlib.pyplot as plt
import numpy as np

import pyat.utils.tiff_driver as tiff


def display_signal(
    values: np.ndarray,
    title: str,
    display_immediate=False,
    x_axis: np.ndarray = None,
):
    # ref.display_matrix(values,title=title,show=show)
    f = plt.figure(title, figsize=(5, 5))
    if x_axis is None:
        plt.plot(values, marker="+")
    else:
        plt.plot(x_axis, values, marker="+")
    if title is not None:
        f.suptitle(title)
    plt.tight_layout()
    if display_immediate:
        plt.show()


def display_matrix(values: np.ndarray, title: str = "Unkown", display_immediate=False, bins=100):
    # ref.display_matrix(values,title=title,show=show,bins=bins)
    """
    Display a numpy matrix and statistics about its content
    """
    # disable import error for test environment
    # pylint: disable=E0401
    try:
        # pylint: disable=C0415
        from IPython import get_ipython

        on_notebook = get_ipython() is not None  # True if we run in a notebook or IPython was imported
    except ModuleNotFoundError:
        on_notebook = False

    f = plt.figure(title, figsize=(15, 5))
    if on_notebook is True:
        ax = plt.subplot(1, 2, 1)
        c = ax.imshow(values, aspect="auto")
        ax.set_xlabel("Data")
        f.colorbar(c, ax=ax)
        ax = plt.subplot(1, 2, 2)
        ax.hist(values.flatten(), bins=bins, histtype="step")
        ax.set_xlabel("Histogram over " + str(bins) + " values")
        print("Mean : ", np.nanmean(values))
        print("Max : ", np.nanmax(values))
        print("Min : ", np.nanmin(values))
        print("Median : ", np.nanmedian(values))
        print("Std : ", np.nanstd(values))
        print(title.center(100))
        if display_immediate:
            plt.show()
    else:
        ax = plt.subplot(1, 3, 1)
        c = ax.imshow(values, aspect="auto")
        ax.set_xlabel("Data")
        f.colorbar(c, ax=ax)
        ax = plt.subplot(1, 3, 2)
        ax.hist(values.flatten(), bins=255, histtype="step")
        ax.set_xlabel("Histogram")
        ax = plt.subplot(1, 3, 3)
        mean = np.nanmean(values)
        lmax = np.nanmax(values)
        lmin = np.nanmin(values)
        median = np.nanmedian(values)
        std = np.nanstd(values)
        t = (
            "mean:"
            + str(mean)
            + "\n"
            + "max:"
            + str(lmax)
            + "\n"
            + "min:"
            + str(lmin)
            + "\n"
            + "median:"
            + str(median)
            + "\n"
            + "std:"
            + str(std)
        )
        ax.set_xlabel("Statistics")
        ax.text(0.1, 0.5, t)
        f.suptitle(title)
        plt.tight_layout()
        if display_immediate:
            plt.show()


def display_geotiff_data(file_name: str, title: str, display_immediate=True):
    """
    Display a geotiff and statistics about its content
    do not take into account for projection and offset, only display data content
    """
    tiff_values = np.ma.filled(tiff.read_tiff(file_name), np.nan)
    # I have had that to delete wrong values from sonarscope
    # tiff_values[tiff_values == 182] = np.nan
    display_matrix(tiff_values, title, display_immediate)


def show():
    plt.show()
