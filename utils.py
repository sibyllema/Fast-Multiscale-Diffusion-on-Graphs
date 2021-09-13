import numpy as np
import matplotlib.pyplot as plt

def plot_fancy_error_bar(x, y, ax=None, type="median_quartiles", **kwargs):
    """ Plot data with errorbars and semi-transparent error region.

    Arguments:
    x -- list or ndarray, shape (nx,)
        x-axis data
    y -- ndarray, shape (nx,ny)
        y-axis data. Usually represents ny attempts for each datum in x.
    ax -- matplotlib Axis
        Axis to plot the data on
    type -- string.
        Type of error. Either "median_quartiles" or "average_std".
    kwargs -- dict
        Extra options for matplotlib (such as color, label, etc).
    """
    if type=="median_quartiles":
        y_center    = np.percentile(y, q=50, axis=-1)
        y_up        = np.percentile(y, q=25, axis=-1)
        y_down      = np.percentile(y, q=75, axis=-1)
    elif type=="average_std":
        y_center    = np.average(y, axis=-1)
        y_std       = np.std(y, axis=-1)
        y_up        = y_center + y_std
        y_down      = y_center - y_std

    fill_color = kwargs["color"] if "color" in kwargs else None

    if ax is None:
        plot_ = plt.errorbar(x, y_center, (y_center - y_down, y_up - y_center), **kwargs)
        plt.fill_between(x, y_down, y_up, alpha=.3, color=fill_color)
    else:
        plot_ = ax.errorbar(x, y_center, (y_center - y_down, y_up - y_center), **kwargs)
        ax.fill_between(x, y_down, y_up, alpha=.3, color=fill_color)
    return plot_
