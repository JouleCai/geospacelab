import matplotlib.pyplot as plt
import geospacelab.toolbox.utilities.pybasic as basic
import numpy as np
import copy


def scatter(*args, **kwargs):
    pass


def plot_with_errorbar(x=None, y=None, xerr=None, yerr=None, ax=None, **kwargs):

    default_plot_config = {
        'linestyle': '-',
        'linewidth': 1.5,
        'marker': '.',
        'markersize': 1,
    }
    default_errorbar_config = copy.deepcopy(default_plot_config)
    default_errorbar_config.update({'elinewidth': 0.5})

    kwargs.setdefault('plot_config', default_plot_config)
    basic.dict_set_default(kwargs['plot_config'], **default_plot_config)
    kwargs.setdefault('errorbar_config', default_errorbar_config)
    basic.dict_set_default(kwargs['errorbar_config'], **default_errorbar_config)
    kwargs.setdefault('legend', 'off')
    if ax is None:
        ax = plt.gca()

    if isinstance(x, np.ndarray):
        x = [x]
    if isinstance(y, np.ndarray):
        y = [y]
    if isinstance(xerr, np.ndarray):
        xerr = [xerr]
    if isinstance(yerr, np.ndarray):
        yerr = [yerr]


    nlines = len(x)
    if (x is None) and (y is None):
        return
    if x is None:
        x = [None] * nlines
    if y is None:
        y = [None] * nlines
    if xerr is None:
        xerr = [None] * nlines
    if yerr is None:
        yerr = [None] * nlines
    hls = []
    # making plots
    for ind in range(nlines):
        x1 = x[ind]
        y1 = y[ind]
        xerr1 = xerr[ind]
        yerr1 = yerr[ind]
        if x1 is None:
            x1 = np.array(range(y[ind].shape[0]))
        elif y1 is None:
            y1 = np.array(range(x[ind].shape[0]))
        if (xerr1 is None) and (yerr1 is None):
            hl = ax.plot(x1, y1, **kwargs['kwargs_plot'])
        else:
            hl = ax.errorbar(x1, y1, xerr=xerr1, yerr=yerr1, **kwargs['kwargs_errorbar'])
        hls.extend(hl)

    return hls


def errorbar(self, **kwargs):
    pass






