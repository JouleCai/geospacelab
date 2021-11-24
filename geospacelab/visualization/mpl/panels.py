# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import matplotlib.pyplot as plt

from geospacelab.visualization.mpl._base import Panel as PanelBase
from geospacelab.datahub.variable_base import VariableModel


def check_panel_ax(func):
    def wrapper(*args, **kwargs):
        obj = args[0]
        kwargs.setdefault('ax', None)
        if kwargs['ax'] is None:
            kwargs['ax'] = obj.axes['major']
        result = func(*args, **kwargs)
        return result
    return wrapper


class Panel(PanelBase):

    def __init__(self, *args, figure=None, from_subplot=True, **kwargs):
        super(Panel, self).__init__(*args, figure=figure, from_subplot=from_subplot, **kwargs)

class TSPanel(Panel):
    def __init__(self, *args, plot_layout=None, figure=None, from_subplot=True, **kwargs):
        super(Panel, self).__init__(*args, figure=figure, from_subplot=from_subplot, **kwargs)
        self._var_for_config = None
        if plot_layout is not None:
            self.overlay_from_variables(plot_layout)

    def overlay_from_variables(self, layout_in, ax=None, level=0, num_y_axis=0):
        if ax is None:
            ax = self()
        self.sca(ax)
        if level == 0 and not list(layout_in):
            raise ValueError

        if level > 1:
            raise NotImplemented

        for ind, elem in enumerate(layout_in):
            if isinstance(elem, list):
                if ind > 0:
                    num_y_axis = num_y_axis + 1
                    ax_in = self.add_twin_axes(
                        ax=ax, which='y', location='right', offset_type='outward', offset=40*(num_y_axis-1)
                    )
                else:
                    ax_in = ax
                self.overlay_from_variables(elem, ax=ax_in, level=level+1, num_y_axis=num_y_axis)
            elif issubclass(elem.__class__, VariableModel):
                if ind == 0:
                    setattr(ax, '_var_for_config', elem)
                self._overlay_variable(elem, ax=ax)
            elif type(elem) is tuple:
                self._overlay_variable(elem, ax=ax)
            else:
                raise NotImplementedError

    def _overlay_variable(self, elem, ax=None):
        plot_style = elem.visual.plot_config.style

        if plot_style == '1P':
            self.overlay_plot(elem, ax=ax)
        elif plot_style in ['1', '1E']:
            self.overlay_errorbar(elem, ax=ax)
        elif plot_style in ['2P']:
            self.overlay_pcolormesh(elem, ax=ax)
        elif plot_style in ['2I']:
            self.overlay_imshow(elem, ax=ax)
        elif plot_style in ['1B']:
            self.overlay_bar(elem, ax=ax)
        elif plot_style in ['1F0']:
            self.overlay_fill_between_y_zero(elem, ax=ax)
        elif plot_style in ['1S']:
            self.overlay_scatter(elem, ax=ax)
        elif plot_style in ['1C']:
            self.overlay_multiple_colored_line(elem, ax=ax)
        else:
            raise NotImplementedError

    def overlay_plot(self, *args, ax=None, **kwargs):
        var = args[0]

        super().overlay_plot()







class Panel1(object):
    def __init__(self, *args, figure=None, **kwargs):
        if figure is None:
            figure = plt.gcf()
        self.figure = figure
        self.axes = {}
        self.label = kwargs.pop('label', None)
        # self.objectives = kwargs.pop('objectives', {})
        ax = self.figure.add_subplot(*args, **kwargs)
        self.axes['major'] = ax

    # def add_subplot(self, *args, major=False, label=None, **kwargs):
    #     if major:
    #         label = 'major'
    #     else:
    #         if label is None:
    #             label = len(self.axes.keys())
    #
    #     ax = self.figure.add_subplot(*args, **kwargs)
    #     self.axes[label] = ax
    #     return ax

    def add_axes(self, *args, major=False, label=None, **kwargs):
        if major:
            label = 'major'
        else:
            if label is None:
                label = len(self.axes.keys())

        ax = self.figure.add_axes(*args, **kwargs)
        ax.patch.set_alpha(0)
        self.axes[label] = ax
        return ax

    @property
    def major_ax(self):
        return self.axes['major']

    @major_ax.setter
    def major_ax(self, ax):
        self.axes['major'] = ax

    @check_panel_ax
    def plot(self, *args, ax=None, **kwargs):
        # plot_type "1P"
        ipl = ax.plot(*args, **kwargs)
        return ipl

    @check_panel_ax
    def errorbar(self, *args, ax=None, **kwargs):
        # plot_type = "1E"
        ieb = ax.errorbar(*args, **kwargs)
        return ieb

    @check_panel_ax
    def pcolormesh(self, *args, ax=None, **kwargs):
        # plot_type: "2P"
        ipm = ax.pcolormesh(*args, **kwargs)
        return ipm

    @check_panel_ax
    def imshow(self, *args, ax=None, **kwargs):
        # plot_type = "2I"
        im = ax.imshow(*args, **kwargs)
        return im

    def add_label(self, x, y, label, ax=None, ha='left', va='center', **kwargs):
        if ax is None:
            ax = self.axes['major']
        if label is None:
            label = ''
        transform = kwargs.pop('transform', ax.transAxes)
        ax.text(x, y, label, transform=transform, ha=ha, va=va, **kwargs)

    def add_title(self, *args, **kwargs):
        self.axes['major'].set_title(*args, **kwargs)