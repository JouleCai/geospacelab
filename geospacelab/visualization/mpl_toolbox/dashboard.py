import string

import matplotlib.pyplot as plt
import numpy

import geospacelab.visualization.mpl_toolbox.figure as mpl_figure
import geospacelab.visualization.mpl_toolbox.panel as mpl_panel
import geospacelab.toolbox.utilities.pybasic as basic

default_gs_config = {
        'left':     0.15,
        'right':    0.8,
        'bottom':   0.15,
        'top':      0.88,
        'hspace':   0.0,
        'wspace':   0.1
}
default_dashboard_fontsize = 12


class Dashboard(object):

    def __init__(self, **kwargs):
        new_figure = kwargs.pop('new_figure', False)
        figure_config = kwargs.pop('figure_config', {})
        if new_figure:
            figure = plt.figure(**figure_config)
        else:
            figure = kwargs.pop('figure', plt.gcf())
        self.figure = figure
        self.panels = {}
        self.title = kwargs.pop('title', None)
        self.label = kwargs.pop('label', None)

        self.gs = None

        super().__init__(**kwargs)

    def set_gridspec(self, num_rows, num_cols, **kwargs):
        basic.dict_set_default(kwargs, **default_gs_config)
        self.gs = self.figure.add_gridspec(num_rows, num_cols)
        self.gs.update(**kwargs)

    def add_panel(self, row_ind=None, col_ind=None, index=None, label=None, plot_type=None, **kwargs):
        if isinstance(row_ind, int):
            row_ind = [row_ind, row_ind+1]
        if isinstance(col_ind, int):
            col_ind = [col_ind, col_ind+1]

        ax = self.figure.add_subplot(self.gs[row_ind[0]:row_ind[1], col_ind[0]:col_ind[1]], **kwargs)
        panel = mpl_panel.Panel(label=label)
        panel.axes['major'] = ax

        if index is None:
            index = len(self.panels.keys()) + 1
        elif index in self.panels.keys():
            raise ValueError('The panel index has been occupied. Change to a new one!')
        self.panels[index] = panel
        return index

    def remove_panel(self, index):
        self.panels[index].remove()
        del self.panels[index]

    def replace_panel(self, index, **kwargs):
        position = self.panels[index].get_position()
        self.remove_panel(index)

        panel = self.figure.add_subplot(**kwargs)
        panel.set_position(position)
        self.panels[index] = panel

    def add_title(self, x=None, y=None, title=None, **kwargs):
        if title is not None:
            self.title = title

        kwargs.setdefault('fontsize', default_dashboard_fontsize)
        kwargs.setdefault('horizontalalignment', 'center')
        kwargs.setdefault('verticalalignment', 'bottom')
        if x is None:
            x = self.gs.left + (self.gs.right - self.gs.left) / 2
        if y is None:
            y = self.gs.top + (1 - self.gs.top) / 10
        self.figure.text(x, y, self.title, **kwargs)

    def show_panel_labels(self, panel_indices=None, style='alphabets', **kwargs):
        if panel_indices is None:
            panel_indices = self.panels.keys()

        if style == 'alphabets':
            label_list = string.ascii_lowercase
        else:
            raise NotImplemented

        if 'position' in kwargs.keys():
            pos = kwargs['position']
            x = pos[0]
            y = pos[1]
        else:
            raise KeyError

        kwargs.setdefault('ha', 'left')     # horizontal alignment
        kwargs.setdefault('va', 'center') # vertical alignment

        for ind, p_index in enumerate(panel_indices):
            panel = self.panels[p_index]
            if panel.label is None:
                label = "({})".format(label_list[ind])
            else:
                label = panel.label
            panel.label(x, y, label, **kwargs)

    def add_vertical_lines(self):
        pass

    def add_shadings(self):
        pass



