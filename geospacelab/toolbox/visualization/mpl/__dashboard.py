import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy

import utilities.basic_utilities as basic
import graphtoolbox.axes_utilities as axtool

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

        self.figure = kwargs.pop('figure', plt.gcf())
        self.nrows = kwargs.pop('nrows', None)
        self.ncols = kwargs.pop('ncols', None)
        self.panels = {}
        self.title = kwargs.pop('title', None)
        self.label = kwargs.pop('label', None)

        self.gs = self.figure.add_gridspec(self.nrows, self.ncols)
        self.gs.update(**kwargs.pop('gs_config', default_gs_config))

    def add_panel(self, row_ind=None, col_ind=None, label=None, kind='base', **kwargs):
        if isinstance(row_ind, int):
            row_ind = [row_ind, row_ind+1]
        if isinstance(col_ind, int):
            col_ind = [col_ind, col_ind+1]

        axes_config = kwargs.pop('axes_config', {})
        panel = self.figure.add_subplot(self.gs[row_ind[0]:row_ind[1], col_ind[0]:col_ind[1]], **axes_config)
        setattr(panel, 'label', label)
        setattr(panel, 'kind', kind)
        setattr(panel, 'objects', {})

        self.panels[label] = panel

    def add_panels(self, row_inds=None, col_inds=None, panel_max_num=None, axis=0, **kwargs):
        # axis: 0 - along rows first, 1 - along cols first
        if panel_max_num is None:
            panel_max_num = self.nrows * self.ncols
        if row_inds is None:
            row_inds = range(panel_max_num)
        if col_inds is None:
            col_inds = [0] * panel_max_num

        if axis == 0:
            m = self.ncols
            n = self.nrows
        elif axis == 1:
            m = self.nrows
            n = self.ncols

        nax = 0
        for i in range(m):
            for j in range(n):
                nax = nax + 1
                if nax > panel_max_num:
                    continue
                row_ind = row_inds[nax-1]
                col_ind = col_inds[nax-1]
                self.add_subplot(row_ind, col_ind, **kwargs)

    def remove_panel(self, label):
        self.panels[label].remove()
        del self.panels[label]

    def replace_panel(self, label, **kwargs):
        position = self.panels[label].get_position()
        self.remove_panel(label)

        panel = self.figure.add_subplot(**kwargs)
        panel.set_position(position)
        self.panels[label] = panel

    def add_title(self, x=None, y=None, title=None, **kwargs):
        if title is not None:
            self.title = title

        kwargs.setdefault('fontsize', default_dashboard_fontsize)
        kwargs.setdefault('horizontalalignment', 'center')
        kwargs.setdefault('verticalalignment', 'bottom')
        if x is None:
            x = self.gs.left + (self.gs.right - self.gs.left) / 2
        if y  is None:
            y = self.gs.top + (1 - self.gs.top) / 10
        plt.gcf().text(x, y, self.title, **kwargs)

