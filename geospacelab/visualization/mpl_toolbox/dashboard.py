import string

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy


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
        self.gs_num_rows = kwargs.pop('gs_num_rows', None)
        self.gs_num_cols = kwargs.pop('gs_num_cols', None)
        self.panels = {}
        self.title = kwargs.pop('title', None)
        self.label = kwargs.pop('label', None)

        self.gs = self.figure.add_gridspec(self.gs_num_rows, self.gs_num_cols)
        self.gs.update(**kwargs.pop('gs_config', default_gs_config))

    def add_panel(self, row_ind=None, col_ind=None, index=None, kind='base', **kwargs):
        if isinstance(row_ind, int):
            row_ind = [row_ind, row_ind+1]
        if isinstance(col_ind, int):
            col_ind = [col_ind, col_ind+1]

        axes_config = kwargs.pop('axes_config', {})
        panel = self.figure.add_subplot(self.gs[row_ind[0]:row_ind[1], col_ind[0]:col_ind[1]], **axes_config)
        setattr(panel, 'label', label)
        setattr(panel, 'kind', kind)
        setattr(panel, 'objects', {})

        if index is None:
            index = len(self.panels.keys()) + 1
        self.panels[index] = panel
        return index

    def add_panels(self, row_inds=None, col_inds=None, max_num_panels=None, axis=0, **kwargs):
        # axis: 0 - along rows first, 1 - along cols first
        if max_num_panels is None:
            max_num_panels = self.gs_num_rows * self.gs_num_cols
        if row_inds is None:
            row_inds = range(max_num_panels)
        if col_inds is None:
            col_inds = [0] * max_num_panels

        if axis == 0:
            m = self.gs_num_cols
            n = self.gs_num_rows
        elif axis == 1:
            m = self.gs_num_rows
            n = self.gs_num_cols

        nax = 0
        for i in range(m):
            for j in range(n):
                nax = nax + 1
                if nax > max_num_panels:
                    continue
                row_ind = row_inds[nax-1]
                col_ind = col_inds[nax-1]
                self.add_panel(row_ind, col_ind, **kwargs)

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

    def add_panel_labels(self, indices=None, style='alphabets', **kwargs):
        if indices is None:
            indices = range(len(self.panels.keys()))

        if style == 'alphabets':
            label_list = string.ascii_lowercase
        else:
            raise NotImplemented

        position = kwargs.pop('position', (0.1, 0.9))
        kwargs.setdefault('horizontalalignment', 'left')
        kwargs.setdefault('verticalalignment', 'center')

        for ind, p_index in enumerate(indices):
            ax = self.panels[p_index]
            label = "({})".format(label_list[p_index])
            ax.text(position[0], position[1], label,  transform=ax.transAxes, **kwargs)

    def add_vertical_lines(self):
        pass

    def add_shadings(self):
        pass



