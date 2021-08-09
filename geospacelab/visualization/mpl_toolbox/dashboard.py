import string
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy

# import geospacelab.visualization.mpl_toolbox.figure as mpl_figure
import geospacelab.visualization.mpl_toolbox.panel as mpl_panel
import geospacelab.toolbox.utilities.pybasic as basic

default_layout_config = {
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
        self.axes = {}

        self.gs = None

        super().__init__(**kwargs)

    def set_layout(self, num_rows=None, num_cols=None, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_layout_config)
        self.gs = self.figure.add_gridspec(num_rows, num_cols)
        self.gs.update(**kwargs)

    def add_dashboard_line(self, x=None, y=None, panel_index=None, **kwargs):
        pass

    def add_patches(self, x=None, y=None, panel_index=None, **kwargs):
        pass

    def add_top_horizontal_bar(self, x=None, y=None, panel_index=None, **kwargs):
        pass

    def add_arrows(self, **kwargs):
        pass

    def add_panel(self, row_ind=None, col_ind=None, index=None, label=None, panel_class=None, **kwargs):
        if isinstance(row_ind, int):
            row_ind = [row_ind, row_ind+1]
        if isinstance(col_ind, int):
            col_ind = [col_ind, col_ind+1]
        if panel_class is None:
            panel_class = mpl_panel.Panel
        elif not issubclass(panel_class, mpl_panel.Panel):
            raise TypeError

        panel = panel_class(label=label, **kwargs)
        panel.add_subplot(self.gs[row_ind[0]:row_ind[1], col_ind[0]:col_ind[1]], major=True, **kwargs)

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

    def add_text(self, x=None, y=None, text=None, **kwargs):
        # add text in dashboard cs
        kwargs.setdefault('fontsize', default_dashboard_fontsize)
        kwargs.setdefault('ha', 'center')
        kwargs.setdefault('va', 'bottom')
        if x is None:
            x = 0.5
        if y is None:
            y = 1.05

        # set in dashboard cs
        x_new = self.gs.left + x * (self.gs.right - self.gs.left)

        y_new = self.gs.bottom + y * (self.gs.top - self.gs.bottom)

        self.figure.text(x_new, y_new, text, **kwargs)

    def add_panel_labels(self, panel_indices=None, style='alphabets', **kwargs):
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
            x = 0.02
            y = 0.9

        kwargs.setdefault('ha', 'left')     # horizontal alignment
        kwargs.setdefault('va', 'center') # vertical alignment

        pos_0 = self.panels[1].axes['major'].get_position()  # adjust y in case of different gs_row_heights
        for ind, p_index in enumerate(panel_indices):
            panel = self.panels[p_index]
            pos_1 = panel.axes['major'].get_position()
            if panel.label is None:
                label = "({})".format(label_list[ind])
            else:
                label = panel.label
            kwargs.setdefault('fontsize', plt.rcParams['axes.labelsize'])
            kwargs.setdefault('fontweight', 'book')
            bbox_config = {'facecolor': 'yellow', 'alpha': 0.3, 'edgecolor': 'none'}
            kwargs.setdefault('bbox', bbox_config)
            y_new = 1 - pos_0.height/pos_1.height + y * pos_0.height / pos_1.height
            panel.add_label(x, y_new, label, **kwargs)

    def add_axes(self, rect, label=None, **kwargs):
        if label is None:
            label = len(self.axes.keys())
            label_str = 'ax_' + str(label)
        else:
            label_str = label
        kwargs.setdefault('facecolor', 'none')
        ax = self.figure.add_axes(rect, label=label_str, **kwargs)
        self.axes[label] = ax
        return ax

    def add_major_axes(self, rect=None):
        if rect is None:
            x = self.gs.left
            y = self.gs.bottom
            w = self.gs.right - self.gs.left
            h = self.gs.top - self.gs.bottom
            rect = [x, y, w, h]
        ax = self.add_axes(rect, label='major', facecolor='none')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        return ax

    

    def add_shadings(self):
        pass



