# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import string
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.dates as mdates

from geospacelab.datahub import DataHub
import geospacelab.visualization.mpl.panels as mpl_panel
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pydatetime as dttool
from geospacelab.visualization.mpl._base import DashboardBase
import geospacelab.visualization.mpl.panels as panels

plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'book'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12


class Dashboard(DataHub, DashboardBase):
    _default_layout_config = {
        'left': 0.15,
        'right': 0.78,
        'bottom': 0.15,
        'top': 0.88,
        'hspace': 0.1,
        'wspace': 0.1
    }

    def __init__(self, figure=None, figure_config=None, **kwargs):
        super(Dashboard, self).__init__(figure=figure, figure_config=figure_config, **kwargs)


class TSDashboard(Dashboard):
    _default_figure_config = {
        'figsize': (10, 8),
    }

    def __init__(
            self,
            figure=None, figure_config=None,
            time_gap=True,
            timeline_major='UT',
            timeline_multiple_labels=None,
            **kwargs
    ):
        self.panel_layouts = []
        self.plot_styles = None
        self._panels_configs = {}

        figure_config = self._default_figure_config if figure_config is None else figure_config

        self.time_gap = time_gap
        self.timeline_major = timeline_major
        self.timeline_extra_labels = [] if timeline_multiple_labels is None else timeline_major # e.g., ['MLT',
        kwargs.update(visual='on')

        super(Dashboard, self).__init__(figure=figure, figure_config=figure_config, **kwargs)

        self._xlim = [self.dt_fr, self.dt_to]

    def set_layout(self, panel_layouts=None, panels_classes=None, plot_styles=None, row_height_scales=1,
                   left=None, right=None, bottom=None, top=None, hspace=None, add_panels=True, **kwargs):

        if left is None:
            left = self._default_layout_config['left']
        if right is None:
            right = self._default_layout_config['right']
        if bottom is None:
            bottom = self._default_layout_config['bottom']
        if top is None:
            top = self._default_layout_config['top']
        if hspace is None:
            hspace = self._default_layout_config['hspace']

        num_rows = len(panel_layouts)
        self.panel_layouts = panel_layouts
        if type(plot_styles) is not list:
            self.plot_styles = [None] * num_rows
        elif len(plot_styles) != num_rows:
            raise ValueError

        if panels_classes is None:
            panels_classes = [panels.TSPanel] * num_rows
        elif issubclass(panels_classes, panels.TSPanel):
            panels_classes = [panels_classes] * num_rows
        elif len(panels_classes) != num_rows:
            raise ValueError
        else:
            raise AttributeError

        if type(row_height_scales) is not list:
            if type(row_height_scales) == int:
                row_height_scales = [row_height_scales] * num_rows
            else:
                raise TypeError
        elif len(row_height_scales) != num_rows:
            raise ValueError

        hspace = hspace * row_height_scales[0]
        gs_num_rows = sum(row_height_scales)
        gs_num_cols = 1
        super().set_layout(num_rows=gs_num_rows, num_cols=gs_num_cols, left=left, right=right, bottom=bottom, top=top,
                           hspace=hspace, **kwargs)
        rec = 0
        for ind, height in enumerate(row_height_scales):
            self._panels_configs[ind] = {
                'row_ind': [rec, rec+height],
                'col_ind': [0, 1],
                'panel_class': panels_classes[ind],
            }
            rec = rec + height
        if add_panels:
            self.add_panels()

    def add_panels(self):
        bottom_panel = False
        for ind, panel_config in self._panels_configs.items():
            # share x
            if ind > 0:
                panel_config.update(sharex=self.panels[1]())
            if ind == len(self._panels_configs.keys())-1:
                bottom_panel = True
            else:
                bottom_panel = False
            panel_config.update(bottom_panel=bottom_panel)
            self.add_panel(**panel_config)

    def draw(self, dt_fr=None, dt_to=None, auto_grid=True):
        if dt_fr is not None:
            self._xlim[0] = dt_fr
        if dt_to is not None:
            self._xlim[1] = dt_to

        npanel = 0
        for ind, panel in self.panels.items():
            panel._xlim = self._xlim
            plot_layout = self.panel_layouts[npanel]
            plot_layout = self._validate_plot_layout(plot_layout)
            panel.draw(plot_layout)
            npanel = npanel + 1

        if auto_grid:
            self.add_grid(panel_id=0, visible=True, which='major', axis='both', lw=0.5, color='grey', alpha=0.3)
            self.add_grid(panel_id=0, visible=True, which='minor', axis='both', lw=0.3, color='grey', alpha=0.1)

    def add_grid(self, panel_id=0, visible=None, which='major', axis='both', **kwargs):
        if panel_id == 0:
            panels = self.panels.values()
        elif type(panel_id) is int:
            panels = [self.panels[int]]
        elif type(panel_id) is list:
            panels = [self.panels[i] for i in panel_id]
        else:
            raise NotImplementedError

        for p in panels:
            p.add_grid(ax=p(), visible=visible, which=which, axis=axis, **kwargs)

    def _validate_plot_layout(self, layout_in, level=0):
        from geospacelab.datahub import VariableModel
        if (level == 0) and (not isinstance(layout_in, list)):
            raise TypeError("The plot layout must be a list!")
        type_layout_in = type(layout_in)
        layout_out = []
        for ind, elem in enumerate(layout_in):
            if isinstance(elem, list):
                layout_out.append(self._validate_plot_layout(elem, level=level+1))
            elif issubclass(elem.__class__, VariableModel):
                var = elem
                layout_out.append(var)
            elif isinstance(elem, int):
                index = elem
                var = self.variables[index]
                layout_out.append(var)
            else:
                raise TypeError
        return type_layout_in(layout_out)

    def save_figure(self, file_dir=None, file_name=None, append_time=True, dpi=300, file_format='png', **kwargs):
        if file_dir is None:
            file_dir = pathlib.Path().cwd()
        else:
            file_dir = pathlib.Path(file_dir)
        if type(file_name) is not str:
            raise ValueError

        if append_time:
            dt_range_str = self.get_dt_range_str(style='filename')
            file_name = '_'.join([file_name, dt_range_str])

        file_name = file_name + '.' + file_format

        plt.savefig(file_dir / file_name, dpi=dpi, format=file_format, **kwargs)

    @staticmethod
    def show():
        plt.show()

    def add_title(self, x=0.5, y=1.08, title=None, **kwargs):
        append_time = kwargs.pop('append_time', True)
        kwargs.setdefault('fontsize', plt.rcParams['figure.titlesize'])
        kwargs.setdefault('fontweight', 'roman')
        if append_time:
            dt_range_str = self.get_dt_range_str(style='title')
            title = title + ', ' + dt_range_str
        title = title.replace(', , ', ', ')
        super().add_title(x=x, y=y, title=title, **kwargs)

    def get_dt_range_str(self, style='title'):
        dt_fr = self._xlim[0]
        dt_to = self._xlim[1]
        if style == 'title':
            diff_days = dttool.get_diff_days(dt1=dt_fr, dt2=dt_to)
            if diff_days == 0:
                fmt1 = "%Y-%m-%dT%H:%M:%S"
                fmt2 = "%H:%M:%S"
            else:
                fmt1 = "%Y-%m-%dT%H:%M:%S"
                fmt2 = fmt1
            dt_range_str = dt_fr.strftime(fmt1) + ' - ' + dt_to.strftime(fmt2)
        elif style == 'filename':
            diff_days = dttool.get_diff_days(dt1=dt_fr, dt2=dt_to)
            if diff_days == 0:
                fmt1 = "%Y%m%d-%H%M%S"
                fmt2 = "%H%M%S"
            else:
                fmt1 = "%Y%m%d-%H%M%S"
                fmt2 = fmt1
            dt_range_str = dt_fr.strftime(fmt1) + '-' + dt_to.strftime(fmt2)
        return dt_range_str

    def add_vertical_line(self, dt_in, panel_index=0,
                          label=None, label_position=None, top_extend=0., bottom_extend=0., **kwargs):
        if type(dt_in) is not datetime.datetime:
            return

        if label_position is None:
            label_position = 'top'
        text_config = kwargs.pop('text_config', {})
        kwargs.setdefault('linewidth', 2)
        kwargs.setdefault('linestyle', '--')
        kwargs.setdefault('color', 'k')
        kwargs.setdefault('alpha', 0.8)

        if panel_index == 0:
            if 'major' not in self.extra_axes.keys():
                ax = self.add_major_axes()
            else:
                ax = self.extra_axes['major']
        else:
            ax = self.panels[panel_index].axes['major']

        xlim = self.panels[1].axes['major'].get_xlim()
        ylim = ax.get_ylim()
        diff_xlim = xlim[1] - xlim[0]
        diff_ylim = ylim[1] - ylim[0]
        x = mdates.date2num(dt_in)
        x = (x - xlim[0]) / diff_xlim

        y0 = 0 - bottom_extend
        y1 = 1. + top_extend
        line = mpl.lines.Line2D([x, x], [y0, y1], transform=ax.transAxes, **kwargs)
        line.set_clip_on(False)
        ax.add_line(line)

        if type(label) is str:
            if label_position == 'top':
                y = y1
                text_config.setdefault('va', 'bottom')
            elif label_position == 'bottom':
                y = y0
                text_config.setdefault('va', 'top')
            else:
                x = label_position[0]
                y = label_position[1]
                text_config.setdefault('va', 'center')
            text_config.setdefault('ha', 'center')
            text_config.setdefault('fontsize', plt.rcParams['axes.labelsize'])
            text_config.setdefault('fontweight', 'medium')
            text_config.setdefault('clip_on', False)
            ax.text(x, y, label, transform=ax.transAxes, **text_config)

    def add_shading(self, dt_fr, dt_to, panel_index=0,
                    label=None, label_position=None, top_extend=0., bottom_extend=0., **kwargs):
        if type(dt_fr) is not datetime.datetime:
            return

        if label_position is None:
            label_position = 'top'
        text_config = kwargs.pop('text_config', {})
        kwargs.setdefault('edgecolor', 'none')
        kwargs.setdefault('facecolor', 'yellow')
        kwargs.setdefault('alpha', 0.4)

        if panel_index == 0:
            if 'major' not in self.extra_axes.keys():
                ax = self.add_major_axes()
            else:
                ax = self.extra_axes['major']
        else:
            ax = self.panels[panel_index].axes['major']

        xlim = self.panels[1].axes['major'].get_xlim()
        ylim = ax.get_ylim()
        diff_xlim = xlim[1] - xlim[0]
        diff_ylim = ylim[1] - ylim[0]
        x0 = mdates.date2num(dt_fr)
        x0 = (x0 - xlim[0]) / diff_xlim
        x1 = mdates.date2num(dt_to)
        x1 = (x1 - xlim[0]) / diff_xlim

        y0 = 0 - bottom_extend
        y1 = 1. + top_extend

        rectangle = mpl.patches.Rectangle((x0, y0), x1-x0, y1-y0, transform=ax.transAxes, **kwargs)
        rectangle.set_clip_on(False)
        ax.add_patch(rectangle)

        if type(label) is str:
            x = (x0 + x1) / 2
            if label_position == 'top':
                y = y1
                text_config.setdefault('va', 'bottom')
            elif label_position == 'bottom':
                y = y0
                text_config.setdefault('va', 'top')
            else:
                x = label_position[0]
                y = label_position[1]
                text_config.setdefault('va', 'center')
            text_config.setdefault('ha', 'center')
            text_config.setdefault('fontsize', plt.rcParams['axes.labelsize'])
            text_config.setdefault('fontweight', 'medium')
            text_config.setdefault('clip_on', False)
            ax.text(x, y, label, transform=ax.transAxes, **text_config)

    def add_top_bar(self, dt_fr, dt_to, bottom=0, top=0.02, **kwargs):
        bottom_extend = -1. - bottom
        top_extend = top
        kwargs.setdefault('alpha', 1.)
        kwargs.setdefault('facecolor', 'orange')
        self.add_shading(dt_fr, dt_to, bottom_extend=bottom_extend, top_extend=top_extend, **kwargs)

