# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import numpy as np
import re
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib.ticker as mpl_ticker
import matplotlib.dates as mpl_dates
from matplotlib import rcParams
from scipy.interpolate import interp1d
from cycler import cycler

from geospacelab.visualization.mpl.__base__ import PanelBase
from geospacelab.datahub.__variable_base__ import VariableBase as VariableModel
import geospacelab.toolbox.utilities.numpyarray as arraytool
import geospacelab.visualization.mpl.axis_ticks as ticktool
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.visualization.mpl._helpers import check_panel_ax
import geospacelab.visualization.mpl.colormaps as mycmap


class Panel(PanelBase):

    def __init__(self, *args, figure=None, from_subplot=True, **kwargs):
        super(Panel, self).__init__(*args, figure=figure, from_subplot=from_subplot, **kwargs)


class TSPanel(Panel):

    _default_plot_config = {
            'linestyle': '-',
            'linewidth': 1.5,
            'marker': '.',
            'markersize': 1
        }

    _default_legend_config = {
        'loc': 'upper left',
        'bbox_to_anchor': (1.0, 1.0),
        'frameon': False,
        'fontsize': 'small'
    }
    _default_xtick_params = {
        'labelsize': plt.rcParams['xtick.labelsize'],
        'direction': 'inout',
        'extral_labels_x_offset': None,
        'extral_labels_y_offset': None, 
    }
    _default_colorbar_offset = 0.1

    def __init__(
            self, *args, 
            dt_fr=None, dt_to=None, figure=None, from_subplot=True,
            bottom_panel=True, timeline_reverse=False, timeline_extra_labels=None,
            time_gap=True, time_res=None, timeline_same_format=False,
            **kwargs
    ):
        
        super(Panel, self).__init__(*args, figure=figure, from_subplot=from_subplot, **kwargs)

        self.dt_fr = dt_fr
        self.dt_to = dt_to

        self._xlim = [dt_fr, dt_to]
        self.timeline_reverse = timeline_reverse
        self.timeline_same_format=timeline_same_format
        self.time_gap = time_gap
        self.time_res = time_res
        if timeline_extra_labels is None:
            timeline_extra_labels = []
        self.timeline_extra_labels = timeline_extra_labels
        self._var_for_config = None
        self.bottom_panel = bottom_panel

    def draw(self, plot_layout, ax=None):
        self.overlay_from_variables(plot_layout, ax=ax)

    def overlay_from_variables(self, layout_in, ax=None, level=0):
        if ax is None:
            ax = self()
        self.sca(ax)

        if level == 0 and not list(layout_in):
            raise TypeError

        if level == 0:
            if type(layout_in[0]) is list:
                raise ValueError("The first element in the plot layout cannot be a list!")
            if any(type(i) is list for i in layout_in):
                shared_cycler = cycler(color=plt.rcParams["axes.prop_cycle"])
                self.axes_overview[ax]['twinx'] = 'on'
                self.axes_overview[ax]['shared']['prop_cycle_iter'] = shared_cycler.__iter__()
                ax.set_prop_cycle(color=[next(self.axes_overview[ax]['shared']['prop_cycle_iter'])['color']])
            if len(layout_in) == 1:
                self.axes_overview[ax]['legend'] = 'off'

        if level > 1:
            raise NotImplemented

        iplts = []

        for ind, elem in enumerate(layout_in):

            if ind == 0 or issubclass(elem.__class__, VariableModel):
                var = elem[0] if type(elem) is list else elem
                iplts_add = self.overlay_a_variable(var, ax=ax)
            elif level == 0 and isinstance(elem, list):
                ax_in = self.add_twin_axes(
                    ax=ax, which='x', location='right', offset_type='outward',
                    offset=50*(len(self.axes_overview[ax]['twinx_axes']))
                )
                self.axes_overview[ax_in]['twinx'] = 'self'
                self.axes_overview[ax_in]['shared']['prop_cycle_iter'] = \
                    self.axes_overview[ax]['shared']['prop_cycle_iter']
                ax_in.set_prop_cycle(color=[next(self.axes_overview[ax_in]['shared']['prop_cycle_iter'])['color']])
                # ax_in._get_lines.prop_cycler = ax._get_lines.prop_cycler
                iplts_add = self.overlay_from_variables(elem, ax=ax_in, level=level+1)
            else:
                raise NotImplementedError
            iplts.extend(iplts_add)

        if level == 0:
            if self.axes_overview[ax]['legend'] == 'on' and self.axes_overview[ax]['twinx'] == 'off':
                self._check_legend(ax)
            if self.axes_overview[ax]['twinx'] == 'on':
                self._check_twinx(ax)
            if self.axes_overview[ax]['colorbar'] == 'on':
                self._check_colorbar(ax)
            self._set_xaxis(ax=ax)

        self._set_yaxis(ax=ax)
        return iplts

    def overlay_a_variable(self, var, ax=None):
        self.axes_overview[ax]['variables'].extend([var])
        var_for_config = self.axes_overview[ax]['variables'][0]
        plot_style = var_for_config.visual.plot_config.style

        if plot_style in ['1P', '1noE']:
            iplt = self.overlay_line(var, ax=ax)
        elif plot_style in ['1', '1E']:
            iplt = self.overlay_line(var, ax=ax, errorbar='on')
        elif plot_style in ['2P']:
            iplt = self.overlay_pcolormesh(var, ax=ax)
        elif plot_style in ['2I']:
            iplt = self.overlay_imshow(var, ax=ax)
        elif plot_style in ['1B']:
            iplt = self.overlay_bar(var, ax=ax)
        elif plot_style in ['1F0']:
            iplt = self.overlay_fill_between_y_zero(var, ax=ax)
        elif plot_style in ['1S']:
            iplt = self.overlay_scatter(var, ax=ax)
        elif plot_style in ['1C']:
            iplt = self.overlay_multiple_colored_line(var, ax=ax)
        else:
            mylog.StreamLogger.warning(f'The attribute "plot_config.style" of {var.name} is not defined!')
            ndim = var.ndim
            if ndim == 1:
                mylog.simpleinfo.info("Set plot_config.style = '1P' for a line plot!")
                iplt = self.overlay_line(var, ax=ax) 
            elif ndim == 2:
                mylog.simpleinfo.info("Set plot_config.style = '2P' for a 2D pcolor plot!") 
                iplt = self.overlay_pcolormesh(var, ax=ax)
            else:
                raise NotImplementedError

        return iplt

    @check_panel_ax
    def overlay_line(self, var, ax=None, errorbar='off', **kwargs):
        """
        Overlay a line plot in the axes.
        :param var: A GeospaceLab Variable object
        :param ax: The axes to plot.
        :param errorbar: If 'on', show errorbar.
        :param kwargs: Other keyword arguments forwarded to ax.plot() or ax.errorbar()
        :return:
        """
        il = None

        data = self._retrieve_data_1d(var)
        x = data['x']
        y = data['y']
        y_err = data['y_err']

        plot_config = basic.dict_set_default(kwargs, **var.visual.plot_config.line)
        plot_config = basic.dict_set_default(plot_config, **self._default_plot_config)
        # if self.axes_overview[ax]['twinx'] in ['on', 'self']:
        #     if var.visual.axis[1].label is None:
        #         var.visual.axis[1].label = '@v.label'
        #     if var.visual.axis[1].unit is None:
        #         var.visual.axis[1].unit = '@v.unit' 
        if var.visual.axis[1].label is None:
            var.visual.axis[1].label = '@v.label'
        if var.visual.axis[1].unit is None:
            var.visual.axis[1].unit = '@v.unit_label'
        label = var.get_visual_axis_attr(axis=2, attr_name='label')
        plot_config = basic.dict_set_default(plot_config, label=label)
        if errorbar == 'off':
            il = ax.plot(x, y, **plot_config)
        elif errorbar == 'on':
            errorbar_config = dict(plot_config)
            errorbar_config.update(var.visual.plot_config.errorbar)
            y_err = y_err.flatten()
            if any(y_err < 0):
                mylog.StreamLogger.warning("Negative values of error detected in {}".format(var))
                y_err[y_err < 0] = np.nan
            il = ax.errorbar(x.flatten(), y.flatten(), yerr=y_err.flatten(), **errorbar_config)
        if type(il) is not list:
            il = [il]
        self.axes_overview[ax]['lines'].extend(il)
        if self.axes_overview[ax]['legend'] is None:
            self.axes_overview[ax]['legend'] = 'on'
        # elif self.axes_overview[ax]['legend'] == 'off':
        #    var.visual.axis[1].label = '@v.label'
        return il

    @check_panel_ax
    def overlay_bar(self, *args, ax=None, **kwargs):

        var = args[0]
        il = None
        data = self._retrieve_data_1d(var)
        x = data['x'].flatten()
        height = data['y'].flatten()

        bar_config = basic.dict_set_default(kwargs, **var.visual.plot_config.bar)

        color_by_value = bar_config.pop('color_by_value', False)
        colormap = bar_config.pop('colormap', 'viridis')
        vmin = bar_config.pop('vmin', np.nanmin(height))
        vmax = bar_config.pop('vmax', np.nanmax(height))
        if color_by_value:
            if isinstance(colormap, str):
                colormap = mpl.cm.get_cmap(colormap)
            norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
            colors = colormap(norm(height))
            bar_config.update(color=colors)

        # plot_config = basic.dict_set_default(plot_config, **self._default_plot_config)
        ib = ax.bar(x, height, **bar_config)
        return [ib]

    @check_panel_ax
    def overlay_pcolormesh(self, *args, ax=None, **kwargs):
        var = args[0]

        data = self._retrieve_data_2d(var)
        x = data['x']
        y = data['y']
        z = data['z']

        if x.shape[0] == z.shape[0]:
            delta_x = np.diff(x, axis=0)
            x[:-1, :] = x[:-1, :] + delta_x/2
            x = np.vstack((
                np.array(x[0, :] - delta_x[0, :] / 2)[np.newaxis, :],
                x[:-1, :],
                np.array(x[-1, :] + delta_x[-1, :] / 2)[np.newaxis, :]
            ))
        if len(x.shape) == 2:
            if x.shape[1] == z.shape[1]:
                x = np.hstack((x, x[:, -1].reshape((x.shape[0], 1))))
        if len(y.shape) == 1:
            y = y[np.newaxis, :]
        if y.shape[1] == z.shape[1]:
            delta_y = np.diff(y, axis=1)
            y[:, :-1] = y[:, :-1] + delta_y/2
            y = np.hstack((
                np.array(y[:, 0] - delta_y[:, 0]/2).reshape((y.shape[0], 1)),
                y[:, :-1],
                np.array(y[:, -1] + delta_y[:, -1]/2).reshape((y.shape[0], 1)),
            ))

        if y.shape[0] == z.shape[0]:
            y = np.vstack((y, y[-1, :].reshape((1, y.shape[1]))))

        pcolormesh_config = var.visual.plot_config.pcolormesh
        z_lim = var.visual.axis[2].lim
        if z_lim is None:
            z_lim = [np.nanmin(z.flatten()), np.nanmax(z.flatten())]
        z_scale = var.visual.axis[2].scale
        if z_scale == 'log':
            norm = mpl_colors.LogNorm(vmin=z_lim[0], vmax=z_lim[1])
            pcolormesh_config.update(norm=norm)
        else:
            pcolormesh_config.update(vmin=z_lim[0])
            pcolormesh_config.update(vmax=z_lim[1])
        colormap = mycmap.get_colormap(var.visual.plot_config.pcolormesh.get('cmap', None))
        pcolormesh_config.update(cmap=colormap)

        im = ax.pcolormesh(x.T, y.T, z.T, **pcolormesh_config)
        self.axes_overview[ax]['collections'].extend([im])
        if self.axes_overview[ax]['colorbar'] is None:
            self.axes_overview[ax]['colorbar'] = 'on'
        return [im]

    @check_panel_ax
    def _get_var_for_config(self, ax=None, ind=0):
        var_for_config = self.axes_overview[ax]['variables'][ind]
        self._var_for_config = var_for_config
        return var_for_config

    def _set_xaxis(self, ax):
        var_for_config = self._get_var_for_config(ax=ax)
        if self._xlim[0] is None:
            self._xlim[0] = mpl_dates.num2date(ax.get_xlim()[0])
        if self._xlim[1] is None:
            self._xlim[1] = mpl_dates.num2date(ax.get_xlim()[1])

        # reverse the x axis if timeline_reverse=True
        if self.timeline_reverse:
            var_for_config.visual.axis[0].reserve = True
        if var_for_config.visual.axis[0].reverse:
            self._xlim = [self._xlim[1], self._xlim[0]]
        ax.set_xlim(self._xlim)
        ax.xaxis.set_tick_params(labelsize=plt.rcParams['xtick.labelsize'])
        ax.xaxis.set_tick_params(
            which='both',
            direction=self._default_xtick_params['direction'],
            bottom=True, top=True
        )
        ax.xaxis.set_tick_params(which='major', length=8)
        ax.xaxis.set_tick_params(which='minor', length=4)

        # use date locators
        # majorlocator, minorlocator, majorformatter = ticktool.set_timeline(self._xlim[0], self._xlim[1])
        from geospacelab.visualization.mpl.axis_ticks import DatetimeMajorFormatter, DatetimeMajorLocator, DatetimeMinorLocator

        maxticks = var_for_config.visual.axis[0].major_tick_max
        if maxticks is None:
            maxticks = 9
        minticks = var_for_config.visual.axis[0].major_tick_min
        if minticks is None:
            minticks = 4
        majorlocator = DatetimeMajorLocator(maxticks=maxticks, minticks=minticks)
        majorformatter = DatetimeMajorFormatter(majorlocator, same_format=self.timeline_same_format)

        if not self.bottom_panel:
            majorformatter = mpl_ticker.NullFormatter()
            plt.setp(ax.get_xticklabels(), visible=False)
        ax.xaxis.set_major_locator(majorlocator)
        ax.xaxis.set_major_formatter(majorformatter)
        # minor locator must be set up after setting the major locator
        minormaxticks = var_for_config.visual.axis[0].minor_tick_max
        if minormaxticks is None:
            minormaxticks = maxticks * 10 + 1
        minorminticks = var_for_config.visual.axis[0].minor_tick_min
        if minorminticks is None:
            minorminticks = minticks * 3 -1
        minorlocator = DatetimeMinorLocator(ax=ax, majorlocator=majorlocator, maxticks=minormaxticks, minticks=minorminticks)
        ax.xaxis.set_minor_locator(minorlocator)
        if self.bottom_panel:
            self._set_xaxis_ticklabels(ax, majorformatter=majorformatter)

    def _set_xaxis_ticklabels(self, ax, majorformatter=None):
        var_for_config = self._get_var_for_config(ax=ax)
        # ax.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'])
        # set UT timeline
        if not list(self.timeline_extra_labels):

            ax.set_xlabel('UT', fontsize=plt.rcParams['axes.labelsize'], fontweight='normal')
            return

        
        ax_pos = ax.get_position()
        
        if self._default_xtick_params['extral_labels_x_offset'] is None:
            figwidth = self.figure.get_size_inches()[0]*2.54
            xoffset = - 0.02 - 0.06 * 16/figwidth * ax_pos.width
        else:
            xoffset = self._default_xtick_params['extral_labels_x_offset']

        if self._default_xtick_params['extral_labels_y_offset'] is None:             
            figheight = self.figure.get_size_inches()[1]*2.54
            yoffset = - 0.013 - 0.031 * 10/figheight * ax_pos.height
        else:
            yoffset = self._default_xtick_params['extral_labels_y_offset']
        
        ticks = ax.get_xticks()
        ylim0, _ = ax.get_ylim()
        xy_fig = []
        # transform from data cs to figure cs
        for tick in ticks:
            px = ax.transData.transform([tick, ylim0])
            xy = self.figure.transFigure.inverted().transform(px)
            xy_fig.append(xy)

        xlabels = ['UT']
        x_depend = var_for_config.get_depend(axis=0, retrieve_data=True)
        x0 = np.array(mpl_dates.date2num(x_depend['UT'])).flatten()
        x1 = np.array(ticks)
        ys = [x1]       # list of tick labels
        for ind, label in enumerate(self.timeline_extra_labels):
            if label in x_depend.keys():
                y0 = x_depend[label].flatten()
            elif label in var_for_config.dataset.keys():
                y0 = var_for_config.dataset[label].value
            else:
                raise KeyError
            if 'MLT' in label.upper():
                y0_sin = np.sin(y0 / 24. * 2 * np.pi)
                y0_cos = np.cos(y0 / 24. * 2 * np.pi)
                itpf_sin = interp1d(x0, y0_sin, bounds_error=False, fill_value='extrapolate')
                itpf_cos = interp1d(x0, y0_cos, bounds_error=False, fill_value="extrapolate")
                y0_sin_i = itpf_sin(x1)
                y0_cos_i = itpf_cos(x1)
                rad = np.sign(y0_sin_i) * (np.pi / 2 - np.arcsin(y0_cos_i))
                rad = np.where((rad >= 0), rad, rad + 2 * np.pi)
                y1 = rad / 2. / np.pi * 24.
            elif 'LON' in label.upper():
                y0_sin = np.sin(y0 * np.pi / 180.)
                y0_cos = np.cos(y0 * np.pi / 180.)
                itpf_sin = interp1d(x0, y0_sin, bounds_error=False, fill_value='extrapolate')
                itpf_cos = interp1d(x0, y0_cos, bounds_error=False, fill_value="extrapolate")
                y0_sin_i = itpf_sin(x1)
                y0_cos_i = itpf_cos(x1)
                rad = np.sign(y0_sin_i) * (np.pi / 2 - np.arcsin(y0_cos_i))
                y1 = rad * 180. / np.pi
            else:
                itpf = interp1d(x0, y0, bounds_error=False, fill_value='extrapolate')
                y1 = itpf(x1)
            ys.append(y1)
            xlabels.append(label)

        for ind, xticks in enumerate(ys):
            ax.text(
                xy_fig[0][0] + xoffset, xy_fig[0][1] + yoffset * ind - 0.013,
                xlabels[ind].replace('_', '/'),
                fontsize=plt.rcParams['xtick.labelsize']-2, fontweight='normal',
                horizontalalignment='right', verticalalignment='top',
                transform=self.figure.transFigure
            )
            for ind_pos, xtick in enumerate(xticks):
                if np.isnan(xtick):
                    continue
                if ind == 0:
                    text = majorformatter.format_data(xtick)
                elif 'MLT' in xlabels[ind]:
                    text = (datetime.datetime(1970, 1, 1) + datetime.timedelta(hours=xtick)).strftime('%H:%M')
                else:
                    text = '%.1f' % xtick
                ax.text(
                    xy_fig[ind_pos][0], xy_fig[ind_pos][1] + yoffset * ind - 0.013,
                    text,
                    fontsize=plt.rcParams['xtick.labelsize']-2,
                    horizontalalignment='center', verticalalignment='top',
                    transform=self.figure.transFigure
                )
        ax.xaxis.set_major_formatter(mpl_ticker.NullFormatter())

        # if self.major_timeline == 'MLT':
        #     for ind_pos, xtick in enumerate(ys[mlt_ind]):
        #         if xtick == np.nan:
        #             continue
        #         text = (datetime.datetime(1970, 1, 1) + datetime.timedelta(hours=xtick)).strftime('%H:%M')
        #         plt.text(
        #             xy_fig[ind_pos][0], xy_fig[ind_pos][1] - yoffset * ind - 0.15,
        #             text,
        #             fontsize=9,
        #             horizontalalignment='center', verticalalignment='top',
        #             transform=self.figure.transFigure
        #         )
        #     ax.set_xlabel('MLT')
        #     return



    @check_panel_ax
    def _set_yaxis(self, ax=None):
        var_for_config = self._get_var_for_config(ax=ax)
        ax.tick_params(axis='y', which='major', labelsize=plt.rcParams['ytick.labelsize'])
        # Set y axis lim
        self._set_ylim(ax=ax)

        # set y labels and alignment two methods: fig.align_ylabels(axs[:, 1]) or yaxis.set_label_coords
        if self.axes_overview[ax]['twinx'] != 'off':
            if var_for_config.visual.axis[1].label in ['@v.group', None]:
                var_for_config.visual.axis[1].label = '@v.label'
        ylabel = var_for_config.get_visual_axis_attr('label', axis=1)
        yunit = var_for_config.get_visual_axis_attr('unit', axis=1)
        ylabel_style = var_for_config.get_visual_axis_attr('label_style', axis=1)

        if self.axes_overview[ax]['twinx'] == 'self':
            ylabel = self.generate_label(ylabel, unit=yunit, style='single')
            ax.set_ylabel(ylabel, va='baseline', rotation=270, fontsize=plt.rcParams['axes.labelsize'])
        else:
            ylabel = self.generate_label(ylabel, unit=yunit, style=ylabel_style)
            label_pos = var_for_config.visual.axis[1].label_pos

            ax.set_ylabel(ylabel, va='bottom', fontsize=plt.rcParams['axes.labelsize'])
            if label_pos is not None:
                ax.yaxis.set_label_coords(label_pos[0], label_pos[1])
        ylim = ax.get_ylim()

        # set yaxis scale
        yscale = var_for_config.visual.axis[1].scale
        ax.set_yscale(yscale)

        # set yticks and yticklabels, usually do not change the matplotlib default
        yticks = var_for_config.visual.axis[1].ticks
        yticklabels = var_for_config.visual.axis[1].tick_labels

        if yscale == 'linear':
            if yticks is not None:
                ax.set_yticks(yticks)
                if yticklabels is not None:
                    ax.set_yticklabels(yticklabels)
            else:
                major_max = var_for_config.visual.axis[1].major_tick_max
                if major_max is None:
                    # ax.yaxis.set_minor_locator(mpl_ticker.AutoLocator())
                    major_max = 5

                ax.yaxis.set_major_locator(mpl_ticker.MaxNLocator(major_max))
            minor_max = var_for_config.visual.axis[1].minor_tick_max
            if minor_max is None:
                ax.yaxis.set_minor_locator(mpl_ticker.AutoMinorLocator())
            else:
                ax.yaxis.set_minor_locator(mpl_ticker.MaxNLocator(minor_max))

        elif yscale == 'log':
            locmin = mpl_ticker.LogLocator(base=10.0,
                                           subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                           numticks=12
                                           )
            ax.yaxis.set_minor_locator(locmin)
            # ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        else:
            raise NotImplementedError

        ax.set_ylim(ylim)

    @check_panel_ax
    def _set_ylim(self, ax=None, zero_line='on'):

        var_for_config = self.axes_overview[ax]['variables'][0]
        ylim = var_for_config.visual.axis[1].lim

        ylim_current = ax.get_ylim()
        if ylim is None:
            ylim = ylim_current
        elif ylim[0] == -np.inf and ylim[1] == np.inf:
            maxlim = np.max(np.abs(ylim_current))
            ylim = [-maxlim, maxlim]
        else:
            if ylim[0] is None:
                ylim[0] = ylim_current[0]
            if ylim[1] is None:
                ylim[1] = ylim_current[1]
        if zero_line == 'on' and self.axes_overview[ax]['twinx']=='off':
            if (ylim[0] < 0) and (ylim[1] > 0):
                ax.plot(ax.get_xlim(), [0, 0], 'k--', linewidth=0.5)

        if var_for_config.visual.axis[1].reverse:
            ylim = [ylim[1], ylim[0]]
        ax.set_ylim(ylim)
        return

    def _check_legend(self, ax):
        ax_ov = self.axes_overview[ax]
        var_for_config = ax_ov['variables'][0]
        legend_config = var_for_config.visual.plot_config.legend
        # get color
        legend_config = basic.dict_set_default(legend_config, **self._default_legend_config)

        if len(ax_ov['lines']) == 1 and ax_ov['lines'][0]._label == '_line0':
            return

        ax.legend(handles=ax_ov['lines'], **legend_config)

    def _check_twinx(self, ax):
        ax_ov = self.axes_overview[ax]
        axes = [ax]
        axes.extend(ax_ov['twinx_axes'])
        for ind, pax in enumerate(axes):

            pax_ov = self.axes_overview[pax]
            if list(pax_ov['lines']):
                il = pax_ov['lines'][0]
                if isinstance(il, mpl.container.ErrorbarContainer):
                    il_ = il[0]
                else:
                    il_ = il
                pax.yaxis.label.set_color(il_.get_color())
                pax.tick_params(axis='y', which='both', colors=il_.get_color())
                pax.spines['right'].set_edgecolor(il_.get_color())
            else:
                continue

    def _check_colorbar(self, ax):
        ax_ov = self.axes_overview[ax]
        var_for_config = ax_ov['variables'][0]
        colorbar_config = var_for_config.visual.plot_config.colorbar

        c_scale = var_for_config.visual.axis[2].scale
        c_label = var_for_config.get_visual_axis_attr('label', axis=2)
        c_unit = var_for_config.get_visual_axis_attr('unit', axis=2)
        c_label_style = var_for_config.get_visual_axis_attr('label_style', axis=2)

        c_label = self.generate_label(c_label, unit=c_unit, style=c_label_style)
        c_ticks = var_for_config.visual.axis[2].ticks
        c_tick_labels = var_for_config.visual.axis[2].tick_labels
        im = ax_ov['collections'][0]

        offset = self._default_colorbar_offset
        ntwinx = len(ax_ov['twinx_axes'])
        cax_position = [1.02 + offset * ntwinx, 0.01, 0.025, 0.85]
        colorbar_config.update(
            cax_scale=c_scale, cax_label=c_label, cax_ticks=c_ticks, cax_tick_labels=c_tick_labels,
            cax_position=cax_position
        )

        cb = self.add_colorbar(im, cax='new', **colorbar_config)

    def _retrieve_data_1d(self, var):
        x_data = var.get_visual_axis_attr(axis=0, attr_name='data')
        if type(x_data) == list:
            x_data = x_data[0]
        if x_data is None:
            depend_0 = var.get_depend(axis=0)
            try:
                x_data = depend_0['UT']  # numpy array, type=datetime
            except:
                mylog.StreamLogger.warning("The dependence on UT is not set!")
                try:
                    x_data = var.dataset['DATETIME'].value
                except:
                    try:
                        x_data = var.dataset['SC_DATETIME'].value
                    except:
                        mylog.StreamLogger.warning("Cannot find the datetime array!")
                        x_data = None

        if x_data is None:
            x_data = np.array([self.dt_fr, self.dt_to])
            var.visual.axis[0].mask_gap = False
        x_shift = var.get_visual_axis_attr(axis=0, attr_name='shift')
        if x_shift is not None:
            data = x_data.flatten() - datetime.timedelta(seconds=x_shift)
            x_data = np.reshape(data, x_data.shape)
        y_data = var.get_visual_axis_attr(axis=1, attr_name='data')
        if y_data is None:
            y_data = var.value
            if y_data is None:
                y_data = np.empty_like(x_data, dtype=np.float32)
                y_data[::] = np.nan

        y_err_data = var.get_visual_axis_attr(axis=1, attr_name='data_err')
        if y_err_data is None:
            y_err_data = var.error

        if self.time_res is None:
            x_data_res = var.visual.axis[0].data_res
        else:
            x_data_res = self.time_res
        x = x_data
        y = y_data * var.visual.axis[1].data_scale
        if y_err_data is None:
            y_err = np.empty_like(y)
            y_err[:] = 0
        else:
            y_err = y_err_data * var.visual.axis[1].data_scale

        # resample time if needed
        x_data = x
        y_data = y
        y_err_data = y_err
        time_gap = var.visual.axis[0].mask_gap
        if time_gap is None:
            time_gap = self.time_gap
        if time_gap:
            x, y = arraytool.data_resample(
                x_data, y_data, xtype='datetime', xres=x_data_res, method='Null', axis=0)

            x, y_err = arraytool.data_resample(
                x_data, y_err_data, xtype='datetime', xres=x_data_res, method='Null', axis=0)
        data = {'x': x, 'y': y, 'y_err': y_err}
        return data

    def _retrieve_data_2d(self, var):
        x_data = var.get_visual_axis_attr(axis=0, attr_name='data')
        if type(x_data) == list:
            x_data = x_data[0]
        if x_data is None:
            depend_0 = var.get_depend(axis=0)
            try:
                x_data = depend_0['UT']  # numpy array, type=datetime
            except:
                x_data = None
            if x_data is None:
                x_data = np.array([self._xlim[0], self._xlim[1]]).reshape(2, 1)
                var.visual.axis[0].mask_gap = False
        x_shift = var.get_visual_axis_attr(axis=0, attr_name='shift')
        if x_shift is not None:
            data = x_data.flatten() - datetime.timedelta(seconds=x_shift)
            x_data = np.reshape(data, x_data.shape)
        y_data = var.get_visual_axis_attr(axis=1, attr_name='data')
        if type(y_data) == list:
            y_data = y_data[0]
        if y_data is None:
            y_data = var.get_depend(axis=1, retrieve_data=True)
            y_data_keys = list(y_data.keys())
            y_data = y_data[y_data_keys[0]]
            if y_data is None:
                y_data = np.array([0, 1]).reshape(1, 2)
        y_data = y_data * var.visual.axis[1].data_scale

        z_data = var.get_visual_axis_attr(axis=2, attr_name='data')
        if type(z_data) == list:
            z_data = z_data[0]
        if z_data is None:
            z_data = var.value
            if z_data is None:
                z_data = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        if (x_data is None) or (z_data is None):
            raise ValueError

        z_data = z_data * var.visual.axis[2].data_scale
        
        if self.time_res is None:
            x_data_res = var.visual.axis[0].data_res
        else:
            x_data_res = self.time_res
        time_gap = var.visual.axis[0].mask_gap
        if time_gap is None:
            time_gap = self.time_gap
        if time_gap:
            # x, y, z = arraytool.data_resample_2d(
            #     x=x_data, y=y_data, z=z_data, xtype='datetime', xres=x_data_res, method='Null', axis=0)
            x, y, z = arraytool.regridding_2d_xgaps(x_data, y_data, z_data, xtype='datetime', xres=x_data_res)
        else:
            x = x_data
            y = y_data
            z = z_data

        data = {'x': x, 'y': y, 'z': z}
        return data

    @staticmethod
    def generate_label(label: str, unit: str='', style: str='double'):
        label = label
        if str(unit):
            if style == 'single':
                label = label + " " + '(' + unit + ')'
            elif style == 'double':
                label = label + "\n" + '(' + unit + ')'
            else:
                raise NotImplementedError
        return label

def check_panel_ax(func):
    def wrapper(*args, **kwargs):
        obj = args[-1]
        kwargs.setdefault('ax', None)
        if kwargs['ax'] is None:
            kwargs['ax'] = obj.axes['major']
        result = func(*args, **kwargs)
        return result
    return wrapper


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


def test_tspanel():
    from geospacelab.express.eiscat_dashboard import EISCATDashboard
    dt_fr = datetime.datetime.strptime('20211010' + '1700', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20211010' + '2100', '%Y%m%d%H%M')

    site = 'UHF'
    antenna = 'UHF'
    modulation = 'ant'
    load_mode = 'AUTO'
    data_file_type = 'eiscat-hdf5'

    viewer = EISCATDashboard(dt_fr, dt_to, site=site, antenna=antenna, modulation=modulation,
                                 data_file_type=data_file_type, load_mode=load_mode, status_control=True,
                                 residual_control=True)

    # select beams before assign the variables
    # viewer.dataset.select_beams(field_aligned=True)
    # viewer.dataset.select_beams(az_el_pairs=[(188.6, 77.7)])
    viewer.status_mask()

    n_e = viewer.assign_variable('n_e')
    T_i = viewer.assign_variable('T_i')
    T_e = viewer.assign_variable('T_e')
    v_i = viewer.assign_variable('v_i_los')
    az = viewer.assign_variable('AZ')
    el = viewer.assign_variable('EL')

    panel1 = TSPanel(dt_fr=dt_fr, dt_to=dt_to)
    panel1.draw([n_e])
    plt.show()


if __name__ == '__main__':
    test_tspanel()