import matplotlib.pyplot as plt
import numpy
import datetime
import matplotlib as mpl
import matplotlib.dates as mdates
import pathlib

import numpy as np
from scipy.interpolate import interp1d


from geospacelab.datahub import DataHub, VariableModel
import geospacelab.config.preferences as pfr
from geospacelab.visualization.mpl_toolbox.dashboard import Dashboard
# from geospacelab.visualization.mpl_toolbox.figure import Figure
import geospacelab.visualization.mpl_toolbox.colormaps as mycmap
import geospacelab.toolbox.utilities.numpyarray as arraytool
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.visualization.mpl_toolbox.axes as axtool
import geospacelab.visualization.mpl_toolbox.axis_ticks as ticktool


def test():
    pfr.datahub_data_root_dir = pathlib.Path('~/01-Work/00-Data')
    dt_fr = datetime.datetime.strptime('20210309' + '0000', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20210309' + '2359', '%Y%m%d%H%M')
    database_name = 'madrigal'
    facility_name = 'eiscat'

    ts = TS(dt_fr=dt_fr, dt_to=dt_to)
    ts_1 = ts.set_dataset(datasource_contents=['madrigal', 'eiscat'],
                          site='UHF', antenna='UHF', modulation='ant', data_file_type='eiscat-hdf5')
    n_e = ts.assign_variable('n_e')
    T_i = ts.assign_variable('T_i')
    T_e = ts.assign_variable('T_e')
    az = ts.assign_variable('az')
    el = ts.assign_variable('el')

    layout = [[n_e], [T_i], [T_e], [az, el]]
    ts.set_layout(layout=layout)
    ts.show()
    ts.save_figure()
    pass

class TS(DataHub, Dashboard):

    def __init__(self, **kwargs):
        self.layout = []
        self.plot_types = None
        new_figure = kwargs.pop('new_figure', True)
        self.resample_time = kwargs.pop('resample_time', True)
        self.major_timeline = kwargs.pop('major_timeline', 'UT')
        self.timeline_extra_labels = kwargs.pop('timeline_extra_labels', [])    # e.g., ['MLT', 'GLAT']
        super().__init__(visual='on', new_figure=new_figure, **kwargs)

        self._xlim = [self.dt_fr, self.dt_to]

    def set_layout(self, layout=None, plot_types = None, gs_row_heights=1, gs_config=None):
        if gs_config is None:
            gs_config = {
                'left': 0.15,
                'right': 0.8,
                'bottom': 0.15,
                'top': 0.88,
                'hspace': 0.0,
                'wspace': 0.0
            }

        num_cols = 1
        num_rows = len(layout)
        if plot_types is None:
            self.plot_types = [None] * num_rows

        if type(gs_row_heights) == int:
            gs_row_heights = [gs_row_heights] * num_rows
        else:
            raise TypeError
        if len(gs_row_heights) != num_rows:
            raise ValueError

        gs_num_rows = sum(gs_row_heights)
        gs_num_cols = 1
        self.set_gridspec(num_rows=gs_num_rows, num_cols=gs_num_cols, **gs_config)
        rec = 0
        for ind, height in enumerate(gs_row_heights):
            row_ind = [rec, rec+height]
            col_ind = [0, 1]
            self.add_panel(row_ind=row_ind, col_ind=col_ind)

    def show(self, dt_fr=None, dt_to=None):

        if dt_fr is not None:
            self._xlim[0] = dt_fr
        if dt_to is not None:
            self._xlim[0] = dt_to

        bottom = False
        for ind, panel in enumerate(self.panels.values()):
            plot_layout = self.layout[ind]
            plot_type = self.plot_types[ind]
            self._set_panel(panel, plot_type, plot_layout)
            if ind == len(self.panels.keys()):
                bottom = True
            self._set_xaxis(panel.axes['major'], plot_layout, bottom=bottom)

    def _validate_plot_layout(self, layout_in):
        from geospacelab.datahub import VariableModel
        if not isinstance(layout_in, list):
            raise TypeError("The plot layout must be a list!")

        layout_out = []
        for ind, elem in enumerate(layout_in):
            if isinstance(elem, list):
                layout_out.append(self._validate_plot_layout(elem))
            elif issubclass(elem.__class__, VariableModel):
                var = elem
                layout_out.append(var)
            elif isinstance(elem, int):
                index = elem
                var = self.variables[index]
                layout_out.append(var)
            else:
                raise TypeError
        return layout_out

    def _set_panel(self, panel, plot_type, plot_layout):
        plot_layout = self._validate_plot_layout(plot_layout)

        if plot_type is None:
            plot_layout_flatten = basic.list_flatten(plot_layout)
            var = plot_layout_flatten[0]
            plot_type = var.visual.plot_type

        ax = panel.axes['major']
        if plot_type in ['1', '1E']:
            valid = self._plot_lines(ax, plot_layout, errorbar='on')
            self._set_yaxis(ax, plot_layout)
            # ax.grid(True, which='both', linewidth=0.2)
        elif plot_type == '1noE':
            valid = self._plot_lines(ax, plot_layout, errorbar='off')

        elif plot_type == '1S':
            valid = self._scatter(ax, plot_layout)
        elif plot_type in ['2', '2P']:
            valid = self._pcolormesh(ax, plot_layout)
            self._set_ylim(ax, plot_layout)
        elif plot_type == '2V':
            valid = self._vector(ax, plot_layout)
        elif plot_type == '1Ly+':
            valid = self._plot_lines_with_additional_y_axis(ax, plot_layout)
        elif plot_type == '1Lyy':
            valid = self._plot_lines_yy(ax, plot_layout)

    def _set_ylim(self, ax, plot_layout, zero_line='on'):
        plot_layout_flatten = basic.list_flatten(plot_layout)
        var_for_config = plot_layout_flatten[0]

        ylim = var_for_config.visual.y_lim

        ylim_current = ax.get_ylim()
        if ylim is None:
            ylim = ylim_current
        elif ylim[0] == -numpy.inf and ylim[1] == numpy.inf:
            maxlim = numpy.max(numpy.abs(ylim_current))
            ylim = [-maxlim, maxlim]
        else:
            if ylim[0] is None:
                ylim[0] = ylim_current[0]
            if ylim[1] is None:
                ylim[1] = ylim_current[1]
        if zero_line == 'on':
            if (ylim[0] < 0) and (ylim[1] > 0):
                ax.plot(ax.get_xlim(), [0, 0], 'k--', linewidth=0.5)
        ax.set_ylim(ylim)
        return

    def _set_yaxis(self, ax, plot_layout):
        ax.tick_params(axis='y', which='major', labelsize=11)

        plot_layout_flatten = basic.list_flatten(plot_layout)
        var_for_config = plot_layout_flatten[0]

        self._set_ylim(ax, plot_layout)
        # set y labels and alignment two methods: fig.align_ylabels(axs[:, 1]) or yaxis.set_label_coords
        ylabel = var_for_config.get_visual_attr('y_label')
        yunit = var_for_config.get_visual_attr('y_unit')
        if yunit is None:
            ylabel = ylabel
        else:
            ylabel = ylabel + "\n" + '(' + yunit + ')'
        pos_label = [-0.1, 0.5]
        ax.set_ylabel(ylabel, va='bottom', fontsize=14)
        ax.yaxis.set_label_coords(pos_label[0], pos_label[1])
        ylim = ax.get_ylim()
        # set yscale
        yscale = var_for_config.visual.y_scale
        ax.set_yscale(yscale)
        # set yticks and yticklabels, usually do not change the matplotlib default
        yticks = var_for_config.visual.yticks
        yticklabels = var_for_config.visual.yticklabels
        if yticks is not None:
            ax.set_yticks(yticks)
            if yticklabels is not None:
                ax.set_yticklabels(yticklabels)
        if yscale == 'linear':
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(8))
            ax.yaxis.set_minor_locator(mpl.ticker.MaxNLocator(40))
        if yscale == 'log':

            locmin = mpl.ticker.LogLocator(base=10.0,
                                           subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                           numticks=12
                                           )
            ax.yaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_ylim(ylim)

    def _set_xaxis(self, ax, plot_layout, bottom=False):
        plot_layout_flatten = basic.list_flatten(plot_layout)
        var_for_config = plot_layout_flatten[0]
        ax.set_xlim(self._xlim)
        # reverse the x axis
        # ax.set_xlim(self.dt_range[1], self.dt_range[0])
        ax.xaxis.set_tick_params(labelsize='small')
        ax.xaxis.set_tick_params(which='both', direction='inout', bottom=True, top=True)
        ax.xaxis.set_tick_params(which='major', length=8)
        ax.xaxis.set_tick_params(which='minor', length=4)

        # use date locators
        majorlocator, minorlocator, majorformatter = ticktool.set_timeline(self.dt_range[0], self.dt_range[1])
        if not bottom:
            majorformatter = mpl.ticker.NullFormatter()
        ax.xaxis.set_major_locator(majorlocator)
        ax.xaxis.set_major_formatter(majorformatter)
        ax.xaxis.set_minor_locator(minorlocator)
        if bottom:
            self._set_xaxis_ticklabels(ax, var_for_config, majorformatter=majorformatter)

    def _set_xaxis_ticklabels(self, ax, var_for_config, majorformatter=None):

        # set UT timeline
        if not list(self.timeline_extra_labels):
            ax.set_xlabel('UT')
            return

        figlength = self.figure.get_size_inches()[1]*2.54
        if figlength>20:
            yoffset = 0.02
        else:
            yoffset = 0.04
        ticks = ax.get_xticks()
        ylim0, _ = ax.get_ylim()
        xy_fig = []
        # transform from data coords to figure coords
        for tick in ticks:
            px = ax.transData.transform([tick, ylim0])
            xy = self.figure.transFigure.inverted().transform(px)
            xy_fig.append(xy)

        xlabels = ['UT']
        x_depend = var_for_config.get_depend(axis=0, retrieve_data=True)
        x0 = numpy.array(mdates.date2num(x_depend['UT'])).flatten()
        x1 = numpy.array(ticks)
        ys = [x1]       # list of tick labels
        for ind, label in enumerate(self.timeline_extra_labels):
            if label in x_depend.keys():
                y0 = x_depend[label].flatten()
            elif label in var_for_config.dataset.keys():
                    y0 = var_for_config.dataset[label]
            else:
                raise KeyError
            if label == 'MLT':
                mlt_ind = ind + 1
                y0_sin = numpy.sin(y0 / 24. * 2 * numpy.pi)
                y0_cos = numpy.cos(y0 / 24. * 2 * numpy.pi)
                itpf_sin = interp1d(x0, y0_sin, bounds_error=False, fill_value='extrapolate')
                itpf_cos = interp1d(x0, y0_cos, bounds_error=False, fill_value="extrapolate")
                y0_sin_i = itpf_sin(x1)
                y0_cos_i = itpf_cos(x1)
                rad = numpy.sign(y0_sin_i) * (numpy.pi / 2 - numpy.arcsin(y0_cos_i))
                rad = numpy.where((rad >= 0), rad, rad + 2 * numpy.pi)
                y1 = rad / 2. / numpy.pi * 24.
            else:
                itpf = interp1d(x0, y0, bounds_error=False, fill_value='extrapolate')
                y1 = itpf(x1)
            ys.append(y1)
            xlabels.append(label)

        ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())

        # if self.major_timeline == 'MLT':
        #     for ind_pos, xtick in enumerate(ys[mlt_ind]):
        #         if xtick == numpy.nan:
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

        for ind, xticks in enumerate(ys):
            plt.text(
                0.1, xy_fig[0][1] - yoffset * ind - 0.013,
                xlabels[ind],
                fontsize=14, fontweight='normal',
                horizontalalignment='right', verticalalignment='top',
                transform=self.figure.transFigure
            )
            for ind_pos, xtick in enumerate(xticks):
                if numpy.isnan(xtick):
                    continue
                if ind == 0:
                    text = majorformatter.format_data(xtick)
                elif xlabels[ind] == 'MLT':
                    text = (datetime.datetime(1970, 1, 1) + datetime.timedelta(hours=xtick)).strftime('%H:%M')
                else:
                    text = '%.1f' % xtick
                plt.text(
                    xy_fig[ind_pos][0], xy_fig[ind_pos][1] - yoffset * ind - 0.013,
                    text,
                    fontsize=14,
                    horizontalalignment='center', verticalalignment='top',
                    transform=self.figure.transFigure
                )

    def _plot_lines(self, ax, plot_layout, errorbar='on', **kwargs):
        default_plot_config = {
            'linestyle': '-',
            'linewidth': 1.5,
            'marker': '.',
            'markersize': 1
        }
        default_legend_config = {
            'loc': 'upper left',
            'bbox_to_anchor': (1.0, 1.0),
            'frameon': False,
            'fontsize': 'medium'
        }
        # set color circle, colormap is None use 'krbcmg', others: 'Set1', 'tab10', ...
        # colormap = var_for_config.visual.color
        # cc = mycmap.get_discrete_colors(len(para_inds), colormap) # color circle
        # ax.set_prop_cycle(color=cs)

        hls = []

        yy = numpy.empty((0, 1)) # for setting ylim
        for ind, var in enumerate(plot_layout):
            x_data = var.visual.x_data
            if x_data is None:
                depend_0 = var.get_depend(axis=0)
                x_data = depend_0['Time']   # numpy array, type=datetime

            y_data = var.visual.y_data
            if y_data is None:
                y_data = var.value

            y_err_data = var.visual.y_err_data
            if y_err_data is None:
                y_err_data = var.error

            x_data_res = var.visual.x_data_res
            x = x_data
            y = y_data * var.visual.y_data_scale
            if y_err_data is None:
                y_err = numpy.empty_like(y)
                y_err[:] = 0
            else:
                y_err = y_err_data * var.visual.y_data_scale

            y_err = y_err * var.visual.y_data_scale
            yy = numpy.vstack((yy, y - y_err))
            yy = numpy.vstack((yy, y + y_err))

            # resample time if needed
            if self.resample_time:
                x, y = arraytool.data_resample(
                    x, y, xtype='datetime', xres=x_data_res, method='Null', axis=0)

                x, y_err = arraytool.data_resample(
                    x, y_err, xtype='datetime', xres=x_data_res, method='Null', axis=0)

            # set default
            plot_config = var.visual.plot_config
            legend_config = var.visual.legend_config
            # get color
            basic.dict_set_default(plot_config, **default_plot_config)
            basic.dict_set_default(legend_config, **default_legend_config)
            label = var.get_visual_atrr('z_label')
            plot_config.update({'label': label})
            plot_config.update({'visible': var.visual.visible})
            errorbar_config = dict(plot_config)
            errorbar_config.update(var.visual.errorbar_config)
            option = {'plot_config': plot_config, 'errorbar_config': errorbar_config}
            if errorbar == 'on':
                hl = ax.errorbar(x, y, yerr=y_err, **errorbar_config)
            else:
                hl = ax.plot(x, y, **plot_config)
            # set legends, default: outside the axis right upper

            hls.extend(hl)  # note plot return a list of handles not a handle

        if not list(hls):
            return False
        hlgd = ax.legend(handles=hls, **legend_config)  # this return a handle

        # set default ylim
        ymin = numpy.nanmin(yy)
        if numpy.isnan(ymin):
            ymin = -1
        ymax = numpy.nanmax(yy)
        if numpy.isnan(ymax):
            ymax = 1

        ax.set_ylim([ymin-(ymax-ymin)*0.05, ymax+(ymax-ymin)*0.05])

        return True

    def _pcolormesh(self, ax, plot_layout):
        import geospacelab.visualization.mpl_toolbox.plot2d as plot2d
        var = plot_layout[0]
        x_data = var.visual.x_data
        if x_data is None:
            x_data = var.get_depend(axis=0, retrieve_data=True)
        y_data = var.visual.y_data
        if y_data is None:
            y_data = var.get_depend(axis=1, retrieve_data=True)
        z_data = var.visual.z_data
        if z_data is None:
            z_data = var.value
        if (x_data is None) or (z_data is None):
            raise ValueError

        x_data_res = var.visual.x_data_res
        if self.resample_time:
            x, y = arraytool.data_resample(
                x_data, y_data, xtype='datetime', xres=x_data_res, method='linear', axis=0)
            _, z = arraytool.data_resample(
                x_data, z_data, xtype='datetime', xres=x_data_res, method='Null', axis=0)
        else:
            x = x_data
            y = y_data
            z = z_data
        if x.shape[0] == z.shape[0]:
            timedelta = datetime.timedelta(seconds=x_data_res)
            x = numpy.vstack(
                (x, numpy.array([x[-1] + timedelta]).reshape((1, 1)))
            )
            x = x - timedelta / 2.
        if y.shape[1] == z.shape[1]:
            y_data_res = var.visual.y_data_res
            if y_data_res is None:
                y_data_res = numpy.mean(y[:, -1] - y[:, -2])
            y_append = y[:, -1] + y_data_res
            y = numpy.hstack((y, y_append.reshape((y.shape[0], 1))))
            y = y - y_data_res / 2.
        if y.shape[0] == z.shape[0]:
            y = numpy.vstack((y, y[-1, :].reshape((1, y.shape[1]))))

        pcolormesh_config = var.visual.pcolormesh_config
        z_lim = var.visual.z_lim
        if z_lim is None:
            z_lim = [numpy.nanmin[z_data.flatten()], numpy.nanmax[z_data.flatten()]]
        z_scale = var.visual.z_scale
        if z_scale == 'log':
            norm = mpl.colors.LogNorm(vmin=z_lim[0], vmax=z_lim[1])
            pcolormesh_config.update(norm=norm)
        else:
            pcolormesh_config.update(vmin=z_lim[0])
            pcolormesh_config.update(vmin=z_lim[1])
        colormap = mycmap.get_colormap(var.visual.colormap)
        pcolormesh_config.update(cmap=colormap)

        # mask zdata
        mconditions = var.visual.zdatamasks
        if mconditions is None:
            mconditions = []
        mconditions.append(numpy.nan)
        z = arraytool.numpy_array_self_mask(z, conditions=mconditions)
        im = plot2d.pcolormesh(x=x.T, y=y.T, z=z.T, ax=ax, **pcolormesh_config)
        #ax.objects.setdefault('pcolormesh', [])
        #ax.objects['pcolormesh'].append(im)

        z_label = var.get_visual_attr('z_label')
        z_unit = var.get_visual_attr('z_unit')
        if z_unit is not None:
            c_label = z_label + '\n' + '(' + z_unit + ')'
        else:
            c_label = z_label
        z_ticks = var.visual.z_ticks
        z_tick_labels = var.visual.z_tick_labels
        cax = self.__add_colorbar(ax, im, cscale=z_scale, clabel=c_label, cticks=z_ticks, cticklabels=z_ticklabels)
        ax.subaxes.append(cax)

        return True

    def __add_colorbar(self, ax, im, cscale='linear', clabel=None, cticks=None, cticklabels=None, ticklabelstep=1):
        pos = ax.get_position()
        left = pos.x1 + 0.02
        bottom = pos.y0
        width = 0.02
        height = pos.y1 - pos.y0 - 0.03
        cax, cax_ind = self.figure.add_axes([left, bottom, width, height])
        cb = self.figure.colorbar(im, cax=cax)
        ylim = cax.get_ylim()

        cb.set_label(clabel, rotation=270, va='bottom', size='medium')
        if cticks is not None:
            cb.ax.yaxis.set_ticks(cticks)
            if cticklabels is not None:
                cb.ax.yaxis.set_ticklabels(cticklabels)
        else:
            if cscale == 'log':
                num_major_ticks = int(numpy.ceil(numpy.diff(numpy.log10(ylim)))) * 2
                cax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=num_major_ticks))
                n = ticklabelstep
                [l.set_visible(False) for (i, l) in enumerate(cax.yaxis.get_ticklabels()) if i % n != 0]
                # [l.set_ha('right') for (i, l) in enumerate(cax.yaxis.get_ticklabels()) if i % n != 0]
                minorlocator = mpl.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                                     numticks=12)
                cax.yaxis.set_minor_locator(minorlocator)
                cax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        cax.yaxis.set_tick_params(labelsize='x-small')
        return [cax, cb, cax_ind]

    def save_figure(self):
        pass



default_figure_config = {
    'title': None,
    'size': (15, 10),    # (width, height)
    'size_unit': 'centimeter',
    'position': (50, 50),   # (left, top)
}

default_plt_style_label = 'seaborn-darkgrid'

if __name__ == "__main__":
    test()