import matplotlib.pyplot as plt
import numpy
import datetime
import matplotlib as mpl
import matplotlib.dates as mdates
import pathlib

from scipy.interpolate import interp1d


from geospacelab.datahub import DataHub
from geospacelab import preferences as pfr
import geospacelab.visualization.mpl_toolbox.dashboard as dashboard
# from geospacelab.visualization.mpl_toolbox.figure import Figure
import geospacelab.visualization.mpl_toolbox.colormaps as mycmap
import geospacelab.toolbox.utilities.numpyarray as arraytool
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.visualization.mpl_toolbox.axis_ticks as ticktool


# plt.style.use('ggplot')

# plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'book'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

default_layout_config = {
    'left': 0.15,
    'right': 0.8,
    'bottom': 0.15,
    'top': 0.88,
    'hspace': 0.1,
    'wspace': 0.0
}

default_figure_config = {
    'figsize': (12, 12),    # (width, height)
    'dpi': 100,
}

default_plt_style_label = 'seaborn-darkgrid'


def test():
    pfr.datahub_data_root_dir = pathlib.Path('/Users/lcai/01-Work/00-Data')
    # pfr.datahub_data_root_dir = '/data/afys-ionosphere/data'

    dt_fr = datetime.datetime.strptime('20201209' + '1800', '%Y%m%d%H%M')
    dt_to = datetime.datetime.strptime('20201210' + '0600', '%Y%m%d%H%M')
    database_name = 'madrigal'
    facility_name = 'eiscat'

    ts = TS(dt_fr=dt_fr, dt_to=dt_to)
    # ds0 = ts.set_dataset(datasource_contents=['madrigal', 'eiscat'])
    ds_1 = ts.set_dataset(datasource_contents=[database_name, facility_name],
                          site='UHF', antenna='UHF', modulation='60', data_file_type='eiscat-hdf5', load_data=False)
    ds_1.load_data(load_mode='AUTO')

    n_e = ts.assign_variable('n_e')
    T_i = ts.assign_variable('T_i')
    T_e = ts.assign_variable('T_e')
    v_i= ts.assign_variable('v_i_los')
    az = ts.assign_variable('az')
    el = ts.assign_variable('el')

    ds_1.list_all_variables()
    ts.list_assigned_variables()
    ts.list_datasets()

    panel_layouts = [[n_e], [T_e], [T_i], [v_i], [az, el]]
    ts.set_layout(panel_layouts=panel_layouts, row_height_scales=[5, 5, 5, 5, 3])
    # plt.style.use('dark_background')
    #dt_fr_1 = datetime.datetime.strptime('20201209' + '1300', '%Y%m%d%H%M')
    #dt_to_1 = datetime.datetime.strptime('20201210' + '1200', '%Y%m%d%H%M')
    dt_fr_1 = dt_fr
    dt_to_1 = dt_to
    ts.draw(dt_fr=dt_fr_1, dt_to=dt_to_1)
    title = ', '.join([ds_1.facility, ds_1.site, ds_1.experiment])
    ts.add_title(x=0.5, y=1.03, title=title)
    ts.add_panel_labels()
    dt_fr_2 = datetime.datetime.strptime('20201209' + '2030', "%Y%m%d%H%M")
    dt_to_2 = datetime.datetime.strptime('20201210' + '0130', "%Y%m%d%H%M")
    ts.add_vertical_line(dt_fr_2, bottom_extend=0, top_extend=0.02, label='Line 1', label_position='top')
    ts.add_shading(dt_fr_2, dt_to_2, bottom_extend=0, top_extend=0.02, label='Shading 1', label_position='top')
    dt_fr_3 = datetime.datetime.strptime('20201210' + '0130', "%Y%m%d%H%M")
    dt_to_3 = datetime.datetime.strptime('20201210' + '0430', "%Y%m%d%H%M")
    ts.add_top_bar(dt_fr_3, dt_to_3, bottom=0., top=0.02, label='Top bar 1')

    #ts.save_figure(file_name=title.replace(', ', '_'))
    ts.show()
    pass


class TS(DataHub, dashboard.Dashboard):

    def __init__(self, **kwargs):
        self.panel_layouts = []
        self.plot_styles = None
        new_figure = kwargs.pop('new_figure', True)
        figure_config = kwargs.pop('figure_config', default_figure_config)
        self.time_gap = kwargs.pop('time_gap', True)
        self.major_timeline = kwargs.pop('major_timeline', 'UT')
        self.timeline_extra_labels = kwargs.pop('timeline_extra_labels', [])    # e.g., ['MLT', 'GLAT']
        super().__init__(visual='on', new_figure=new_figure, figure_config=figure_config, **kwargs)

        self._xlim = [self.dt_fr, self.dt_to]

    def set_layout(self, panel_layouts=None, plot_styles=None, row_height_scales=None,
                   left=None, right=None, bottom=None, top=None, hspace=None, **kwargs):
        if row_height_scales is None:
            gs_row_heights = 1
        
        if left is None:
            left = default_layout_config['left']
        if right is None:
            right = default_layout_config['right']
        if bottom is None:
            bottom = default_layout_config['bottom']
        if top is None:
            top = default_layout_config['top']
        if hspace is None:
            hspace = default_layout_config['hspace']

        num_cols = 1
        num_rows = len(panel_layouts)
        self.panel_layouts = panel_layouts
        if type(plot_styles) is not list:
            self.plot_styles = [None] * num_rows
        elif len(plot_styles) != num_rows:
            raise ValueError

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
            row_ind = [rec, rec+height]
            col_ind = [0, 1]
            self.add_panel(row_ind=row_ind, col_ind=col_ind)
            rec = rec + height

    def draw(self, dt_fr=None, dt_to=None, axis_minor_grid=False, axis_major_grid=True):

        if dt_fr is not None:
            self._xlim[0] = dt_fr
        if dt_to is not None:
            self._xlim[1] = dt_to

        bottom = False
        for ind, panel in enumerate(self.panels.values()):
            plot_layout = self.panel_layouts[ind]
            plot_style = self.plot_styles[ind]
            plot_layout_flatten = basic.list_flatten(plot_layout)
            var = plot_layout_flatten[0]
            if plot_style is None:
                plot_style = var.visual.plot_config.style
            self._set_panel(panel, plot_style, plot_layout, var_for_config=var)
            if ind == len(self.panels.keys())-1:
                bottom = True
            self._set_xaxis(panel.axes['major'], var_for_config=var, bottom=bottom)

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

    def _set_panel(self, panel, plot_style, plot_layout, var_for_config=None):
        plot_layout = self._validate_plot_layout(plot_layout)

        ax = panel.axes['major']

        if plot_style in ['1', '1E']:
            valid = self._plot_lines(ax, plot_layout, errorbar='on')
            self._set_yaxis(ax, var_for_config=var_for_config)
            ax.grid(True, which='major', linewidth=0.5, alpha=0.5)
            ax.grid(True, which='minor', linewidth=0.1, alpha=0.5)
        elif plot_style == '1noE':
            valid = self._plot_lines(ax, plot_layout, errorbar='off')
            self._set_yaxis(ax, var_for_config=var_for_config)
            ax.grid(True, which='major', linewidth=0.5, alpha=0.5)
            ax.grid(True, which='minor', linewidth=0.1, alpha=0.5)
        elif plot_style == '1S':
            valid = self._scatter(ax, plot_layout)
            self._set_yaxis(ax, var_for_config=var_for_config)
        elif plot_style in ['2', '2P']:
            valid = self._pcolormesh(ax, plot_layout)
            ax.grid(True, which='major', linewidth=0.5, alpha=0.5)
            self._set_yaxis(ax, var_for_config=var_for_config)
        elif plot_style == '2V':
            valid = self._vector(ax, plot_layout)
        elif plot_style == 'y+':
            valid = self._plot_additional_y_axis(ax, plot_layout)

    def _prepare_data_1d(self, var):
        x_data = var.get_visual_axis_attr(axis=0, attr_name='data')
        if type(x_data) == list:
            x_data = x_data[0]
        if x_data is None:
            depend_0 = var.get_depend(axis=0)
            x_data = depend_0['UT']  # numpy array, type=datetime

        y_data = var.get_visual_axis_attr(axis=1, attr_name='data')
        if y_data is None:
            y_data = var.value

        y_err_data = var.get_visual_axis_attr(axis=1, attr_name='data_err')
        if y_err_data is None:
            y_err_data = var.error

        x_data_res = var.visual.axis[0].data_res
        x = x_data
        y = y_data * var.visual.axis[1].data_scale
        if y_err_data is None:
            y_err = numpy.empty_like(y)
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

    def _prepare_data_2d(self, var):
        x_data = var.get_visual_axis_attr(axis=0, attr_name='data')
        if type(x_data) == list:
            x_data = x_data[0]
        if x_data is None:
            depend_0 = var.get_depend(axis=0)
            x_data = depend_0['UT']  # numpy array, type=datetime

        y_data = var.get_visual_axis_attr(axis=1, attr_name='data')
        if type(y_data) == list:
            y_data = y_data[0]
        if y_data is None:
            y_data = var.get_depend(axis=1, retrieve_data=True)
            y_data_keys = list(y_data.keys())
            y_data = y_data[y_data_keys[0]]
        y_data = y_data * var.visual.axis[1].data_scale

        z_data = var.get_visual_axis_attr(axis=2, attr_name='data')
        if type(z_data) == list:
            z_data = z_data[0]
        if z_data is None:
            z_data = var.value
        if (x_data is None) or (z_data is None):
            raise ValueError

        z_data = z_data * var.visual.axis[2].data_scale
        x_data_res = var.visual.axis[0].data_res
        time_gap = var.visual.axis[0].mask_gap
        if time_gap is None:
            time_gap = self.time_gap
        if time_gap:
            x, y, z = arraytool.data_resample_2d(
                x=x_data, y=y_data, z=z_data, xtype='datetime', xres=x_data_res, method='Null', axis=0)
        else:
            x = x_data
            y = y_data
            z = z_data

        data = {'x': x, 'y': y, 'z': z}
        return data

    def _set_ylim(self, ax, var_for_config=None, zero_line='on'):

        ylim = var_for_config.visual.axis[1].lim

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

    @staticmethod
    def generate_label(label, unit='', style='double'):
        label = label
        if str(unit):
            if style == 'single':
                label = label + " " + '(' + unit + ')'
            elif style == 'double':
                label = label + "\n" + '(' + unit + ')'
            else:
                raise NotImplementedError
        return label

    def _set_yaxis(self, ax, var_for_config=None):
        ax.tick_params(axis='y', which='major', labelsize=11)
        # Set y axis lim
        self._set_ylim(ax, var_for_config=var_for_config)

        # set y labels and alignment two methods: fig.align_ylabels(axs[:, 1]) or yaxis.set_label_coords
        ylabel = var_for_config.get_visual_axis_attr('label', axis=1)
        yunit = var_for_config.get_visual_axis_attr('unit', axis=1)
        ylabel_style = var_for_config.get_visual_axis_attr('label_style', axis=1)

        ylabel = self.generate_label(ylabel, unit=yunit, style=ylabel_style)
        label_pos = var_for_config.visual.axis[1].label_pos
        if label_pos is None:
            label_pos = [-0.1, 0.5]
        ax.set_ylabel(ylabel, va='bottom', fontsize=plt.rcParams['axes.labelsize'])
        ax.yaxis.set_label_coords(label_pos[0], label_pos[1])
        ylim = ax.get_ylim()

        # set yaxis scale
        yscale = var_for_config.visual.axis[1].scale
        ax.set_yscale(yscale)

        # set yticks and yticklabels, usually do not change the matplotlib default
        yticks = var_for_config.visual.axis[1].ticks
        yticklabels = var_for_config.visual.axis[1].tick_labels
        if yticks is not None:
            ax.set_yticks(yticks)
            if yticklabels is not None:
                ax.set_yticklabels(yticklabels)
        if yscale == 'linear':
            major_max = var_for_config.visual.axis[1].major_tick_max
            minor_max = var_for_config.visual.axis[1].minor_tick_max
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(major_max))
            ax.yaxis.set_minor_locator(mpl.ticker.MaxNLocator(minor_max))
        if yscale == 'log':
            locmin = mpl.ticker.LogLocator(base=10.0,
                                           subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                           numticks=12
                                           )
            ax.yaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_ylim(ylim)

    def _set_xaxis(self, ax, var_for_config, bottom=False):
        ax.set_xlim(self._xlim)
        # reverse the x axis
        # ax.set_xlim(self.dt_range[1], self.dt_range[0])
        ax.xaxis.set_tick_params(labelsize='small')
        ax.xaxis.set_tick_params(which='both', direction='inout', bottom=True, top=True)
        ax.xaxis.set_tick_params(which='major', length=8)
        ax.xaxis.set_tick_params(which='minor', length=4)

        # use date locators
        majorlocator, minorlocator, majorformatter = ticktool.set_timeline(self._xlim[0], self._xlim[1])
        if not bottom:
            majorformatter = mpl.ticker.NullFormatter()
        ax.xaxis.set_major_locator(majorlocator)
        ax.xaxis.set_major_formatter(majorformatter)
        ax.xaxis.set_minor_locator(minorlocator)
        if bottom:
            self._set_xaxis_ticklabels(ax, var_for_config, majorformatter=majorformatter)

    def _set_xaxis_ticklabels(self, ax, var_for_config, majorformatter=None):
        ax.tick_params(axis='x', labelsize=12)
        # set UT timeline
        if not list(self.timeline_extra_labels):

            ax.set_xlabel('UT', fontsize=12, fontweight='normal')
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
                    y0 = var_for_config.dataset[label].value
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

    def get_visual_data(self, var):
        pass

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

        yy = numpy.empty((0, 1))    # for setting ylim
        for ind, var in enumerate(plot_layout):
            data = self._prepare_data_1d(var)
            x = data['x']
            y = data['y']
            y_err = data['y_err']

            yy = numpy.vstack((yy, y - y_err))
            yy = numpy.vstack((yy, y + y_err))
            # set default
            plot_config = var.visual.plot_config.line
            legend_config = var.visual.plot_config.legend
            # get color
            plot_config = basic.dict_set_default(plot_config, **default_plot_config)
            legend_config = basic.dict_set_default(legend_config, **default_legend_config)
            label = var.get_visual_axis_attr(axis=2, attr_name='label')
            plot_config.update({'label': label})
            errorbar_config = dict(plot_config)
            errorbar_config.update(var.visual.plot_config.errorbar)

            if errorbar == 'on':
                hl = ax.errorbar(x, y, yerr=y_err.flatten(), **errorbar_config)
            else:
                hl = ax.plot(x, y, **plot_config)
            # set legends, default: outside the axis right upper
            if type(hl) == list:
                hls.extend(hl)
            else:
                hls.append(hl)

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

    def _plot_yys(self, ax, plot_layout):
        pass

    def _pcolormesh(self, ax, plot_layout):
        var = plot_layout[0]
        data = self._prepare_data_2d(var)
        x = data['x']
        y = data['y']
        z = data['z']
        if x.shape[0] == z.shape[0]:
            delta_x = numpy.diff(x, axis=0)
            x[:-1, :] = x[:-1, :] + delta_x/2
            x = numpy.vstack((
                numpy.array(x[0, 0] - delta_x[0, 0] / 2).reshape((1, 1)),
                x[:-1, :],
                numpy.array(x[-1, 0] + delta_x[-1, 0] / 2).reshape((1, 1))
            ))

        if y.shape[1] == z.shape[1]:
            delta_y = numpy.diff(y, axis=1)
            y[:, :-1] = y[:, :-1] + delta_y/2
            y = numpy.hstack((
                numpy.array(y[:, 0] - delta_y[:, 0]/2).reshape((y.shape[0], 1)),
                y[:, :-1],
                numpy.array(y[:, -1] + delta_y[:, -1]/2).reshape((y.shape[0], 1)),
            ))
            # y_data_res = var.visual.y_data_res
            # if y_data_res is None:
            #     y_data_res = numpy.mean(y[:, -1] - y[:, -2])
            # y_append = y[:, -1] + y_data_res
            # y = numpy.hstack((y, y_append.reshape((y.shape[0], 1))))
            # y = y - y_data_res / 2.
        if y.shape[0] == z.shape[0]:
            y = numpy.vstack((y, y[-1, :].reshape((1, y.shape[1]))))

        pcolormesh_config = var.visual.plot_config.pcolormesh
        z_lim = var.visual.axis[2].lim
        if z_lim is None:
            z_lim = [numpy.nanmin[z.flatten()], numpy.nanmax[z.flatten()]]
        z_scale = var.visual.axis[2].scale
        if z_scale == 'log':
            norm = mpl.colors.LogNorm(vmin=z_lim[0], vmax=z_lim[1])
            pcolormesh_config.update(norm=norm)
        else:
            pcolormesh_config.update(vmin=z_lim[0])
            pcolormesh_config.update(vmax=z_lim[1])
        colormap = mycmap.get_colormap(var.visual.plot_config.color)
        pcolormesh_config.update(cmap=colormap)

        im = ax.pcolormesh(x.T, y.T, z.T, **pcolormesh_config)
        #ax.objects.setdefault('pcolormesh', [])
        #ax.objects['pcolormesh'].append(im)

        z_label = var.get_visual_axis_attr(axis=2, attr_name='label')
        z_unit = var.get_visual_axis_attr(axis=2, attr_name='unit')
        if str(z_unit):
            c_label = z_label + '\n' + '(' + z_unit + ')'
        else:
            c_label = z_label
        z_ticks = var.visual.axis[2].ticks
        z_tick_labels = var.visual.axis[2].tick_labels
        cax = self.__add_colorbar(ax, im, cscale=z_scale, clabel=c_label, cticks=z_ticks, cticklabels=z_tick_labels)
        # ax.subaxes.append(cax)

        return True

    def __add_colorbar(self, ax, im, cscale='linear', clabel=None, cticks=None, cticklabels=None, ticklabelstep=1):
        pos = ax.get_position()
        left = pos.x1 + 0.02
        bottom = pos.y0
        width = 0.02
        height = pos.y1 - pos.y0 - 0.03
        cax = self.figure.add_axes([left, bottom, width, height])
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
        return [cax, cb]

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

    def add_title(self, x=None, y=None, title=None, **kwargs):
        append_time = kwargs.pop('append_time', True)
        kwargs.setdefault('fontsize', plt.rcParams['figure.titlesize'])
        kwargs.setdefault('fontweight', 'roman')
        if append_time:
            dt_range_str = self.get_dt_range_str(style='title')
            title = title + ', ' + dt_range_str
        super().add_text(x=None, y=None, text=title, **kwargs)

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
            if 'major' not in self.axes.keys():
                ax = self.add_major_axes()
            else:
                ax = self.axes['major']
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
            text_config.setdefault('fontsize', 14)
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
            if 'major' not in self.axes.keys():
                ax = self.add_major_axes()
            else:
                ax = self.axes['major']
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
            text_config.setdefault('fontsize', 14)
            text_config.setdefault('fontweight', 'medium')
            text_config.setdefault('clip_on', False)
            ax.text(x, y, label, transform=ax.transAxes, **text_config)

    def add_top_bar(self, dt_fr, dt_to, bottom=0, top=0.02, **kwargs):
        bottom_extend = -1. - bottom
        top_extend = top
        kwargs.setdefault('alpha', 1.)
        kwargs.setdefault('facecolor', 'orange')
        self.add_shading(dt_fr, dt_to, bottom_extend=bottom_extend, top_extend=top_extend, **kwargs)


if __name__ == "__main__":
    test()