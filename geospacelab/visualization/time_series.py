import matplotlib.pyplot as plt
import numpy
import datetime
import matplotlib as mpl
import pathlib

import numpy as np
from scipy.interpolate import interp1d


from geospacelab.datahub import DataHub, VariableModel
from geospacelab.visualization.mpl_toolbox.dashboard import Dashboard
# from geospacelab.visualization.mpl_toolbox.figure import Figure
import geospacelab.visualization.mpl_toolbox.colormaps as mycmap
import geospacelab.toolbox.utilities.numpyarray as arraytool
import geospacelab.toolbox.utilities.pybasic as basic

def test():
    pass

class TS(DataHub, Dashboard):

    def __init__(self, **kwargs):
        self.layout = []
        self.plot_types = None
        new_figure = kwargs.pop('new_figure', True)
        self.resample_time = kwargs.pop('resample_time', True)
        super().__init__(visual='on', new_figure=new_figure, **kwargs)

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
            self.xlim[0] = dt_fr
        if dt_to is not None:
            self.xlim[0] = dt_to

        bottom = False
        for ind, panel in enumerate(self.panels.values()):
            plot_layout = self.layout[ind]
            plot_type = self.plot_types[ind]
            self._set_panel(panel, plot_type, plot_layout)
            if ind == len(self.panels.keys()):
                bottom = True

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
            # ax.grid(True, which='both', linewidth=0.2)
        elif plot_type == '1noE':
            valid = self._plot_lines(ax, plot_layout, errorbar='off')
        elif plot_type == '1S':
            valid = self._scatter(ax, plot_layout)
        elif plot_type in ['2', '2P']:
            valid = self._pcolormesh(ax, plot_layout)
        elif plot_type == '2V':
            valid = self._vector(ax, plot_layout)
        elif plot_type == '1Ly+':
            valid = self._plot_lines_with_additional_y_axis(ax, plot_layout)
        elif plot_type == '1Lyy':
            valid = self._plot_lines_yy(ax, plot_layout)

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
        plot_layout_flatten = basic.list_flatten(plot_layout)
        var_for_config = plot_layout_flatten[0]
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

        # set ylim
        ylim = var_for_config.visual.ylim
        ymin = numpy.nanmin(yy)
        if numpy.isnan(ymin):
            ymin = -1
        ymax = numpy.nanmax(yy)
        if numpy.isnan(ymax):
            ymax = 1
        if ylim is None:
            ylim = [ymin, ymax]
        if ylim[0] == -numpy.inf and ylim[1] == numpy.inf:
            max = numpy.max(numpy.abs([ymin, ymax]))
            ylim = [-max, max]
        else:
            if ylim[0] is None:
                ylim[0] = ymin
            if ylim[1] is None:
                ylim[1] = ymax

        if (ylim[0] < 0) and (ylim[1] > 0):
            ax.plot(ax.get_xlim(), [0, 0], 'k--', linewidth=0.5)

        ax.set_ylim(ylim)
        return True

    def _pcolormesh(self, ax, var):
        import geospacelab.visualization.mpl_toolbox.plot2d as plot2d

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
        ax.objects.setdefault('pcolormesh', [])
        ax.objects['pcolormesh'].append(im)

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



    def save(self):
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