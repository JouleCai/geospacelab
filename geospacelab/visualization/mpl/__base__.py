import copy
import weakref
import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
from matplotlib.axes import Axes
from typing import Dict
# import palettable

from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.figure import Figure

from cycler import cycler

from geospacelab.config import pref
from geospacelab.toolbox.utilities.pyclass import StrBase
import geospacelab.toolbox.utilities.pybasic as pybasic
from geospacelab.visualization.mpl._helpers import check_panel_ax
import geospacelab.toolbox.utilities.pylogging as mylog
# from geospacelab.visualization.mpl.dashboards import Dashboard


try:
    mpl_style = pref.user_config['visualization']['mpl']['style']
except KeyError:
    uc = pref.user_config
    uc['visualization']['mpl']['style'] = 'light'
    pref.set_user_config(user_config=uc, set_as_default=True)


# plt.rcParams['font.serif'] = 'Ubuntu'
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'book'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12

# plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle')
mpl_style = pref.user_config['visualization']['mpl']['style']

if mpl_style == 'light':
    plt.rcParams['axes.facecolor'] = '#FCFCFC'
    plt.rcParams['text.color'] = 'k'
    default_cycler = (cycler(color=['tab:blue', 'tab:red', 'tab:green', 'tab:purple',  'tab:orange', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']))
    default_cycler = (cycler(color=['#1f77b4DD', '#ff7f0eDD', '#2ca02cDD', '#d62728DD', '#9467bdDD', '#8c564bDD', '#e377c2DD', '#7f7f7fDD', '#bcbd22DD', '#17becfDD']))
    # colors = [
    #     (0.8980392156862745, 0.5254901960784314, 0.023529411764705882),
    #     (0.36470588235294116, 0.4117647058823529, 0.6941176470588235),
    #     (0.3215686274509804, 0.7372549019607844, 0.6392156862745098),
    #     (0.6, 0.788235294117647, 0.27058823529411763),
    #     (0.8, 0.3803921568627451, 0.6901960784313725),
    #     (0.1411764705882353, 0.4745098039215686, 0.4235294117647059),
    #     (0.8549019607843137, 0.6470588235294118, 0.10588235294117647),
    #     (0.1843137254901961, 0.5411764705882353, 0.7686274509803922),
    #     (0.4627450980392157, 0.3058823529411765, 0.6235294117647059),
    #     (0.9294117647058824, 0.39215686274509803, 0.35294117647058826),
    # ]
    colors = [
        (0.36470588235294116, 0.4117647058823529, 0.6941176470588235),
        (0.9294117647058824, 0.39215686274509803, 0.35294117647058826),
        (0.3215686274509804, 0.7372549019607844, 0.6392156862745098),
        (0.8980392156862745, 0.5254901960784314, 0.023529411764705882),
        (0.6, 0.788235294117647, 0.27058823529411763),
        (0.8, 0.3803921568627451, 0.6901960784313725),
        (0.1411764705882353, 0.4745098039215686, 0.4235294117647059),
        (0.8549019607843137, 0.6470588235294118, 0.10588235294117647),
        (0.1843137254901961, 0.5411764705882353, 0.7686274509803922),
        (0.4627450980392157, 0.3058823529411765, 0.6235294117647059),

    ]
    default_cycler = (cycler(color=colors))
    plt.rc('axes', prop_cycle=default_cycler)
elif mpl_style == 'dark':
    plt.rcParams['figure.facecolor'] = '#0C1C23'
    plt.rcParams['savefig.facecolor'] = '#0C1C23'

    plt.rcParams['axes.facecolor'] = '#FFFFFF20'
    plt.rcParams['axes.edgecolor'] = '#FFFFFF3D'
    plt.rcParams['axes.labelcolor'] = '#FFFFFFD9'

    plt.rcParams['xtick.color'] = '#FFFFFFD9'
    plt.rcParams['ytick.color'] = '#FFFFFFD9'
    plt.rcParams['text.color'] = 'white'

    plt.rcParams['grid.color'] = '#FFFFFF'
    plt.rcParams['legend.facecolor'] = plt.rcParams['axes.facecolor']
    plt.rcParams['legend.edgecolor'] = '#FFFFFFD9'

    # seaborn dark:['#001c7f', '#b1400d', '#12711c', '#8c0800', '#591e71', '#592f0d', '#a23582', '#3c3c3c', '#b8850a', '#006374']
    # seaborn pastel '#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b', '#d0bbff', '#debb9b', '#fab0e4', '#cfcfcf', '#fffea3', '#b9f2f0'
    default_cycler = (cycler(color=['#F5EE33', '#33FF99', 'r', '#9467bd', '#08C7FE', '#FE66BB', ]))
    colors = [
        (0.1843137254901961, 0.5411764705882353, 0.7686274509803922),
        (0.9294117647058824, 0.39215686274509803, 0.35294117647058826),
        (0.3215686274509804, 0.7372549019607844, 0.6392156862745098),
        (0.8980392156862745, 0.5254901960784314, 0.023529411764705882),
        (0.6, 0.788235294117647, 0.27058823529411763),
        (0.8, 0.3803921568627451, 0.6901960784313725),
        (0.1411764705882353, 0.4745098039215686, 0.4235294117647059),
        (0.8549019607843137, 0.6470588235294118, 0.10588235294117647),
        (0.36470588235294116, 0.4117647058823529, 0.6941176470588235),
        (0.4627450980392157, 0.3058823529411765, 0.6235294117647059),
    ]
    default_cycler = (cycler(color=colors))
    # default_cycler = (cycler(color=palettable.cartocolors.qualitative.Safe_10.mpl_colors))
    plt.rc('axes', prop_cycle=default_cycler)
else:
    plt.style.use(mpl_style)


class FigureBase(Figure):
    """
    GeospaceLab canvas inherits from ``matplotlib.figure.Figure`` with additional functions and settings.

    :param dashboards: A collection of the dashboards in the canvas, the keys can be an integer or a string.
    :type dashboards: dict.
    :param watermark: If not ``None``, add the watermark in the canvas.
    :type watermark: {str, :class:`~geospacelab.visualization.mpl._base.Watermark`}, default: ``None``.
    :param watermark_style: The style of watermarks in the canvas.
    :type watermark_style: str, default: 'Single Box'.
    """

    _default_canvas_fontsize = 12
    _default_dashboard_class = 'DashboardBase'

    def __init__(self, *args, watermark=None, watermark_style=None, **kwargs):
        super(FigureBase, self).__init__(*args, **kwargs)

        self.dashboards = {}

        self.watermark = watermark
        self.watermark.style = watermark_style
        if watermark is not None:
            self.add_watermark()

    def add_dashboard(self, *args, label=None, dashboard_class=None, **kwargs):
        import geospacelab.visualization.mpl.dashboards as dashboards
        
        if label is None:
            label = len(self.dashboards)
        
        if len(args) == 1:
            if issubclass(args[0].__class__, DashboardBase):
                if args[0] not in self.dashboards.values():
                    self.dashboards[label] = args[0]
                return args[0]
            
        if dashboard_class is None:
            dashboard_class = self._default_dashboard_class
        if isinstance(dashboard_class, str):
            try:
                db = getattr(dashboards, dashboard_class)
            except AttributeError:
                mylog.StreamLogger.warning("Cannot find the assigned dashboard class! Use the default Dashboard instead")
                db = dashboards.Dashboard
        elif issubclass(dashboard_class, DashboardBase):
            db = dashboard_class
        else:
            raise ValueError

        self.dashboards[label] = db(*args, figure=self, from_figure=True, **kwargs)
        return self.dashboards[label]

    def add_text(self, *args, **kwargs):
        super().text(*args, **kwargs)

    def add_watermark(self, watermark=None, style=None):
        if watermark is not None:
            self.watermark = Watermark(watermark, style=style)

        if self.watermark.style.lower() == 'single box':
            bbox = dict(boxstyle='square', lw=3, ec='gray',
                        fc=(0.9, 0.9, .9, .5), alpha=0.5)
            self.add_text(
                0.5, 0.5, watermark,
                ha='center', va='center', rotation=30,
                fontsize=40, color='gray', alpha=0.5, bbox=bbox
            )
        else:
            raise NotImplemented

    @property
    def watermark(self):
        return self._watermark

    @watermark.setter
    def watermark(self, value):
        if isinstance(value, str):
            self._watermark = Watermark(value)
        elif issubclass(value.__class__, Watermark):
            self._watermark = value
        elif value is None:
            self._watermark = Watermark('')
        else:
            raise TypeError

    def __repr__(self):
        r = super().__repr__()
        return r


class DashboardBase(object):
    """
    A dashboard is a collection of panels in a figure or GeospaceLab canvas. The class inherits
    from :class:`~geospacelab.datahub.DataHub`

    :param canvas: The canvas that the dashboad will be placed.
    :type canvas: Instance of :class:`GeospaceLab Canvas<geospacelab.visualization.mpl._base.Canvas>`.
    :param panels: A collection of the panels that are placed in the dashboard.
    :type panels: dict, default: {}, the keys are integers starting from 1 if not specified.
    :param extra_axes: A collection of the axes additionally appended to the dashboard.
    :type extra_axes: dict, default: {}.
    :param gs:
    """

    _default_layout_config = {
        'left':     0.15,
        'right':    0.8,
        'bottom':   0.15,
        'top':      0.88,
        'hspace':   0.0,
        'wspace':   0.1
    }

    _default_dashboard_fontsize = 12

    def __init__(self, figure=None, figure_config=None, figure_class=FigureBase, from_figure=False, **kwargs):
        """
        Initialization

        :param figure: If ``None``, a new canvas will be created.
        :type figure: Instance of :class:`GeospaceLab Canvas<geospacelab.visualization.mpl._base.Canvas>`.
        :param figure_config: The optional keyword arguments used to create the canvas.
        :type figure_config: dict, default: {}.
        :param args: The arguments used to create a :class:`DataHub <geospacelab.datahub.DataHub>` instance..
        :param kwargs: Other keyword arguments used to create a :class:`DataHub <geospacelab.datahub.DataHub>` instance.
        """
        super().__init__(**kwargs)

        self._figure_class = figure_class
        self._from_figure = from_figure
        if figure_config is None:
            figure_config = {}
        self._figure_config = figure_config

        self.figure = figure

        self.panels : Dict[int, PanelBase] = {}
        self.num_rows = None
        self.num_cols = None
        self.title = kwargs.pop('title', None)
        self.label = kwargs.pop('label', None)
        self.extra_axes = {}
        self.gs = None

    def clear(self):
        keys = list(self.panels.keys())
        for ind_p in keys:
            self.remove_panel(ind_p)
        for ax in self.extra_axes.values():
            ax.remove()
        self.extra_axes = {}


    def set_layout(self, num_rows=None, num_cols=None, **kwargs):
        """
        Set the layout of the dashboard in a canvas using matplotlib GridSpec.

        :param num_rows: The number of rows of the grid.
        :type num_rows: int.
        :param num_cols: The number of columns of the grid.
        :type num_cols: int.
        :param kwargs: Optional keyword arguments used in **matplotlib GridSpec**.
        """
        kwargs = pybasic.dict_set_default(kwargs, **self._default_layout_config)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.gs = self.figure.add_gridspec(num_rows, num_cols)
        self.gs.update(**kwargs)

    def add_panel(self, row_ind=None, col_ind=None,
                  index=None, label=None, panel_class=None, **kwargs):
        """
        Add a panel.

        :param row_ind: The row indices (i or [i, j]) of the panel in the dashboard grid.
        :type row_ind: int or list.
        :param col_ind: The column indices of the panel in the dashboard grid.
        :type col_ind: int or list.
        :param index: The panel index as a key in self.panels.
        :type index: int or str.
        :param label: The panel label
        :type label: str
        :param panel_class: If None, use the default Panel class.
        :type panel_class: Subclass of :class:`Panel <geospacelab.visualization.mpl._base.Panel>`
        :param kwargs: Optional keyword arguments to create a Panel instance.
        :return: index
        """
        if (row_ind is None) or (col_ind is None):
            num_panels = len(self.panels.keys())
            row_ind = int(np.floor(num_panels / self.num_cols))
            col_ind = int(num_panels - row_ind * self.num_cols)
        if isinstance(row_ind, int):
            row_ind = [row_ind, row_ind + 1]
        if isinstance(col_ind, int):
            col_ind = [col_ind, col_ind + 1]
        if panel_class is None:
            panel_class = PanelBase
        elif not issubclass(panel_class, PanelBase):
            raise TypeError

        args = [self.gs[row_ind[0]:row_ind[1], col_ind[0]:col_ind[1]]]
        panel = panel_class(*args, label=label, **kwargs)

        # panel.add_subplot(self.gs[row_ind[0]:row_ind[1], col_ind[0]:col_ind[1]], major=True, **kwargs)

        if index is None:
            index = len(self.panels.keys())
        elif index in self.panels.keys():
            raise ValueError('The panel index has been occupied. Change to a new one!')
        self.panels[index] = panel
        return panel

    def remove_panel(self, index):
        """
        Remove a panel

        :param index: The panel index.
        :type index: int or str.
        """

        self.panels[index].clear()
        del self.panels[index]
    #
    # def replace_panel(self, index, **kwargs):
    #     position = self.panels[index].get_position()
    #     self.remove_panel(index)
    #
    #     panel = self.figure.add_subplot(**kwargs)
    #     panel.set_position(position)
    #     self.panels[index] = panel

    def add_text(self, x, y, text, **kwargs):
        # add text in dashboard cs

        x_new = self.gs.left + x * (self.gs.right - self.gs.left)

        y_new = self.gs.bottom + y * (self.gs.top - self.gs.bottom)

        self.figure.add_text(x_new, y_new, text, **kwargs)

    def add_title(self, x, y, title, **kwargs):
        # add text in dashboard cs
        kwargs.setdefault('fontsize', self._default_dashboard_fontsize)
        kwargs.setdefault('ha', 'center')
        kwargs.setdefault('va', 'bottom')
        if x is None:
            x = 0.5
        if y is None:
            y = 1.05

        self.add_text(x, y, title, **kwargs)

    def add_panel_labels(self, panel_indices=None, style='alphabets', bbox_config=None, labels=list(), **kwargs):
        if panel_indices is None:
            panel_indices = self.panels.keys()
        default_bbox_config = {'facecolor': 'yellow', 'alpha': 0.3, 'edgecolor': 'none'}

        if style == 'alphabets':
            label_list = string.ascii_lowercase
        else:
            raise NotImplemented

        pos = kwargs.pop('position', [0.02, 0.9])
        x = pos[0]
        y = pos[1]

        kwargs.setdefault('ha', 'left')  # horizontal alignment
        kwargs.setdefault('va', 'center')  # vertical alignment

        pos_0 = self.panels[0].axes['major'].get_position()  # adjust y in case of different gs_row_heights
        for ind, p_index in enumerate(panel_indices):
            panel = self.panels[p_index]
            pos_1 = panel.axes['major'].get_position()
            if list(labels) :
                label = labels[ind]
            elif panel.label is None:
                label = "{}".format(label_list[ind])
            else:
                label = panel.label
                
            label = "({})".format(label)
            kwargs.setdefault('fontsize', plt.rcParams['axes.labelsize'])
            kwargs.setdefault('fontweight', 'book')
            if bbox_config is None:
                bbox_config = default_bbox_config
            else:
                default_bbox_config.update(**bbox_config)
                bbox_config = default_bbox_config
            kwargs.setdefault('bbox', bbox_config)
            y_new = 1 - pos_0.height / pos_1.height + y * pos_0.height / pos_1.height
            panel.add_label(x, y_new, label, **kwargs)

    def add_axes(self, rect, label=None, **kwargs):
        if label is None:
            label = len(self.extra_axes.keys())
            label_str = 'ax_' + str(label)
        else:
            label_str = label
        kwargs.setdefault('facecolor', 'none')
        ax = self.figure.add_axes(rect, label=label_str, **kwargs)
        self.extra_axes[label] = ax
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
    
    def save_figure(self, *args, file_dir=None, file_name=None, dpi=300, **kwargs):
        default_filetypes = {'ps': 'Postscript', 
                     'eps': 'Encapsulated Postscript', 
                     'pdf': 'Portable Document Format', 
                     'pgf': 'PGF code for LaTeX', 
                     'png': 'Portable Network Graphics', 
                     'raw': 'Raw RGBA bitmap', 
                     'rgba': 'Raw RGBA bitmap', 
                     'svg': 'Scalable Vector Graphics', 
                     'svgz': 'Scalable Vector Graphics', 
                     'jpg': 'Joint Photographic Experts Group', 
                     'jpeg': 'Joint Photographic Experts Group', 
                     'tif': 'Tagged Image File Format', 
                     'tiff': 'Tagged Image File Format'}
        if len(args) == 1:
            file_path = args[0]
        elif len(args) == 0:
            if type(file_name) is not str:
                raise ValueError('The file name ("file_name") must be assigned!')
            
            if file_dir is None:
                file_dir = pathlib.Path().cwd()
            else:
                file_dir = pathlib.Path(file_dir)

            file_path = file_dir / file_name
        else:
            raise ValueError
        
        # check the file extension
        sufs = file_path.suffixes
        if not list(sufs):
            format = 'png'
            file_path = file_path.with_suffix('.png')
        else:
            if sufs[-1] in default_filetypes.keys():
                format = sufs[-1].split('.')[-1]
            else:
                format = 'png'
                file_path = file_path.with_suffix('.png')

        plt.savefig(file_path, dpi=dpi, format=format, **kwargs)

    @staticmethod
    def show():
        plt.show()

    @property
    def figure(self):
        if self._figure_ref is None:
            return None
        else:
            return self._figure_ref()

    @figure.setter
    def figure(self, figure_obj):
        if figure_obj is None:
            fig_nums = plt.get_fignums()
            if list(fig_nums):
                figure_obj = plt.gcf()
            else:
                figure_obj = 'new'

        if figure_obj == 'new':
            figure = plt.figure(FigureClass=self._figure_class, **self._figure_config)
            figure._default_dashboard_class = self.__class__
            mylog.simpleinfo.info(f"Create a new figure: {figure}.")
        elif issubclass(figure_obj.__class__, self._figure_class):
            figure = figure_obj
        elif issubclass(figure_obj.__class__, plt.Figure):
            figure = figure_obj
        elif figure_obj=='off':
            figure = 'off'
            self._figure_ref = None
            return
        else:
            raise TypeError

        if issubclass(figure.__class__, FigureBase) and not self._from_figure:
            figure.add_dashboard(self)
        self._figure_ref = weakref.ref(figure)

    def __repr__(self):
        r = "GeospaceLab DashboardBase"
        return r


class PanelBase(object):
    _ax_attr_model = {
        'twinx': 'off',
        'twinx_axes': [],
        'twiny_axes': [],
        'shared': {},
        'lines': [],
        'collections': [],
        'legend': None,
        'colorbar': None,
        'variables': [],
    }
    axes_overview = {}

    def __init__(self, *args, figure=None, from_subplot=True, **kwargs):
        if figure is None:
            figure = plt.gcf()
        elif figure == 'new':
            figure_config = kwargs.pop('figure_config', {})
            figure = plt.figure(**figure_config)
        elif issubclass(figure.__class__, plt.Figure):
            figure = figure
        else:
            raise ValueError
        self.figure = figure
        self.axes = {}
        self.label = kwargs.pop('label', None)
        self._current_ax = None
        # self.objectives = kwargs.pop('objectives', {})
        if from_subplot:
            ax = self.figure.add_subplot(*args, **kwargs)
        else:
            if len(args) > 0:
                if isinstance(args[0], SubplotSpec):
                    pos = args[0].get_position(self.figure)
                    x, y, w, h = pos.x0, pos.y0, pos.x1-pos.x0, pos.y1-pos.y0
                    args = ((x, y, w, h),)
            ax = self.figure.add_axes(*args, **kwargs)
        self.axes['major'] = ax
        self.axes_overview[ax] = copy.deepcopy(self._ax_attr_model)
        self._current_ax = ax

    def __call__(self, ax=None) -> Axes:
        """
        Get the axes instance.

        :param ax: The axes key in the attribute axes.
        :type ax: str or the ax instance.
        :return: The ax instance.
        """
        if ax is None:
            ax = 'major'
        if type(ax) in [str]:
            return self.axes[ax]
        elif ax in self.axes:
            return ax
        else:
            raise AttributeError

    def clear(self):
        for ax in self.axes.values():
            ax.remove()

        self.axes = {}

    def sca(self, ax):
        """
        Set current axes.

        :param ax: the ax instance belong to the attribute axes.
        """
        plt.sca(ax)
        self._current_ax = ax

    def gca(self):
        """
        Get the current ax.

        :return: Axes instance.
        """
        return self._current_ax

    def add_axes(self, *args, major=False, label=None, **kwargs):
        """
        Add a new ax.

        :param args:
        :param major:
        :param label:
        :param kwargs:
        :return:
        """
        if major:
            label = 'major'
        else:
            if label is None:
                label = len(self.axes.keys())

        ax = self.figure.add_axes(*args, **kwargs)
        ax.patch.set_alpha(0)
        self.axes[label] = ax
        self.axes_overview[ax] = copy.deepcopy(self._ax_attr_model)
        self.sca(ax)
        return ax

    @check_panel_ax
    def add_twin_axes(self, ax=None, label=None, which='x', location='right', offset_type='outward', offset=60, **kwargs):
        if which == 'x':
            twin_func = ax.twinx
        elif which == 'y':
            twin_func = ax.twiny
        else:
            raise ValueError

        if label is None:
            label = 'ax_' + str(len(self.axes.keys()))

        ax_new = twin_func()
        ax_new.spines[location].set_position((offset_type, offset))
        self.axes[label] = ax_new
        if which == 'x':
            self.axes_overview[ax]['twinx_axes'].append(ax_new)
        else:
            self.axes_overview[ax]['twiny_axes'].append(ax_new)
        self.axes_overview[ax_new] = copy.deepcopy(self._ax_attr_model)
        return ax_new

    @check_panel_ax
    def add_grid(self, ax=None, visible=True, which='major', axis='both', **kwargs):
        self.sca(ax)
        plt.grid(visible=visible, which=which, axis=axis, **kwargs)

    @check_panel_ax
    def clear_axes(self, ax=None, collection_names=('lines', 'collections', 'images', 'patches')):
        for cn in collection_names:
            cs = getattr(ax, cn)
            ncs = len(cs)
            for i in range(ncs):
                cs.pop(ncs-1-i)

    def add_text(self, x, y, text, ax=None, **kwargs):

        if ax is None:
            ax = self()
        kwargs.setdefault('transform', ax.transAxes)
        ax.text(x, y, text, **kwargs)

    def add_label(self, x=0.02, y=0.9, label=None, **kwargs):
        ax = self.axes['major']
        if label is None:
            label = self.label
        kwargs.setdefault('ha', 'left')
        kwargs.setdefault('va', 'center')
        self.add_text(x, y, label, **kwargs)

    def add_title(self, x=0.5, y=1.02, title=None, **kwargs):
        kwargs.setdefault('ha', 'center')
        kwargs.setdefault('va', 'baseline')
        self.add_text(x, y, title, **kwargs)

    @check_panel_ax
    def overlay_plot(self, *args, ax=None, **kwargs):
        # plot_type="1P"
        ipl = ax.plot(*args, **kwargs)
        return ipl

    @check_panel_ax
    def overlay_errorbar(self, *args, ax=None, **kwargs):
        # plot_type = "1E"
        ieb = ax.errorbar(*args, **kwargs)
        return ieb

    @check_panel_ax
    def overlay_pcolormesh(self, *args, ax=None, **kwargs):
        # plot_type: "2P"
        ipm = ax.pcolormesh(*args, **kwargs)
        return ipm

    @check_panel_ax
    def overlay_imshow(self, *args, ax=None, **kwargs):
        # plot_type = "2I"
        im = ax.imshow(*args, **kwargs)
        return im

    @check_panel_ax
    def overlay_fill_between_y(self, x, y1, y2=0, ax=None, where=None, interpolate=False, **kwargs):
        ip = ax.fill_between(x, y1, y2=y2, where=where, interpolate=interpolate, **kwargs)
        return ip

    @check_panel_ax
    def overlay_fill_between_x(self, y, x1, x2=0, ax=None, where=None, interpolate=False, **kwargs):
        ip = ax.fill_between(y, x1, x2=x2, where=where, interpolate=interpolate, **kwargs)
        return ip

    @staticmethod
    def get_line_collection(
            x, y, z,
            linewidth=3, vmin=None, vmax=None, norm='linear', cmap='jet', **kwargs
    ):
        """
        Create a set of line segments and a line collection, see also: :ref:`https://matplotlib.org/2.0.2/examples/pylab_examples/multicolored_line.html`

        :param x, y, z: The line points (x, y) with the colored z values.
        :type x, y, z: list or np.ndarray.
        :type norm: Normalize z into color space.
        :type norm: {'linear', 'log', ...} or matplotlib.colors.Normalize, default: 'linear'.
        :param cmap: The color map for the plot.
        :type cmap: str or matplotlib.colors.Colormap, default: 'jet'.
        :param kwargs: Other optional keyword arguments forwarded to LineCollection.
        :return: The line collection instance.
        """

        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap, Normalize, LogNorm

        # Set norm
        if norm == 'linear':
            norm = Normalize(vmin, vmax)
        elif norm == 'log':
            norm = LogNorm(vmin, vmax)

        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be numlines x points per line x 2 (x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='cmap', norm=norm, **kwargs)
        lc.set_array(z)
        lc.set_linewidth(linewidth)

        return lc

    @check_panel_ax
    def add_multicolored_line(self, *args, ax=None, line_collection=None, **kwargs):
        if line_collection is None:
            line_collection = self.get_line_collection(*args, **kwargs)

        line = self(ax).add_collection(line_collection)
        return line

    @check_panel_ax
    def add_colorbar(
            self, im,
            ax=None, cax=None, cax_position=None, cax_scale=None, cax_label=None, cax_ticks=None, cax_tick_labels=None,
            cax_label_config=None, cax_tick_label_step=1,
            **kwargs
    ):
        """
        Add a colorbar appended to the parent ax.

        :param im: the mappable
        :param cax: If None, create a colorbar axes using the default settings, see also matplotlib.colorbar.Colorbar.
        :type cax: axes instance or {None, 'new'}
        :param ax: The appended ax instance.
        :type ax: matplotlib.pyplot.Axes isntance.
        :param cax_position: If not None, set the colorbar ax position at [x, y, width, height], which are normalized to the main ax coordinates.
        :type cax_position: 4-tuple, default: [1.02, 0.01, 0.025, 0.85].
        :param cax_label_config: Optional keyword arguments for setting the colorbar label.
        :param kwargs: Other optional keyword arguments forwarded to matplotlib.colorbar.Colorbar.
        :return: The colarbar instance
        """

        if cax_label_config is None:
            cax_label_config = {}

        if cax == 'new':
            if cax_position is None:
                cax_position = [1.02, 0.01, 0.025, 0.85]

            pos_ax = ax.get_position()
            # convert from Axes coordinates to Figure coordinates
            pos_cax = [
                pos_ax.x0 + (pos_ax.x1 - pos_ax.x0) * cax_position[0],
                pos_ax.y0 + (pos_ax.y1 - pos_ax.y0) * cax_position[1],
                (pos_ax.x1 - pos_ax.x0) * cax_position[2],
                (pos_ax.y1 - pos_ax.y0) * cax_position[3],
            ]
            cax = self.add_axes(pos_cax)

        icb = self.figure.colorbar(im, cax=cax, **kwargs)

        self.sca(ax)

        # set colorbar label
        cax_label_config = pybasic.dict_set_default(cax_label_config, rotation=270, va='bottom', size='medium')
        if cax_label is not None:
            icb.set_label(cax_label, **cax_label_config)

        ylim = cax.get_ylim()
        # set cticks
        if cax_ticks is not None:
            icb.ax.yaxis.set_ticks(cax_ticks)
            if cax_tick_labels is not None:
                icb.ax.yaxis.set_ticklabels(cax_tick_labels)
        else:
            if cax_scale == 'log':
                num_major_ticks = int(np.ceil(np.diff(np.log10(ylim)))) * 2
                cax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=num_major_ticks))
                n = cax_tick_label_step
                [l.set_visible(False) for (i, l) in enumerate(cax.yaxis.get_ticklabels()) if i % n != 0]
                # [l.set_ha('right') for (i, l) in enumerate(cax.yaxis.get_ticklabels()) if i % n != 0]
                minorlocator = mpl.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                                     numticks=12)
                cax.yaxis.set_minor_locator(minorlocator)
                cax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        cax.yaxis.set_tick_params(labelsize='x-small')
        return icb

    @property
    def major_ax(self):
        return self.axes['major']

    @major_ax.setter
    def major_ax(self, ax):
        self.axes['major'] = ax

    def _raise_error(self, error):
        raise error



class Watermark(StrBase):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in)
        style = kwargs.pop('style', None)
        if style is None:
            style = 'Single Box'
        obj.style = style
        return obj