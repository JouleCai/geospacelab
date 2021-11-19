
import weakref
import string
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.figure import Figure

from geospacelab.datahub import DataHub
from geospacelab.toolbox.utilities.pyclass import StrBase
import geospacelab.toolbox.utilities.pybasic as pybasic
from geospacelab.visualization.mpl._helpers import check_panel_ax


class Canvas(Figure):
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

    def __init__(self, *args, watermark=None, watermark_style=None, **kwargs):
        super(Canvas, self).__init__(*args, **kwargs)

        self.dashboards = {}

        self.watermark = watermark
        self.watermark.style = watermark_style
        if watermark is not None:
            self.add_watermark()

    def add_dashboard(self, *args, label=None, **kwargs):
        if label is None:
            label = len(self.dashboards) + 1
        self.dashboards[label] = Dashboard(*args, canvas=self, label=label, **kwargs)

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


class Dashboard(DataHub):
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

    def __init__(self, *args, canvas=None, canvas_config=None, **kwargs):
        """
        Initialization

        :param canvas: If ``None``, a new canvas will be created.
        :type canvas: Instance of :class:`GeospaceLab Canvas<geospacelab.visualization.mpl._base.Canvas>`.
        :param canvas_config: The optional keyword arguments used to create the canvas.
        :type canvas_config: dict, default: {}.
        :param args: The arguments used to create a :class:`DataHub <geospacelab.datahub.DataHub>` instance..
        :param kwargs: Other keyword arguments used to create a :class:`DataHub <geospacelab.datahub.DataHub>` instance.
        """
        super().__init__(*args, **kwargs)

        if canvas_config is None:
            canvas_config = dict()
        if canvas is None:
            canvas = Canvas(**canvas_config)
            print(canvas)

        self.canvas = canvas

        self.panels = {}
        self.title = kwargs.pop('title', None)
        self.label = kwargs.pop('label', None)
        self.extra_axes = {}
        self.gs = None

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
        self.gs = self.canvas.add_gridspec(num_rows, num_cols)
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
        if isinstance(row_ind, int):
            row_ind = [row_ind, row_ind + 1]
        if isinstance(col_ind, int):
            col_ind = [col_ind, col_ind + 1]
        if panel_class is None:
            panel_class = Panel
        elif not issubclass(panel_class, Panel):
            raise TypeError

        args = [self.gs[row_ind[0]:row_ind[1], col_ind[0]:col_ind[1]]]
        panel = panel_class(*args, label=label, **kwargs)

        # panel.add_subplot(self.gs[row_ind[0]:row_ind[1], col_ind[0]:col_ind[1]], major=True, **kwargs)

        if index is None:
            index = len(self.panels.keys()) + 1
        elif index in self.panels.keys():
            raise ValueError('The panel index has been occupied. Change to a new one!')
        self.panels[index] = panel
        return index

    def remove_panel(self, index):
        """
        Remove a panel

        :param index: The panel index.
        :type index: int or str.
        """

        self.panels[index].remove()
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

        self.canvas.add_text(x_new, y_new, text, **kwargs)

    def add_title(self, x=None, y=None, title=None, **kwargs):
        # add text in dashboard cs
        kwargs.setdefault('fontsize', self._default_dashboard_fontsize)
        kwargs.setdefault('ha', 'center')
        kwargs.setdefault('va', 'bottom')
        if x is None:
            x = 0.5
        if y is None:
            y = 1.05

        self.add_text(x, y, title, **kwargs)

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

        kwargs.setdefault('ha', 'left')  # horizontal alignment
        kwargs.setdefault('va', 'center')  # vertical alignment

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
            y_new = 1 - pos_0.height / pos_1.height + y * pos_0.height / pos_1.height
            panel.add_label(x, y_new, label, **kwargs)

    def add_axes(self, rect, label=None, **kwargs):
        if label is None:
            label = len(self.extra_axes.keys())
            label_str = 'ax_' + str(label)
        else:
            label_str = label
        kwargs.setdefault('facecolor', 'none')
        ax = self.canvas.add_axes(rect, label=label_str, **kwargs)
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

    @property
    def canvas(self):
        if self._canvas_ref is None:
            return None
        else:
            return self._canvas_ref()

    @canvas.setter
    def canvas(self, canvas_obj):
        if canvas_obj is None:
            self._canvas_ref = None
            return

        if issubclass(canvas_obj.__class__, Canvas):
            self._canvas_ref = weakref.ref(canvas_obj)
        else:
            raise TypeError


class Panel(object):
    def __init__(self, *args, figure=None, from_subplot=True, **kwargs):
        if figure is None:
            figure = plt.gcf()
        self.figure = figure
        self.axes = {}
        self.label = kwargs.pop('label', None)
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

    def __call__(self, ax_key=None):
        if ax_key is None:
            ax_key = 'major'
        return self.axes[ax_key]

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
        ax.text(x, y, label, **kwargs)

    def add_title(self, x=0.5, y=1.02, title=None, **kwargs):
        kwargs.setdefault('ha', 'center')
        kwargs.setdefault('va', 'baseline')
        self.axes['major'].set_title(x, y, title, **kwargs)

    @check_panel_ax
    def plot(self, *args, ax=None, **kwargs):
        # plot_type="1P"
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

    @check_panel_ax
    def add_lines(self, xs, ys, ax=None, **kwargs):
        pass

    

class Watermark(StrBase):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in)
        style = kwargs.pop('style', None)
        if style is None:
            style = 'Single Box'
        obj.style = style
        return obj