from matplotlib.pyplot import Figure
import matplotlib as mpl
import numpy
import graphtoolbox.my_figures as fig
import graphtoolbox.set_axes_layout as axtool
import graphtoolbox.fig_utilities as figtool
import graphtoolbox.my_panels as mypanel


class MyFigure(Figure):

    def __init__(self, *args, **kwargs):
        """
        custom kwargs:
        figtitle: figure title added on the top-center
        fignote: figure note added on the right conner
        figsizeunit: 'centimeters' or 'inches', call set_figsize
        figposition: (x,y) starting from the left-upper conner of the screen, call move_figure
        """
        kwargs.setdefault('figureTitle', None)
        kwargs.setdefault('figureNote', None)
        kwargs.setdefault('figureSize', [10, 10])
        kwargs.setdefault('figureSizeUnit', "inches")
        kwargs.setdefault('figurePosition', None)
        kwargs.setdefault('kwargs_fig', dict())

        self.title = kwargs['figureTitle']
        self.note = kwargs['figureNote']
        self.figureSize = kwargs['figureSize']
        self.figureSizeUnit = kwargs['figureSizeUnit']
        self.figurePosition = kwargs['figurePosition']
        self.panels = []
        self.axesOutPanels = []
        super(MyFigure, self).__init__(*args, **kwargs['kwargs_fig'])

    def set_figure_size(self, figsize=None, unit="inches"):
        self.figureSize = figsize
        self.figureSizeUnit = unit
        figtool.set_figsize(figsize=figsize, unit=unit)

    def move_figure(self, position):
        self.figurePosition = position
        figtool.move_figure(position)

    def add_panel(self, nrows, ncols, **kwargs):
        panelObj = mypanel.MyPanel(nrows, ncols, **kwargs)
        self.panels.append(panelObj)
        panel_ind = len(self.panels) - 1
        return panel_ind

    def add_axes(self, rect, projection=None, polar=None, sharex=None, sharey=None, label=None, **kwargs):
        ax = super(MyFigure, self).add_axes(
            rect, projection=projection, polar=polar, sharex=sharex, sharey=sharey, label=label, **kwargs)
        self.axesOutPanels.append(ax)
        axes_ind = len(self.axesOutPanels) - 1
        return ax, axes_ind

    def add_text(self, x, y, s, fontdict=None, withdash=False, **kwargs):
        # add text in figure coordinates
        kwargs.setdefault('transform', self.transFigure)
        mpl.pyplot.text(x, y, s, fontdict=fontdict, withdash=withdash, **kwargs)

    def update_my_figure(self, **kwargs):
        kwargs.setdefault('figureSize', self.figureSize)
        kwargs.setdefault('figureSizeUnit', self.figureSizeUnit)
        kwargs.setdefault('figurePosition', self.figurePosition)

        self.set_figure_size(figsize=kwargs['figureSize'], unit=kwargs['figureSizeUnit'])
        self.move_figure(kwargs['figurePosition'])








