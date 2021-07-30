import matplotlib as mpl
import matplotlib.pyplot as plt

from geospacelab.visualization.mpl_toolbox.dashboard import Dashboard


def test():
    myfig = plt.figure()
    move_figure((500, 50))
    set_figure_size([15, 15])
    plt.show()


class Figure(plt.Figure):

    def __init__(self, *args, **kwargs):
        """
        custom kwargs:
        figtitle: figure title added on the top-center
        fignote: figure note added on the right conner
        figsizeunit: 'centimeters' or 'inches', call set_figsize
        figposition: (x,y) starting from the left-upper conner of the screen, call move_figure
        """

        self.title = kwargs.pop('title', None)
        self.note = kwargs.pop('note', None)
        self.figure_size = kwargs.pop('figure_size', (10, 10))
        self.figure_size_unit = kwargs.pop('figure_size_unit', 'inches')
        self.position = kwargs.pop('position', (300, 100))
        self.dashboards = {}
        super().__init__(*args, **kwargs)

        self.set_figure_size(self.figure_size, self.figure_size_unit)

    def set_figure_size(self, size=None, unit="inches"):
        self.figure_size = size
        self.figure_size_unit = unit
        set_figure_size(size=size, unit=unit)

    def add_dashboard(self, index=None, gs_num_rows=None, gs_num_cols=None, **kwargs):
        dashboard = Dashboard(gs_num_rows=gs_num_rows, gs_num_cols=gs_num_cols, **kwargs)
        if index is None:
            index = len(self.dashboards.keys()) + 1
        self.dashboards[index] = dashboard
        return index

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        if value is None:
            self._position = (300, 100)
        elif isinstance(value, tuple):
            self._position = value
            move_figure(self._position[0], self._position[1])
        else:
            raise TypeError("The position argument must be a 2-element tuple, e.g., (500, 200)!")


def set_figure_size(size=None, unit='centimeters', fig=None):
    if fig is None:
        fig = plt.gcf()
    if unit == 'centimeters':
        size[0] = size[0] / 2.54
        size[1] = size[1] / 2.54
    fig.set_size_inches(size[0], size[1], forward=True)
    return


def move_figure(x, y, fig=None):
    """Move figure's upper left corner to pixel (x, y)"""
    if fig is None:
        fig = plt.gcf()
    backend = mpl.get_backend()
    if backend == 'TkAgg':
        fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        fig.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        try:
            fig.canvas.manager.window.move(x, y)
        except:
            print('Fail to set the figure position. Backend: ' + backend)



if __name__ == "__main__":
    test()