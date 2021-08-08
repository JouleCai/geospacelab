import matplotlib.pyplot as plt


def check_panel_ax(func):
    def wrapper(*args, **kwargs):
        obj = args[0]
        kwargs.setdefault('ax', None)
        if kwargs['ax'] is None:
            kwargs['ax'] = obj.axex['major']
        result = func(*args, **kwargs)
        return result
    return wrapper


class Panel(object):
    def __init__(self, figure=None, **kwargs):
        if figure is None:
            figure = plt.gcf()
        self.figure = figure
        self.axes = {}
        self.label = kwargs.pop('label', None)
        # self.objectives = kwargs.pop('objectives', {})

    def add_subplot(self, *args, major=False, label=None, **kwargs):
        if major:
            label = 'major'
        else:
            if label is None:
                label = len(self.axes.keys())

        ax = self.figure.add_subplot(*args, **kwargs)
        self.axes[label] = ax
        return ax

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

    def add_label(self, x, y, label, ha='left', va='center', **kwargs):
        ax = self.axes['major']
        if label is None:
            label = ''
        transform = kwargs.pop('transform', ax.transAxes)
        ax.text(x, y, label, transform=transform, ha=ha, va=va, **kwargs)

    def add_title(self, *args, **kwargs):
        self.axes['major'].set_title(*args, **kwargs)
