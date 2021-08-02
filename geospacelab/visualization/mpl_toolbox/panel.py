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
    def __init__(self, **kwargs):
        self.axes = {'major': None}
        self.label = kwargs.pop('label', None)
        # self.objectives = kwargs.pop('objectives', {})

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

    def add_sub_axes(self, *args, label=None, **kwargs):
        ax = plt.gcf().add_axes(*args, **kwargs)
        ax.patch.set_alpha(0)
        if label is None:
            label = len(self.sub_axes.keys()) + 1
        self.axes[label] = ax

    def add_label(self, x, y, label, ha='left', va='center', **kwargs):
        ax = self.axes[0]
        if label is None:
            label = ''
        transform = kwargs.pop('transform', ax.transAxes)
        ax.text(x, y, label, transform=transform, ha=ha, va=va, **kwargs)

    def add_title(self, *args, **kwargs):
        self.axes[0].set_title(*args, **kwargs)
