""" Initial configuration for the viewer toolbox
"""

from geospacelab.toolbox import logger


class Visual(object):
    def __init__(self, **kwargs):
        self.plot_type = kwargs.pop('plot_type', None)
        self.plot_config = kwargs.pop('plot_config', **kwargs)
        self.color = kwargs.pop('color', None)
        self.visible = kwargs.pop('visible', None)
        self.x_axis = AxisAttributes(axis='x', **kwargs)
        self.y_axis = AxisAttributes(axis='y', **kwargs)
        self.z_axis = AxisAttributes(axis='z', **kwargs)


class AxisAttributes(object):
    def __init__(self, axis=None, **kwargs):
        if axis is None:
            logger.StreamLogger.error("The axis must be declared, e.g., x, y, or z!")

        self.data = kwargs.pop(axis + '_data', None)
        self.data_scale = kwargs.pop(axis + '_data_scale', None)
        self.data_resolution = kwargs.pop(axis + '_data_resolution', None)
        self.error = kwargs.pop(axis + '_error', None)
        self.lim = kwargs.pop(axis + '_lim', None)
        self.label = kwargs.pop(axis + '_label', None)
        self.unit = kwargs.pop(axis + '_unit', None)
        self.scale = kwargs.pop(axis + '_scale', None)
        self.ticks = kwargs.pop(axis + '_ticks', None)
        self.tick_labels = kwargs.pop(axis + '_tick_labels', None)

