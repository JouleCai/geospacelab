# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

from geospacelab.visualization.mpl.dashboards import TSDashboard
from geospacelab.visualization.mpl.__base__ import plt, FigureBase


def mpl_viewer(figure_class=FigureBase, **kwargs):
    """
    Create a viewer based on **matplotlib**.

    :param figure_class: Optionally use a custom :class:`GeospaceLab Canvas <geospacelab.visualization.mpl._base.Canvas>
    instance.
    :type figure_class: subclass of :class:`GeospaceLab Canvas <geospacelab.visualization.mpl._base.Canvas>
    :param kwargs: Optional keyword arguments as same as in ``plt.figure``
    :return: The canvas instance
    """
    kwargs.setdefault('FigureClass', figure_class)
    fig = plt.figure(**kwargs)
    return fig


