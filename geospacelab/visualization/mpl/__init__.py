# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

from geospacelab.visualization.mpl.__base__ import FigureBase
from geospacelab.visualization.mpl.dashboards import Dashboard
from geospacelab.visualization.mpl.panels import Panel

def create_figure(*args, watermark=None, watermark_style=None, **kwargs):
    return FigureBase(*args, watermark=None, watermark_style=None, **kwargs)

