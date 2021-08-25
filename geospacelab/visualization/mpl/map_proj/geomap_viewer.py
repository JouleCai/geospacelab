# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import geospacelab.visualization.mpl as mpl
import geospacelab.datahub as datahub
import geospacelab.visualization.mpl.map_proj.geopanel as geopanel

default_layout_config = {
    'left': 0.15,
    'right': 0.8,
    'bottom': 0.15,
    'top': 0.88,
    'hspace': 0.1,
    'wspace': 0.1
}

default_figure_config = {
    'figsize': (12, 12),    # (width, height)
    'dpi': 100,
}


class GeoMapViewer(datahub.DataHub, mpl.Dashboard):
    def __init__(self, **kwargs):
        new_figure = kwargs.pop('new_figure', True)
        figure_config = kwargs.pop('figure_config', default_figure_config)

        super().__init__(visual='on', figure_config=figure_config, new_figure=new_figure, **kwargs)

    def set_layout(self, num_rows=None, num_cols=None, left=None, right=None, bottom=None, top=None,
                   hspace=None, wspace=None, **kwargs):

        if left is None:
            left = default_layout_config['left']
        if right is None:
            right = default_layout_config['right']
        if bottom is None:
            bottom = default_layout_config['bottom']
        if top is None:
            top = default_layout_config['top']
        if hspace is None:
            hspace = default_layout_config['hspace']
        if hspace is None:
            wspace = default_layout_config['wspace']

        super().set_layout(num_rows=num_rows, num_cols=num_cols, left=left, right=right, bottom=bottom, top=top,
                           hspace=hspace, wspace=wspace, **kwargs)

    def add_polar_map(self, **kwargs):
        kwargs.setdefault('row_ind', None)
        kwargs.setdefault('col_ind', None)
        kwargs.setdefault('label', None)
        kwargs.setdefault('panel_class', geopanel.PolarMap)
        kwargs.setdefault('cs', 'GEO')
        kwargs.setdefault('style', 'lon-fixed')
        kwargs.setdefault('pole', 'N')
        kwargs.setdefault('ut', None)
        kwargs.setdefault('lon_c', None)
        kwargs.setdefault('lst_c', None)
        kwargs.setdefault('mlt_c', None)
        kwargs.setdefault('mlon_c', None)
        kwargs.setdefault('boundary_lat', 30.)
        kwargs.setdefault('boundary_style', 'circle')
        kwargs.setdefault('grid_lat_res', 10.)
        kwargs.setdefault('grid_lon_res', 15.)
        kwargs.setdefault('mirror_south', False)
        kwargs.setdefault('proj_type', 'Stereographic')
        super().add_panel(**kwargs)





