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
from geospacelab.visualization.mpl.geomap.geopanels import PolarMapPanel

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


class GeoDashboard(mpl.Dashboard):
    def __init__(self, **kwargs):
        figure = kwargs.pop('figure', 'new')
        figure_config = kwargs.pop('figure_config', default_figure_config)

        super().__init__(visual='on', figure_config=figure_config, figure=figure, **kwargs)

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

    def add_polar_map(self, row_ind=None, col_ind=None, label=None, cs='GEO', style='lon-fixed', pole='N',
                      ut=None, lon_c=None, lst_c=None, mlt_c=None, mlon_c=None, boundary_lat=30.,
                      boundary_style='circle', mirror_south=False,
                      proj_type='Stereographic', **kwargs) -> PolarMapPanel:
        """

        :param pole:
        :param row_ind: the row
        :param style: 'lon-fixed', 'lst-fixed', 'mlt-fixed'  or 'mlon-fixed'
        :return:
        """

        panel = super().add_panel(row_ind=row_ind, col_ind=col_ind, panel_class=PolarMapPanel,
                                     label=label, cs=cs, style=style, pole=pole,
                                     ut=ut, lon_c=lon_c, lst_c=lst_c, mlt_c=mlt_c, mlon_c=mlon_c, boundary_lat=boundary_lat,
                                     boundary_style=boundary_style,
                                     mirror_south=mirror_south,
                                     proj_type=proj_type, **kwargs)
        return panel


