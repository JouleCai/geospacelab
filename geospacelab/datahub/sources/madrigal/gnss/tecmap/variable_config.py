# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

from geospacelab.datahub import VariableModel as Var
import geospacelab.visualization.mpl.colormaps as cm

database = 'Madrigal'

timestamps = {
    'DATETIME': 'DATETIME',
}

depend_0 = {'UT': 'DATETIME'}
depend_1 = {'GEO_LAT': 'GEO_LAT', 'GEO_LON': 'GEO_LON', 'GEO_ALT': 'GEO_ALT'}

default_colormap = cm.cmap_jhuapl_ssj_like()

default_plot_config = {
    'line':         {
        'linestyle':        '-',
        'linewidth':        1.5,
        'marker':           '',
        'markersize':       3,
    },
    'pcolormesh':   {
        'cmap':            default_colormap,
    }
}

configured_variables = {}
visual = 'on'


####################################################################################################################
var_name = 'TEC_MAP'
var = Var(name=var_name, ndim=3, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'GNSS TEC MAP'
var.label = r'TEC'
var.group = r'TEC'
var.unit = 'TECU'
var.depends = {0: depend_0, 1: depend_1}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[2].label = '@v.label'
axis[2].unit = '@v.unit'

configured_variables[var_name] = var
