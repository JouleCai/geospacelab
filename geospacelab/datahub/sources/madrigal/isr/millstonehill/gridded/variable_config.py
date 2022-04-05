# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import copy
from geospacelab.datahub import VariableModel as Var
import geospacelab.visualization.mpl.colormaps as cm

database = 'Madrigal'

timestamps = {
    'DATETIME': 'DATETIME',
}

coords = {
    'GEO': ['GEO_LAT', 'GEO_LON', 'GEO_ALT', 'GEO_ST'],
    'AACGM': ['AACGM_LAT', 'AACGM_LON', 'AACGM_R', 'AACGM_MLT'],
    'APEX': ['APEX_LAT', 'APEX_LON', 'APEX_ALT', 'APEX_MLT']
}

depend_0 = {'UT': 'DATETIME'}
depend_1 = {'HEIGHT': 'HEIGHT', 'RANGE': 'RANGE', 'GEO_LAT': 'GEO_LAT', 'GEO_LON': 'GEO_LON',
            'GEO_ALT': 'GEO_ALT', 'AACGM_LAT': 'AACGM_LAT', 'AACGM_LON': 'AACGM_LON'}

default_colormap = cm.cmap_jet_modified()

default_axis_dict_2d = {
    1:     {
        'data':     '@d.HEIGHT.value',
        'lim':      [90, 500],
        'label':    'h',
        'unit':     'km',
    },
    2:  {
        'data':     '@v.value',
        'label':    '@v.label',
        'unit':     '@v.unit',
    }
}

default_plot_config = {
    'line':         {
        'linestyle':        '',
        'linewidth':        1.5,
        'marker':           '.',
        'markersize':       3,
    },
    'pcolormesh':   {
        'cmap':            default_colormap,
    }
}

configured_variables = {}
visual = 'on'

####################################################################################################################
var_name = 'n_e'
var = Var(ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'electron density',
    'label': r'$n_e$',
    'unit': 'm-3',
    'unit_label': r'm$^{-3}$',
    'error': var_name + '_err',
    'depends': {0: depend_0, 1: depend_1},
}
var.config(**var_config)
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '2P'
# set axis attrs
var.visual.axis[1].config(**default_axis_dict_2d[1])
var.visual.axis[2].config(**default_axis_dict_2d[2])
var.visual.axis[2].scale = 'log'
var.visual.axis[2].lim = [2e9, 5e12]
# axis = var.visual.axis
# axis[1].data = "@d.height.value"
# axis[2].data = "@v.value"
# axis[1].lim = [90, 350]
# axis[1].label = 'h'
# axis[1].unit = 'km'
# axis[2].lim = [8e9, 9e11]
# axis[2].scale = 'log'
# axis[2].label = '@v.label'
# axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################
var_name = 'T_i'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'ion temperature'
var.label = r'$T_i$'
var.unit = 'K'
var.error = var_name + '_err'
var.depends = {0: depend_0, 1: depend_1}
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.HEIGHT.value"
axis[2].data = "@v.value"
axis[1].lim = [90, 500]
axis[1].label = 'h'
axis[1].unit = 'km'
axis[2].lim = [100, 2500]
axis[2].scale = 'linear'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################
var_name = 'T_e'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'electron temperature'
var.label = r'$T_e$'
var.unit = 'K'
var.error = var_name + '_err'
var.depends = {0: depend_0, 1: depend_1}
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.HEIGHT.value"
axis[2].data = "@v.value"
axis[1].lim = [90, 500]
axis[1].label = 'h'
axis[1].unit = 'km'
axis[2].lim = [100, 3500]
axis[2].scale = 'linear'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################
var_name = 'v_i_UP'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Ion velocity (Upward)'
var.label = r'$v_{iUP}$'
var.unit = 'm/s'
var.error = var_name + '_err'
var.depends = {0: depend_0, 1: depend_1}
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.HEIGHT.value"
axis[2].data = "@v.value"
axis[1].lim = [90, 500]
axis[1].label = 'h'
axis[1].unit = 'km'
axis[2].lim = [-150, 150]
axis[2].scale = 'linear'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################
var_name = 'TEC'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'TEC'
var.label = 'TEC'
var.group = ''
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 20]
axis[1].label = '@v.label'
axis[1].unit = ''
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################
var_name = 'n_e_max'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Maximal electron density'
var.label = r'$n_e^{max}$'
var.group = ''
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [8e9, 2e12]
axis[1].label = '@v.label'
axis[1].scale = 'log'
axis[1].unit = ''
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var
