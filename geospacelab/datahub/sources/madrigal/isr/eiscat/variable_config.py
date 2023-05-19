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

depend_0 = {'UT': 'DATETIME', 'TIME_1': 'DATETIME_1', 'TIME_2': 'DATETIME_2'}
depend_1 = {'HEIGHT': 'HEIGHT', 'RANGE': 'RANGE', 'GEO_LAT': 'GEO_LAT', 'GEO_LON': 'GEO_LON',
            'GEO_ALT': 'GEO_ALT', 'AACGM_LAT': 'AACGM_LAT', 'AACGM_LON': 'AACGM_LON'}

default_colormap = cm.cmap_jet_modified()

default_axis_dict_2d = {
    1:     {
        'data':     '@d.HEIGHT.value',
        'lim':      [90, 350],
        'label':    'h',
        'unit':     'km',
    },
    2:  {
        'data':     '@v.value',
        'label':    '@v.label',
        'unit':     '@v.unit_label',
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
var.visual.axis[2].lim = [8e9, 9e11]
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
axis[1].lim = [90, 350]
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
axis[1].lim = [90, 350]
axis[1].label = 'h'
axis[1].unit = 'km'
axis[2].lim = [100, 3500]
axis[2].scale = 'linear'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################
var_name = 'v_i_los'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Line-of-sight ion velocity'
var.label = r'$v_i^{LOS}$'
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
axis[1].lim = [90, 350]
axis[1].label = 'h'
axis[1].unit = 'km'
axis[2].lim = [-400, 400]
axis[2].scale = 'linear'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################
var_name = 'AZ'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'azimuthal angle'
var.label = 'AZ'
var.group = 'radar param'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-5, 365]
axis[1].ticks = [0, 90, 180, 270, 360]
axis[1].label = '@v.group'
axis[1].unit = ''
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################
var_name = 'EL'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'elevation angle'
var.label = 'EL'
var.group = 'radar param'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-5, 185]
axis[1].ticks = [0, 45, 90, 135, 180]
axis[1].label = '@v.group'
axis[1].unit = ''
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################
var_name = 'P_Tx'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Transmitter Power'
var.label = 'Power'
var.unit = 'kW'
var.group = 'radar param'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
plot_config.line['linewidth'] = 2.5
plot_config.line['linestyle'] = '-'
plot_config.line['marker'] = ''
plot_config.line['alpha'] = 0.5
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 1600]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################

var_name = 'T_SYS_1'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'System temperature 1'
var.label = r'$T_{1}^{sys}$'
var.unit = 'K'
var.group = 'radar param'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.line['linewidth'] = 2.5
plot_config.line['linestyle'] = '-'
plot_config.line['marker'] = ''
plot_config.line['alpha'] = 0.5

plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 500]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################