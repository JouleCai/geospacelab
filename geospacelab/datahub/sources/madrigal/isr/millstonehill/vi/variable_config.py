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
        'lim':      [90, 200],
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
var_name = 'v_i_N'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Ion velocity perpendicular to B (Northward)'
var.label = r'$v_{iN}$'
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
var_name = 'v_i_E'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Ion velocity perpendicular to B (Eastward)'
var.label = r'$v_{iE}$'
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
var_name = 'v_i_Z'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Ion velocity anti-parallel to B'
var.label = r'$v_{iZ}$'
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
var_name = 'E_N'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Electric field perpendicular to B (Northward)'
var.label = r'$E_{N}$'
var.unit = 'mV/m'
var.error = var_name + '_err'
var.depends = {0: depend_0, 1: depend_1}
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.HEIGHT.value"
axis[2].data = "@v.value"
axis[2].data_scale = 1000
axis[1].lim = [90, 500]
axis[1].label = 'h'
axis[1].unit = 'km'
axis[2].lim = [-10, 10]
axis[2].scale = 'linear'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################
var_name = 'E_E'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Electric field perpendicular to B (Eastward)'
var.label = r'$E_{E}$'
var.unit = 'mV/m'
var.error = var_name + '_err'
var.depends = {0: depend_0, 1: depend_1}
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.HEIGHT.value"
axis[2].data = "@v.value"
axis[2].data_scale = 1000
axis[1].lim = [90, 500]
axis[1].label = 'h'
axis[1].unit = 'km'
axis[2].lim = [-5, 5]
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
plot_config.line['alpha'] = 0.3
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 3000]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################

var_name = 'T_SYS'
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
plot_config.line['alpha'] = 0.3

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