# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import numpy as np
import copy
from geospacelab.datahub import VariableModel as Var
import geospacelab.visualization.mpl.colormaps as cm

database = 'Madrigal'

timestamps = {
    'DATETIME': 'SC_DATETIME',
}


coords = {
    'GEO': ['SC_GEO_LAT', 'SC_GEO_LON', 'SC_GEO_ALT', 'SC_GEO_ST'],
    'AACGM': ['SC_AACGM_LAT', 'SC_AACGM_LON', 'SC_AACGM_R', 'SC_AACGM_MLT'],
    'APEX': ['SC_APEX_LAT', 'SC_APEX_LON', 'SC_APEX_ALT', 'SC_APEX_MLT']
}

depend_0 = {
    'UT': 'SC_DATETIME',
    'GEO_LAT': 'SC_GEO_LAT', 'GEO_LON': 'SC_GEO_LON',
    'AACGM_LAT': 'SC_AACGM_LAT', 'AACGM_LON': 'SC_AACGM_LON', 'AACGM_MLT': 'SC_AACGM_MLT'
}


default_colormap = cm.cmap_jet_modified()

default_axis_dict_1d = {
    1: {
        'data': '@v.value',
        'label': '@v.group',
        'unit': '@v.unit_label',
        'label_pos': [-0.1, 0.5],
    },
    2: {
        'label': '@v.label'
    },
}

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
        'linestyle':        '-',
        'linewidth':        1.5,
        'marker':           '.',
        'markersize':       2,
    },
    'pcolormesh':   {
        'cmap':            default_colormap,
    }
}

configured_variables = {}
visual = 'on'

####################################################################################################################
var_name = 'v_i_H'
var = Var(ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Cross-track ion velocity (horizontal)',
    'label': r'$v_i^H$',
    'group': 'ion velocity',
    'unit': 'm/s',
    'unit_label': None,
    'error': None,
    'depends': {0: depend_0},
}
var.config(**var_config)
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '1noE'
# set axis attrs
var.visual.axis[1].config(**default_axis_dict_1d[1])
var.visual.axis[1].lim = [-np.inf, np.inf]

var.visual.axis[2].config(**default_axis_dict_1d[2])

configured_variables[var_name] = var

####################################################################################################################
var_name = 'v_i_V'
var = Var(ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Cross-track ion velocity (vertical)',
    'label': r'$v_i^V$',
    'group': 'ion velocity',
    'unit': 'm/s',
    'unit_label': None,
    'error': None,
    'depends': {0: depend_0},
}
var.config(**var_config)
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '1noE'
# set axis attrs
var.visual.axis[1].config(**default_axis_dict_1d[1])
var.visual.axis[1].lim = [-np.inf, np.inf]

var.visual.axis[2].config(**default_axis_dict_1d[2])

configured_variables[var_name] = var

####################################################################################################################
var_name = 'd_B_D'
var = Var(ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Downward magnetic field subtracted from the background',
    'label': r'$\delta B_D$',
    'group': r'$\delta B$',
    'unit': 'nT',
    'unit_label': None,
    'error': None,
    'depends': {0: depend_0},
}
var.config(**var_config)
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '1noE'
# set axis attrs
var.visual.axis[1].config(**default_axis_dict_1d[1])
var.visual.axis[1].lim = [-np.inf, np.inf]
var.visual.axis[1].data_scale = 1e9

var.visual.axis[2].config(**default_axis_dict_1d[2])

configured_variables[var_name] = var

####################################################################################################################
var_name = 'd_B_P'
var = Var(ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Cross-track magnetic field subtracted from the background',
    'label': r'$\delta B_P$',
    'group': r'$\delta B$',
    'unit': 'nT',
    'unit_label': None,
    'error': None,
    'depends': {0: depend_0},
}
var.config(**var_config)
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '1noE'
# set axis attrs
var.visual.axis[1].config(**default_axis_dict_1d[1])
var.visual.axis[1].lim = [-np.inf, np.inf]
var.visual.axis[1].data_scale = 1e9

var.visual.axis[2].config(**default_axis_dict_1d[2])

configured_variables[var_name] = var

####################################################################################################################
var_name = 'd_B_F'
var = Var(ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Forward-track magnetic field subtracted from the background',
    'label': r'$\delta B_F$',
    'group': r'$\delta B$',
    'unit': 'nT',
    'unit_label': None,
    'error': None,
    'depends': {0: depend_0},
}
var.config(**var_config)
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '1noE'
# set axis attrs
var.visual.axis[1].config(**default_axis_dict_1d[1])
var.visual.axis[1].lim = [-np.inf, np.inf]
var.visual.axis[1].data_scale = 1e9

var.visual.axis[2].config(**default_axis_dict_1d[2])

configured_variables[var_name] = var

####################################################################################################################
var_name = 'n_e'
var = Var(ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Electron density',
    'label': r'$n_e$',
    'group': 'Density',
    'unit': 'm-3',
    'unit_label': r'm$^{-3}$',
    'error': None,
    'depends': {0: depend_0},
}
var.config(**var_config)
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '1noE'
# set axis attrs
var.visual.axis[1].config(**default_axis_dict_1d[1])
var.visual.axis[1].lim = [5e8, 3e11]
var.visual.axis[1].scale = 'log'

var.visual.axis[2].config(**default_axis_dict_1d[2])

configured_variables[var_name] = var
