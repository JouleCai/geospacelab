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
        'unit': '@v.unit',
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
var_name = 'T_e'
var = Var(ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Electron temperature',
    'label': r'$T_e$',
    'group': 'T',
    'unit': 'K',
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
var.visual.axis[1].lim = [500, 5e3]

var.visual.axis[2].config(**default_axis_dict_1d[2])

configured_variables[var_name] = var

####################################################################################################################
var_name = 'T_i'
var = Var(ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Ion temperature',
    'label': r'$T_i$',
    'group': 'T',
    'unit': 'K',
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
var.visual.axis[1].lim = [500, 5e3]

var.visual.axis[2].config(**default_axis_dict_1d[2])

configured_variables[var_name] = var

####################################################################################################################
var_name = 'COMP_O_p'
var = Var(ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'O+ fraction',
    'label': r'$r_{O+}$',
    'unit': '%',
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
var.visual.axis[1].lim = [0, 100]
var.visual.axis[1].data_scale = 100

var.visual.axis[2].config(**default_axis_dict_1d[2])

configured_variables[var_name] = var
