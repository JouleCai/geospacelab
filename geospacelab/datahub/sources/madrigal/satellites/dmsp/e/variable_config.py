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

depend_1 = {'ENERGY_CHANNEL': 'ENERGY_CHANNEL_GRID'}

default_colormap = cm.cmap_jet_modified()

default_axis_dict_1d = {
    1: {
        'data':     '@v.value',
        'label':    '@v.group',
        'unit':     '@v.unit_label',
        'label_pos': [-0.1, 0.5],
    },
    2: {
        'label':    '@v.label'
    },
}

default_axis_dict_2d = {
    1:     {
        'data':     '@d.ENERGY_CHANNEL_GRID.value',
        'lim':      [2.5e1, 4e4],
        'scale':    'log',
        'label':    'Energy',
        'unit':     'eV',
        'label_pos': [-0.1, 0.5],
    },
    2:  {
        'data':     '@v.value',
        'label':    '@v.label',
        'unit':     '@v.unit_label',
        'scale':    'log'
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
    },
}

configured_variables = {}
visual = 'on'

####################################################################################################################
var_name = 'JE_e'
var = Var(ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Integrated energy flux of electrons',
    'label': r'JE$_e$',
    'group': 'JE',
    'unit': r'eV/cm2/s/ster',
    'unit_label': r'eV$/$cm$^{2}/$s$/$ster',
    'error': None,
    'depends': {0: depend_0},
}
var.config(**var_config)
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '1noE'
# set axis attrs
var.visual.axis[1].config(**default_axis_dict_1d[1])
var.visual.axis[1].lim = [2e7, 3e13]
var.visual.axis[1].scale = 'log'

var.visual.axis[2].config(**default_axis_dict_1d[2])

configured_variables[var_name] = var

####################################################################################################################
var_name = 'JE_i'
var = Var(ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Integrated energy flux of ions',
    'label': r'JE$_i$',
    'group': 'JE',
    'unit': r'eV/cm2/s/ster',
    'unit_label': r'eV$/$cm$^{2}/$s$/$ster',
    'error': None,
    'depends': {0: depend_0},
}
var.config(**var_config)
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '1noE'
# set axis attrs
var.visual.axis[1].config(**default_axis_dict_1d[1])
var.visual.axis[1].lim = [2e7, 3e13]
var.visual.axis[1].scale = 'log'

var.visual.axis[2].config(**default_axis_dict_1d[2])

configured_variables[var_name] = var

####################################################################################################################
var_name = 'JN_e'
var = Var(ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Integrated number flux of electrons',
    'label': r'JN$_e$',
    'group': 'JN',
    'unit': r'#/cm2/s/ster',
    'unit_label': r'$\#/$cm$^{2}/$s$/$ster',
    'error': None,
    'depends': {0: depend_0},
}
var.config(**var_config)
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '1noE'
# set axis attrs
var.visual.axis[1].config(**default_axis_dict_1d[1])
var.visual.axis[1].lim = [1e4, 3e9]
var.visual.axis[1].scale = 'log'

var.visual.axis[2].config(**default_axis_dict_1d[2])

configured_variables[var_name] = var

####################################################################################################################
var_name = 'JN_i'
var = Var(ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Integrated number flux of ions',
    'label': r'JN$_e$',
    'group': 'JN',
    'unit': r'#/cm2/s/ster',
    'unit_label': r'$\#/$cm$^{2}/$s$/$ster',
    'error': None,
    'depends': {0: depend_0},
}
var.config(**var_config)
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '1noE'
# set axis attrs
var.visual.axis[1].config(**default_axis_dict_1d[1])
var.visual.axis[1].lim = [1e4, 3e9]
var.visual.axis[1].scale = 'log'

var.visual.axis[2].config(**default_axis_dict_1d[2])

configured_variables[var_name] = var

####################################################################################################################
var_name = 'E_e_MEAN'
var = Var(ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Mean energy of electrons',
    'label': r'$\bar{E}_e$',
    'group': r'$\bar{E}$',
    'unit': r'eV',
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
var.visual.axis[1].lim = [2.5e1, 4e4]
var.visual.axis[1].scale = 'log'

var.visual.axis[2].config(**default_axis_dict_1d[2])

configured_variables[var_name] = var

####################################################################################################################
var_name = 'E_i_MEAN'
var = Var(ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Mean energy of ions',
    'label': r'$\bar{E}_i$',
    'group': r'$\bar{E}$',
    'unit': r'eV',
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
var.visual.axis[1].lim = [2.5e1, 4e4]
var.visual.axis[1].scale = 'log'

var.visual.axis[2].config(**default_axis_dict_1d[2])

configured_variables[var_name] = var

####################################################################################################################
var_name = 'jE_e'
var = Var(ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Integrated number flux of electrons',
    'label': r'jE$_e$',
    'group': 'jE',
    'unit': r'eV/cm2/s/ster/Delta eV',
    'unit_label': r'eV$/$cm$^{2}/$s$/$ster$/\Delta$eV',
    'error': None,
    'depends': {0: depend_0, 1: depend_1},
}
var.config(**var_config)
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '2P'
# set axis attrs
var.visual.axis[1].config(**default_axis_dict_2d[1])

var.visual.axis[2].config(**default_axis_dict_2d[2])
var.visual.axis[2].lim = [1e5, 1e10]

configured_variables[var_name] = var

####################################################################################################################
var_name = 'jE_i'
var = Var(ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Integrated number flux of ions',
    'label': r'jE$_i$',
    'group': 'jE',
    'unit': r'eV/cm2/s/ster/Delta eV',
    'unit_label': r'eV$/$cm$^{2}/$s$/$ster$/\Delta$eV',
    'error': None,
    'depends': {0: depend_0, 1: depend_1},
}
var.config(**var_config)
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '2P'
# set axis attrs
var.visual.axis[1].config(**default_axis_dict_2d[1])

var.visual.axis[2].config(**default_axis_dict_2d[2])
var.visual.axis[2].lim = [1e3, 1e8]

configured_variables[var_name] = var

####################################################################################################################
var_name = 'jN_e'
var = Var(ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Integrated number flux of electrons',
    'label': r'jN$_e$',
    'group': 'jN',
    'unit': r'#/cm2/s/ster/Delta eV',
    'unit_label': r'$\#/$cm$^{2}/$s$/$ster$/\Delta$eV',
    'error': None,
    'depends': {0: depend_0, 1: depend_1},
}
var.config(**var_config)
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '2P'
# set axis attrs
var.visual.axis[1].config(**default_axis_dict_2d[1])

var.visual.axis[2].config(**default_axis_dict_2d[2])
var.visual.axis[2].lim = [1e4, 1e9]

configured_variables[var_name] = var

####################################################################################################################
var_name = 'jN_i'
var = Var(ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var_config = {
    'name': var_name,
    'fullname': 'Integrated number flux of ions',
    'label': r'jN$_i$',
    'group': 'jN',
    'unit': r'#/cm2/s/ster/Delta eV',
    'unit_label': r'$\#/$cm$^{2}/$s$/$ster$/\Delta$eV',
    'error': None,
    'depends': {0: depend_0, 1: depend_1},
}
var.config(**var_config)
# set plot attrs
var.visual.plot_config.config(**default_plot_config)
var.visual.plot_config.style = '2P'
# set axis attrs
var.visual.axis[1].config(**default_axis_dict_2d[1])

var.visual.axis[2].config(**default_axis_dict_2d[2])
var.visual.axis[2].lim = [1e3, 1e8]

configured_variables[var_name] = var
