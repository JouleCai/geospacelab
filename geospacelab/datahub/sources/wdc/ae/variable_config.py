# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import copy
import numpy as np
from geospacelab.datahub import VariableModel as Var

database = 'WDC'

timestamps = {
    'DATETIME': 'DATETIME',
}

depend_0 = {'UT': 'DATETIME'}

default_colormap = "gist_ncar"

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
var_name = 'AE'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'AE index'
var.label = r'AE'
var.group = r'AE indices'
var.unit = 'nT'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-np.inf, np.inf]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################
var_name = 'AU'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'AU index'
var.label = r'AU'
var.group = r'AE indices'
var.unit = 'nT'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-np.inf, np.inf]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################
var_name = 'AO'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'AO index'
var.label = r'AO'
var.group = r'AE indices'
var.unit = 'nT'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-np.inf, np.inf]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

####################################################################################################################
var_name = 'AL'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'AL index'
var.label = r'AL'
var.group = r'AE indices'
var.unit = 'nT'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-np.inf, np.inf]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var
