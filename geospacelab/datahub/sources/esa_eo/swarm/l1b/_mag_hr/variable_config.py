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

import numpy as np

database = 'ESA/EarthOnline'

timestamps = {
    'SC_DATETIME': 'SC_DATETIME',
}


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

depend_0 = {'UT': 'SC_DATETIME',
            'GEO_LAT': 'SC_GEO_LAT', 'GEO_LON': 'SC_GEO_LON',
            'AACGM_LAT': 'SC_AACGM_LAT', 'AACGM_LON': 'SC_AACGM_LON', 'AACGM_MLT': 'SC_AACGM_MLT'}
# depend_c = {'SPECTRA': 'EMISSION_SPECTRA'}

####################################################################################################################
var_name = 'B_N'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Magnetic field (Northward)'
var.label = r'$B_N$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'$B$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
# axis[1].lim = [np.nan, np.nan]
axis[1].label = '@v.group'
axis[1].unit = ''
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'B_E'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Magnetic field (Eastward)'
var.label = r'$B_E$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'$B$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
# axis[1].lim = [np.nan, np.nan]
axis[1].label = '@v.group'
axis[1].unit = ''
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'B_C'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Magnetic field (Radial)'
var.label = r'$B_C$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'$B$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
# axis[1].lim = [np.nan, np.nan]
axis[1].label = '@v.group'
axis[1].unit = ''
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'


configured_variables[var_name] = var

####################################################################################################################
var_name = 'FLAG_B'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag bits indicating data quality or properties'
var.label = r'B Flag'
var.unit = ''
var.group = r''
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
# axis[1].lim = [-2000, 2000]
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

var_name = 'FLAG_q'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag bits indicating data quality or properties'
var.label = r'q Flag'
var.unit = ''
var.group = r''
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
# axis[1].lim = [-2000, 2000]
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

var_name = 'FLAG_Platform'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag bits indicating data quality or properties'
var.label = r'Platform Flag'
var.unit = ''
var.group = r''
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
# axis[1].lim = [-2000, 2000]
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var