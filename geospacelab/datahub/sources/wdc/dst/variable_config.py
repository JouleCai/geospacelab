import copy
import numpy as np
from geospacelab.datahub import VariableModel as Var

database = 'WDC'

timestamps = {
    'DATETIME': 'DATETIME',
}

depend_0 = {'UT': 'DATETIME'}

default_colormap = "gist_ncar"

plot_config = {
    'linestyle': '-',
    'linewidth': 1.5,
    'marker': '',
    'markersize': 3,
}

default_plot_config = {
    'line':         {
        'linestyle':        '-',
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
var_name = 'Dst'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Dst index'
var.label = r'Dst'
var.group = r'Dst'
var.unit = 'nT'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [None, None]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var
