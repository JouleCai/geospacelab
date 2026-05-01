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

default_colormap = "turbo"

default_plot_config_EEJ_PEAK = {
    'line':         {
        'linestyle':        '',
        'linewidth':        1.5,
        'marker':           'o',
        'markersize':       6,
        'markerfacecolor':  'r',
        'markeredgecolor':  'r',
    },
    'pcolormesh':   {
        'cmap':            default_colormap,
    }
}
default_plot_config_EEJ_EB = {
    'line':         {
        'linestyle':        '',
        'linewidth':        1.5,
        'marker':           '+',
        'markersize':       6,
        'markerfacecolor':  'r',
        'markeredgecolor':  'r',
    },
    'pcolormesh':   {
        'cmap':            default_colormap,
    }
}
default_plot_config_EEJ_PB = {
    'line':         {
        'linestyle':        '',
        'linewidth':        1.5,
        'marker':           'x',
        'markersize':       6,
        'markerfacecolor':  'r',
        'markeredgecolor':  'r',
    },
    'pcolormesh':   {
        'cmap':            default_colormap,
    }
}
default_plot_config_EEJ = {
    'line':         {
        'linestyle':        '',
        'linewidth':        1.5,
        'marker':           '.',
        'markersize':       6,
        'color':            'r',
    },
    'pcolormesh':   {
        'cmap':            default_colormap,
    }
}


default_plot_config_WEJ_PEAK = {
    'line':         {
        'linestyle':        '',
        'linewidth':        1.5,
        'marker':           'o',
        'markersize':       6,
        'markerfacecolor':  'b',
        'markeredgecolor':  'b',
    },
    'pcolormesh':   {
        'cmap':            default_colormap,
    }
}
default_plot_config_WEJ_EB = {
    'line':         {
        'linestyle':        '',
        'linewidth':        1.5,
        'marker':           '+',
        'markersize':       6,
        'markerfacecolor':  'b',
        'markeredgecolor':  'b',
    },
    'pcolormesh':   {
        'cmap':            default_colormap,
    }
}
default_plot_config_WEJ_PB = {
    'line':         {
        'linestyle':        '',
        'linewidth':        1.5,
        'marker':           'x',
        'markersize':       6,
        'markerfacecolor':  'b',
        'markeredgecolor':  'b',
    },
    'pcolormesh':   {
        'cmap':            default_colormap,
    }
}
default_plot_config_WEJ = {
    'line':         {
        'linestyle':        '',
        'linewidth':        1.5,
        'marker':           '.',
        'markersize':       6,
        'color':            'b',
    },
    'pcolormesh':   {
        'cmap':            default_colormap,
    }
}

configured_variables = {}
visual = 'on'

depend_0_WEJ_PEAK = {
    'UT': 'DATETIME_WEJ_PEAK',
    'GEO_LAT': 'GEO_LAT_WEJ_PEAK', 
    'GEO_LON': 'GEO_LON_WEJ_PEAK', 
    'GEO_ALT': 'GEO_ALT_WEJ_PEAK', 
    'GEO_r': 'GEO_r_WEJ_PEAK',
    'AACGM_LAT': 'AACGM_LAT_WEJ_PEAK', 
    'AACGM_LON': 'AACGM_LON_WEJ_PEAK', 
    'AACGM_MLT': 'AACGM_MLT_WEJ_PEAK',
    'QD_LAT': 'QD_LAT_WEJ_PEAK', 
    'QD_LON': 'QD_LON_WEJ_PEAK', 
    'QD_MLT': 'QD_MLT_WEJ_PEAK',
    'APEX_LAT': 'APEX_LAT_WEJ_PEAK', 
    'APEX_LON': 'APEX_LON_WEJ_PEAK', 
    'APEX_MLT': 'APEX_MLT_WEJ_PEAK',
    }
depend_0_WEJ_EB = {
    'UT': 'DATETIME_WEJ_EB',
    'GEO_LAT': 'GEO_LAT_WEJ_EB', 
    'GEO_LON': 'GEO_LON_WEJ_EB', 
    'GEO_ALT': 'GEO_ALT_WEJ_EB', 
    'GEO_r': 'GEO_r_WEJ_EB',
    'AACGM_LAT': 'AACGM_LAT_WEJ_EB', 
    'AACGM_LON': 'AACGM_LON_WEJ_EB', 
    'AACGM_MLT': 'AACGM_MLT_WEJ_EB',
    'QD_LAT': 'QD_LAT_WEJ_EB', 
    'QD_LON': 'QD_LON_WEJ_EB', 
    'QD_MLT': 'QD_MLT_WEJ_EB',
    'APEX_LAT': 'APEX_LAT_WEJ_EB', 
    'APEX_LON': 'APEX_LON_WEJ_EB', 
    'APEX_MLT': 'APEX_MLT_WEJ_EB',
    }
depend_0_WEJ_PB = {
    'UT': 'DATETIME_WEJ_PB',
    'GEO_LAT': 'GEO_LAT_WEJ_PB', 
    'GEO_LON': 'GEO_LON_WEJ_PB', 
    'GEO_ALT': 'GEO_ALT_WEJ_PB', 
    'GEO_r': 'GEO_r_WEJ_PB',
    'AACGM_LAT': 'AACGM_LAT_WEJ_PB', 
    'AACGM_LON': 'AACGM_LON_WEJ_PB', 
    'AACGM_MLT': 'AACGM_MLT_WEJ_PB',
    'QD_LAT': 'QD_LAT_WEJ_PB', 
    'QD_LON': 'QD_LON_WEJ_PB',
    'QD_MLT': 'QD_MLT_WEJ_PB',
    'APEX_LAT': 'APEX_LAT_WEJ_PB', 
    'APEX_LON': 'APEX_LON_WEJ_PB', 
    'APEX_MLT': 'APEX_MLT_WEJ_PB',
    }
depend_0_EEJ_PEAK = {
    'UT': 'DATETIME_EEJ_PEAK',
    'GEO_LAT': 'GEO_LAT_EEJ_PEAK', 
    'GEO_LON': 'GEO_LON_EEJ_PEAK', 
    'GEO_ALT': 'GEO_ALT_EEJ_PEAK', 
    'GEO_r': 'GEO_r_EEJ_PEAK',
    'AACGM_LAT': 'AACGM_LAT_EEJ_PEAK', 
    'AACGM_LON': 'AACGM_LON_EEJ_PEAK', 
    'AACGM_MLT': 'AACGM_MLT_EEJ_PEAK',
    'QD_LAT': 'QD_LAT_EEJ_PEAK', 
    'QD_LON': 'QD_LON_EEJ_PEAK', 
    'QD_MLT': 'QD_MLT_EEJ_PEAK',
    'APEX_LAT': 'APEX_LAT_EEJ_PEAK', 
    'APEX_LON': 'APEX_LON_EEJ_PEAK', 
    'APEX_MLT': 'APEX_MLT_EEJ_PEAK',
    }
depend_0_EEJ_EB = {
    'UT': 'DATETIME_EEJ_EB',
    'GEO_LAT': 'GEO_LAT_EEJ_EB', 
    'GEO_LON': 'GEO_LON_EEJ_EB', 
    'GEO_ALT': 'GEO_ALT_EEJ_EB', 
    'GEO_r': 'GEO_r_EEJ_EB',
    'AACGM_LAT': 'AACGM_LAT_EEJ_EB', 
    'AACGM_LON': 'AACGM_LON_EEJ_EB', 
    'AACGM_MLT': 'AACGM_MLT_EEJ_EB',
    'QD_LAT': 'QD_LAT_EEJ_EB', 
    'QD_LON': 'QD_LON_EEJ_EB', 
    'QD_MLT': 'QD_MLT_EEJ_EB',
    'APEX_LAT': 'APEX_LAT_EEJ_EB', 
    'APEX_LON': 'APEX_LON_EEJ_EB', 
    'APEX_MLT': 'APEX_MLT_EEJ_EB',
    }
depend_0_EEJ_PB = {
    'UT': 'DATETIME_EEJ_PB',
    'GEO_LAT': 'GEO_LAT_EEJ_PB', 
    'GEO_LON': 'GEO_LON_EEJ_PB', 
    'GEO_ALT': 'GEO_ALT_EEJ_PB', 
    'GEO_r': 'GEO_r_EEJ_PB',
    'AACGM_LAT': 'AACGM_LAT_EEJ_PB', 
    'AACGM_LON': 'AACGM_LON_EEJ_PB', 
    'AACGM_MLT': 'AACGM_MLT_EEJ_PB',
    'QD_LAT': 'QD_LAT_EEJ_PB', 
    'QD_LON': 'QD_LON_EEJ_PB',
    'QD_MLT': 'QD_MLT_EEJ_PB',
    'APEX_LAT': 'APEX_LAT_EEJ_PB', 
    'APEX_LON': 'APEX_LON_EEJ_PB', 
    'APEX_MLT': 'APEX_MLT_EEJ_PB',
    }


####################################################################################################################
var_name = 'DATETIME_WEJ_PEAK'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Time of observation'
var.label = r't'
var.unit = 'UT'
var.unit_label = r'UT'
var.group = r''
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_PEAK}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_PEAK)
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
var_name = 'GEO_LAT_WEJ_PEAK'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Latitude'
var.label = r'GLAT WEJ PEAK'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Latitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_PEAK}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_PEAK)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-90, 90]
axis[1].ticks = np.arange(-90, 91, 30)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'GEO_LON_WEJ_PEAK'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Longitude'
var.label = r'GLON WEJ PEAK'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Longitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_PEAK}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_PEAK)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 360]
axis[1].ticks = np.arange(0, 361, 90)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

###################################################################################################################
var_name = 'QD_LAT_WEJ_PEAK'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Latitude'
var.label = r'QD_LAT WEJ PEAK'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Latitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_PEAK}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_PEAK)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-90, 90]
axis[1].ticks = np.arange(-90, 91, 30)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

###################################################################################################################
var_name = 'QD_LON_WEJ_PEAK'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Longitude'
var.label = r'QD_LON WEJ PEAK'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Longitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_PEAK}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_PEAK)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 360]
axis[1].ticks = np.arange(0, 361, 90)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

####################################################################################################################
var_name = 'QD_MLT_WEJ_PEAK'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Magnetic Local Time'
var.label = r'QD_MLT WEJ PEAK'
var.unit = 'hours'
var.unit_label = r'h'
var.group = r'MLT'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_PEAK}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_PEAK)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 24]
axis[1].ticks = np.arange(0, 25, 6)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'WEJ_PEAK'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Northward horizontal sheet current density'
var.label = r'WEJ PEAK'
var.unit = 'A/km'
var.unit_label = r'A/km'
var.group = r'Electrojet'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_PEAK}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
# axis[1].lim = [-2000, 2000]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'DATETIME_EEJ_PEAK'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Time of observation'
var.label = r't'
var.unit = 'UT'
var.unit_label = r'UT'
var.group = r''
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_PEAK}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_PEAK)
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
var_name = 'GEO_LAT_EEJ_PEAK'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Latitude'
var.label = r'GLAT EEJ PEAK'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Latitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_PEAK}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_PEAK)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-90, 90]
axis[1].ticks = np.arange(-90, 91, 30)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'GEO_LON_EEJ_PEAK'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Longitude'
var.label = r'GLON EEJ PEAK'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Longitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_PEAK}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_PEAK)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 360]
axis[1].ticks = np.arange(0, 361, 90)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

###################################################################################################################
var_name = 'QD_LAT_EEJ_PEAK'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Latitude'
var.label = r'QD_LAT EEJ PEAK'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Latitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_PEAK}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_PEAK)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-90, 90]
axis[1].ticks = np.arange(-90, 91, 30)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

###################################################################################################################
var_name = 'QD_LON_EEJ_PEAK'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Longitude'
var.label = r'QD_LON EEJ PEAK'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Longitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_PEAK}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_PEAK)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 360]
axis[1].ticks = np.arange(0, 361, 90)
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

####################################################################################################################
var_name = 'QD_MLT_EEJ_PEAK'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Magnetic Local Time'
var.label = r'QD_MLT EEJ PEAK'
var.unit = 'hours'
var.unit_label = r'h'
var.group = r'MLT'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_PEAK}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_PEAK)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 24]
axis[1].ticks = np.arange(0, 25, 6)
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'EEJ_PEAK'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Northward horizontal sheet current density'
var.label = r'EEJ PEAK'
var.unit = 'A/km'
var.unit_label = r'A/km'
var.group = r'Electrojet'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_PEAK}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
# axis[1].lim = [-2000, 2000]
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'DATETIME_WEJ_EB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Time of observation'
var.label = r't'
var.unit = 'UT'
var.unit_label = r'UT'
var.group = r''
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_EB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_EB)
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
var_name = 'GEO_LAT_WEJ_EB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Latitude'
var.label = r'GLAT WEJ EB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Latitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_EB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_EB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-90, 90]
axis[1].ticks = np.arange(-90, 91, 30)
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'GEO_LON_WEJ_EB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Longitude'
var.label = r'GLON WEJ EB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Longitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_EB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_EB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 360]
axis[1].ticks = np.arange(0, 361, 90)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

###################################################################################################################
var_name = 'QD_LAT_WEJ_EB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Latitude'
var.label = r'QD_LAT WEJ EB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Latitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_EB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_EB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-90, 90]
axis[1].ticks = np.arange(-90, 91, 30)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

###################################################################################################################
var_name = 'QD_LON_WEJ_EB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Longitude'
var.label = r'QD_LON WEJ EB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Longitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_EB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_EB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 360]
axis[1].ticks = np.arange(0, 361, 90)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

####################################################################################################################
var_name = 'QD_MLT_WEJ_EB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Magnetic Local Time'
var.label = r'QD_MLT WEJ EB'
var.unit = 'hours'
var.unit_label = r'h'
var.group = r'MLT'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_EB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_EB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 24]
axis[1].ticks = np.arange(0, 25, 6)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'DATETIME_EEJ_EB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Time of observation'
var.label = r't'
var.unit = 'UT'
var.unit_label = r'UT'
var.group = r''
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_EB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_EB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
# axis[1].lim = [np.nan, np.nan]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'GEO_LAT_EEJ_EB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Latitude'
var.label = r'GLAT EEJ EB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Latitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_EB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_EB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-90, 90]
axis[1].ticks = np.arange(-90, 91, 30)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'GEO_LON_EEJ_EB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Longitude'
var.label = r'GLON EEJ EB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Longitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_EB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_EB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 360]
axis[1].ticks = np.arange(0, 361, 90)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

###################################################################################################################
var_name = 'QD_LAT_EEJ_EB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Latitude'
var.label = r'QD_LAT EEJ EB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Latitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_EB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_EB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-90, 90]
axis[1].ticks = np.arange(-90, 91, 30)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

###################################################################################################################
var_name = 'QD_LON_EEJ_EB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Longitude'
var.label = r'QD_LON EEJ EB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Longitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_EB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_EB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 360]
axis[1].ticks = np.arange(0, 361, 90)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

####################################################################################################################
var_name = 'QD_MLT_EEJ_EB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Magnetic Local Time'
var.label = r'QD_MLT EEJ EB'
var.unit = 'hours'
var.unit_label = r'h'
var.group = r'MLT'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_EB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_EB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 24]
axis[1].ticks = np.arange(0, 25, 6)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'DATETIME_WEJ_PB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Time of observation'
var.label = r't'
var.unit = 'UT'
var.unit_label = r'UT'
var.group = r''
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_PB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_PB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
# axis[1].lim = [np.nan, np.nan]
axis[1].label = '@v.group'
axis[1].unit = ''
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'GEO_LAT_WEJ_PB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Latitude'
var.label = r'GLAT WEJ PB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Latitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_PB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_PB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-90, 90]
axis[1].ticks = np.arange(-90, 91, 30)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'GEO_LON_WEJ_PB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Longitude'
var.label = r'GLON WEJ PB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Longitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_PB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_PB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 360]
axis[1].ticks = np.arange(0, 361, 90)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

###################################################################################################################
var_name = 'QD_LAT_WEJ_PB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Latitude'
var.label = r'QD_LAT WEJ PB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Latitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_PB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_PB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-90, 90]
axis[1].ticks = np.arange(-90, 91, 30)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

###################################################################################################################
var_name = 'QD_LON_WEJ_PB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Longitude'
var.label = r'QD_LON WEJ PB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Longitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_PB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_PB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 360]
axis[1].ticks = np.arange(0, 361, 90)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

####################################################################################################################
var_name = 'QD_MLT_WEJ_PB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Magnetic Local Time'
var.label = r'QD_MLT WEJ PB'
var.unit = 'hours'
var.unit_label = r'h'
var.group = r'MLT'
# var.error = var_name + '_err'
var.depends = {0: depend_0_WEJ_PB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ_PB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 24]
axis[1].ticks = np.arange(0, 25, 6)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'DATETIME_EEJ_PB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Time of observation'
var.label = r't'
var.unit = 'UT'
var.unit_label = r'UT'
var.group = r''
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_PB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_PB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
# axis[1].lim = [np.nan, np.nan]
axis[1].label = '@v.group'
axis[1].unit = ''
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'GEO_LAT_EEJ_PB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Latitude'
var.label = r'GLAT EEJ PB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Latitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_PB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_PB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-90, 90]
axis[1].ticks = np.arange(-90, 91, 30)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'GEO_LON_EEJ_PB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Longitude'
var.label = r'GLON EEJ PB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Longitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_PB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_PB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 360]
axis[1].ticks = np.arange(0, 361, 90)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

###################################################################################################################
var_name = 'QD_LAT_EEJ_PB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Latitude'
var.label = r'QD_LAT EEJ PB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Latitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_PB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_PB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-90, 90]
axis[1].ticks = np.arange(-90, 91, 30)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

###################################################################################################################
var_name = 'QD_LON_EEJ_PB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Longitude'
var.label = r'QD_LON EEJ PB'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Longitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_PB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_PB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 360]
axis[1].ticks = np.arange(0, 361, 90)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

####################################################################################################################
var_name = 'QD_MLT_EEJ_PB'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Magnetic Local Time'
var.label = r'QD_MLT EEJ PB'
var.unit = 'hours'
var.unit_label = r'h'
var.group = r'MLT'
# var.error = var_name + '_err'
var.depends = {0: depend_0_EEJ_PB}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_EEJ_PB)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 24]
axis[1].ticks = np.arange(0, 25, 6)
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'QUALITY_FLAG'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quality flag'
var.label = r'QUALITY FLAG'
var.unit = ''
var.unit_label = ''
var.group = r'Quality FLAG'
var.depends = {0: depend_0_WEJ_PEAK}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config_WEJ)
plot_config.line.update(color='b')
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-10, None]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var
