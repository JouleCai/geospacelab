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


default_colormap = "binary"

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
            # 'AACGM_LAT': 'SC_AACGM_LAT', 'AACGM_LON': 'SC_AACGM_LON', 'AACGM_MLT': 'SC_AACGM_MLT',
            'APEX_LAT': 'SC_APEX_LAT', 'APEX_LON': 'SC_APEX_LON', 'APEX_MLT': 'SC_APEX_MLT',
            }

depends_1_FLAG_1 = {'Binary Index': 'FLAG_1_BIN_IND'}
depends_1_FLAG_2 = {'Binary Index': 'FLAG_2_BIN_IND'}
# depend_c = {'SPECTRA': 'EMISSION_SPECTRA'}


####################################################################################################################
var_name = 'SC_DATETIME'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Time of observation'
var.label = r't'
var.unit = 'UT'
var.unit_label = r'UT'
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
# axis[1].lim = [np.nan, np.nan]
axis[1].label = '@v.group'
axis[1].unit = ''
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'SC_GEO_LAT'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Latitude'
var.label = r'GLAT'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Latitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-90, 90]
axis[1].ticks = np.arange(-90, 91, 30)
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'SC_GEO_LON'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Longitude'
var.label = r'GLON'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Longitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 360]
axis[1].ticks = np.arange(0, 361, 90)
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'  
configured_variables[var_name] = var

####################################################################################################################
var_name = 'SC_GEO_LST'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'S/C geographic local solar time'
var.label = r'LST'
var.unit = 'hour'
var.unit_label = r'h'
var.group = r'LST'  
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 24]
axis[1].ticks = np.arange(0, 25, 6)
axis[1].label = '@v.label'
axis[1].unit = '@v.unit_label'  
configured_variables[var_name] = var

####################################################################################################################
var_name = 'v_SC_ITRF'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'S/C velocity in ITRF'
var.label = r'v$_{SC}$'
var.unit = 'm/s'
var.unit_label = r'm/s'
var.group = r'Velocity'
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
axis[1].label = '@v.label'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'n_i'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Ion density'
var.label = r'$n_i$'
var.unit = 'cm$^{-3}$'
var.unit_label = r'cm$^{-3}$'
var.group = r'Plasma Density'
var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1E'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-10, None]
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

#####################################################################################################################
var_name = 'CALIB_n_i'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Calibration parameter for n_i, wrt. ISRs'
var.label = r'Calib. $n_i$'
var.unit = 'cm$^{-3}$'
var.unit_label = r'cm$^{-3}$'
var.group = r'Plasma Density'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [None, None]
axis[1].label = '@v.label'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'n_e'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Electron density'
var.label = r'$n_e$'
var.unit = 'cm$^{-3}$'
var.unit_label = r'cm$^{-3}$'
var.group = r'Plasma Density'
var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1E'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-10, None]
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

#####################################################################################################################
var_name = 'T_e'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Electron temperature'
var.label = r'$T_e$'
var.unit = 'K'
var.unit_label = r'K'
var.group = r'Plasma Temperature'
var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1E'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-10, None]
axis[1].label = '@v.label'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

#####################################################################################################################
var_name = 'CALIB_T_e'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Calibration parameter for T_e, wrt. ISRs'
var.label = r'Calib. $T_e$'
var.unit = 'K'
var.unit_label = r'K'
var.group = r'Plasma Temperature'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [None, None]
axis[1].label = '@v.label'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'V_SC'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'S/C potential'
var.label = r'$V_{SC}$'
var.unit = 'V'
var.unit_label = r'V'
var.group = r'S/C Potential'
var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1E'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-10, 10]
axis[1].label = '@v.label'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'FLAG_LP'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag of source: 1, 5, 9'
var.label = r'FLAG LP'
var.unit = ''
var.unit_label = ''
var.group = r'FLAG'  
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 50]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'  
configured_variables[var_name] = var

####################################################################################################################
var_name = 'FLAG_n_i'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag of n_i: 10 - 40'
var.label = r'FLAG n_i'
var.unit = ''
var.unit_label = ''
var.group = r'FLAG'  
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 50]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'  
configured_variables[var_name] = var

####################################################################################################################
var_name = 'FLAG_n_e'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag of n_e: 10 - 40'
var.label = r'FLAG $n_e$'
var.unit = ''
var.unit_label = ''
var.group = r'FLAG'  
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 50]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'  
configured_variables[var_name] = var

####################################################################################################################
var_name = 'FLAG_T_e'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag of T_e: 10 - 40'
var.label = r'FLAG $T_e$'
var.unit = ''
var.unit_label = ''
var.group = r'FLAG'  
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 50]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'  
configured_variables[var_name] = var

####################################################################################################################
var_name = 'FLAG_V_SC'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag of V_SC: 20, 30'
var.label = r'FLAG $V_{SC}$'
var.unit = ''  
var.unit_label = ''
var.group = r'FLAG'  
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 50]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'  
configured_variables[var_name] = var

####################################################################################################################
var_name = 'FLAG_BITS_1'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Binary flag (Instrument)'
var.label = r'FLAG Instrument'
var.unit = ''
var.unit_label = ''
var.group = r'FLAG'  
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
# axis[1].lim = [0, 2**16]
axis[1].label = '@v.label'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
configured_variables[var_name] = var 

####################################################################################################################
var_name = 'FLAG_BITS_2'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Binary flag (Gain)'
var.label = r'FLAG Gain'
var.unit = ''
var.unit_label = ''
var.group = r'FLAG'  
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
# axis[1].lim = [0, 2**16]
axis[1].label = '@v.label'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
configured_variables[var_name] = var

#####################################################################################################################
var_name = 'FLAG_1_BIN_AUX'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Binary flag (Instrument)'
var.label = r'FLAG Instrument'
var.unit = ''
var.unit_label = ''
var.group = r'FLAG'  
# var.error = var_name + '_err'
var.depends = {
    0: depend_0,
    1: depends_1_FLAG_1
    }
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.FLAG_1_BIN_IND"
axis[1].lim = [-0.5, 32.5]
axis[1].label = 'Binary Index'
axis[1].unit = ''
axis[2].value = "@v.value"
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
axis[2].lim = [0, 1]
configured_variables[var_name] = var

#####################################################################################################################
var_name = 'FLAG_2_BIN_AUX'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Binary flag (Gain)'
var.label = r'FLAG Gain'
var.unit = ''
var.unit_label = ''
var.group = r'FLAG'  
# var.error = var_name + '_err'
var.depends = {
    0: depend_0,
    1: depends_1_FLAG_2
    }
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.FLAG_2_BIN_IND"
axis[1].lim = [-0.5, 16.5]
axis[1].label = 'Binary Index'
axis[1].unit = ''
axis[2].value = "@v.value"
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
axis[2].lim = [0, 1]
configured_variables[var_name] = var    
