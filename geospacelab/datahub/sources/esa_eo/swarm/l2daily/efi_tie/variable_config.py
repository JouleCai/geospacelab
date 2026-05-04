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
            # 'QD_LAT': 'SC_QD_LAT', 'QD_LON': 'SC_QD_LON', 'QD_MLT': 'SC_QD_MLT',
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

#####################################################################################################################
var_name = 'SC_GEO_ALT'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Altitude'
var.label = r'Height'
var.unit = 'km'
var.unit_label = r'km'
var.group = r'Altitude'
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
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

######################################################################################################################
var_name = 'SC_SZA'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Solar Zenith Angle'
var.label = r'SZA'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Solar Zenith Angle'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, 180]
axis[1].ticks = np.arange(0, 181, 30)
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var    

#################################################################################################################
var_name = 'SC_SAz'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Solar Azimuth Angle'
var.label = r'SAz'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Solar Azimuth Angle'
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

#####################################################################################################################
var_name = 'SC_GEO_ST'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic True Solar Time'
var.label = r'GST'
var.unit = 'hour'
var.unit_label = r'h'
var.group = r'Geographic Solar Time'
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
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

###################################################################################################
var_name = 'SC_QD_LAT'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Latitude'
var.label = r'QD LAT'
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

####################################################################################################
var_name = 'SC_QD_LON'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Longitude'
var.label = r'QD LON'
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

#####################################################################################################
var_name = 'SC_QD_MLT'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Magnetic Local Time'
var.label = r'QD MLT'
var.unit = 'hour'
var.unit_label = r'h'
var.group = r'Magnetic Local Time'
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
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

######################################################################################################
var_name = 'SC_AACGM_LAT'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'AACGM Latitude'
var.label = r'AACGM LAT'
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

######################################################################################################
var_name = 'SC_AACGM_LON'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'AACGM Longitude'
var.label = r'AACGM LON'
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

#######################################################################################################################
var_name = 'SC_AACGM_MLT'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'AACGM Magnetic Local Time'
var.label = r'AACGM MLT'
var.unit = 'hour'
var.unit_label = r'h'
var.group = r'Magnetic Local Time'
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
var_name = 'T_i_DRIFT'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Ion temperature derived from TII drift at high latitudes'
var.label = r'$T_i^{DRIFT}$'
var.unit = 'K'
var.unit_label = r'K'
var.group = r'$T_i$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, None]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'

configured_variables[var_name] = var

###################################################################################################################
var_name = 'T_i_DRIFT_MODEL'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Ion temperature derived from Weimer 2005 model at high latitudes'
var.label = r'$T_i^{DRIFT, model}$'
var.unit = 'K'
var.unit_label = r'K'
var.group = r'$T_i$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, None]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'  
configured_variables[var_name] = var

##################################################################################################################
var_name = "T_e_LP"
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Corrected electron temperature from Langmuir probe'
var.label = r'$T_e^{LP}$'
var.unit = 'K'
var.unit_label = r'K'
var.group = r'$T_e$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, None]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var   

####################################################################################################################
var_name = 'T_n_MSIS'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Neutral temperature from MSIS model'
var.label = r'$T_n^{MSIS}$'
var.unit = 'K'
var.unit_label = r'K'
var.group = r'$T_n$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [0, None]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

########################################################################################################################
var_name = "FLAG_T_i_DRIFT"
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag for ion temperature derived from TII at high latitudes'
var.label = r'FLAG $T_i^{DRIFT}$'
var.unit = r''
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
axis[1].lim = [None, None]
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

########################################################################################################################
var_name = "FLAG_T_i_DRIFT_MODEL"
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag for ion temperature derived from Weimer 2005 model at high latitudes'
var.label = r'FLAG $T_i^{DRIFT, model}$'
var.unit = r''
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
axis[1].lim = [None, None]
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

##########################################################################################################################
var_name = "FLAG_T_i_DRIFT_BIN_AUX"
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Binary quality flag for ion temperature derived from TII at high latitudes'
var.label = r'FLAG $T_i^{DRIFT}$'
var.unit = r''
var.group = r'FLAG'
# var.error = var_name + '_err'
var.depends = {
    0: depend_0,
    1: {'Binary Index': 'FLAG_T_i_DRIFT_BIN_IND'},
    }
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.FLAG_T_i_DRIFT_BIN_IND"
axis[1].lim = [-0.5, 5.5]
axis[1].label = 'Binary Index'
axis[1].unit = ''
axis[2].value = "@v.value"
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
axis[2].lim = [0, 1]
configured_variables[var_name] = var

##########################################################################################################################
var_name = "FLAG_T_i_DRIFT_MODEL_BIN_AUX"
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Binary quality flag for ion temperature derived from Weimer 2005 model at high latitudes'
var.label = r'FLAG $T_i^{DRIFT, model}$'
var.unit = r''
var.group = r'FLAG'
# var.error = var_name + '_err'
var.depends = {
    0: depend_0,
    1: {'Binary Index': 'FLAG_T_i_DRIFT_MODEL_BIN_IND'},
    }
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.FLAG_T_i_DRIFT_MODEL_BIN_IND"
axis[1].lim = [-0.5, 5.5]
axis[1].label = 'Binary Index'
axis[1].unit = ''
axis[2].value = "@v.value"
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
axis[2].lim = [0, 1]
configured_variables[var_name] = var