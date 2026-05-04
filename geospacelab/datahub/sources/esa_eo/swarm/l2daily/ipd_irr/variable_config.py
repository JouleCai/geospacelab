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


########################################################################################################################
var_name = 'n_e'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Electron density'
var.label = r'$n_e$'
var.unit = r'cm$^{-3}$'
var.unit_label = r'cm$^{-3}$'
var.group = r'$n_e$'
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

##################################################################################################################################
var_name = 'n_e_BKG'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Background Electron density'
var.label = r'$n_e^{BKG}$'
var.unit = r'cm$^{-3}$'
var.unit_label = r'cm$^{-3}$'
var.group = r'$n_e$'
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

########################################################################################################################
var_name = 'n_e_FRG'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Foreground Electron density'
var.label = r'$n_e^{FRG}$'
var.unit = r'cm$^{-3}$'
var.unit_label = r'cm$^{-3}$'
var.group = r'$n_e$'
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

########################################################################################################################
var_name = 'T_e'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Electron temperature'
var.label = r'$T_e$'
var.unit = r'eV'
var.unit_label = r'eV'
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
axis[1].lim = [None, None]
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

#########################################################################################################################
var_name = 'FLAG_PCP'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag for polar cap patches'
var.label = r'FLAG PCP'
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
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

#########################################################################################################################
var_name = 'GRAD_n_e_100km'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Electron density gradient at 100 km scale'
var.label = r'$\nabla n_e$@100km'
var.unit = r'cm$^{-3}$/m'
var.unit_label = r'cm$^{-3}$/m'
var.group = r'$\nabla n_e$'
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

#########################################################################################################################
var_name = 'GRAD_n_e_50km'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Electron density gradient at 50 km scale'
var.label = r'$\nabla n_e$@50km'
var.unit = r'cm$^{-3}$/m'
var.unit_label = r'cm$^{-3}$/m'
var.group = r'$\nabla n_e$'
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

#########################################################################################################################
var_name = 'GRAD_n_e_20km'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Electron density gradient at 20 km scale'
var.label = r'$\nabla n_e$@20km'
var.unit = r'cm$^{-3}$/m'
var.unit_label = r'cm$^{-3}$/m'
var.group = r'$\nabla n_e$'
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

#########################################################################################################################
var_name = 'GRAD_n_e_PCP_EDGE'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Electron density gradient at polar cap patch edge'
var.label = r'$\nabla n_e$@PCP edge'
var.unit = r'cm$^{-3}$/m'
var.unit_label = r'cm$^{-3}$/m'
var.group = r'$\nabla n_e$'
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

##########################################################################################################################
var_name = 'ROD'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Rate of change of electron density'
var.label = r'ROD'
var.unit = r'cm$^{-3}$/s'
var.unit_label = r'cm$^{-3}$/s'
var.group = r'ROD'
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

###########################################################################################################################
var_name = 'RODI_10s'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Rate of change of electron density index at 10s scale'
var.label = r'RODI@10s'
var.unit = r'cm$^{-3}$/s'
var.unit_label = r'cm$^{-3}$/s'
var.group = r'RODI'
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

############################################################################################################################
var_name = 'RODI_20s'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Rate of change of electron density index at 20s scale'
var.label = r'RODI@20s'
var.unit = r'cm$^{-3}$/s'
var.unit_label = r'cm$^{-3}$/s'
var.group = r'RODI'
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

#########################################################################################################################
var_name = 'd_n_e_10s'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Difference of electron density at 10s scale'
var.label = r'$\delta n_e$@10s'
var.unit = r'cm$^{-3}$'
var.unit_label = r'cm$^{-3}$'
var.group = r'$\delta n_e$'
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

##########################################################################################################################
var_name = 'd_n_e_20s'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Difference of electron density at 20s scale'
var.label = r'$\delta n_e$@20s'
var.unit = r'cm$^{-3}$'
var.unit_label = r'cm$^{-3}$'
var.group = r'$\delta n_e$'
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

##########################################################################################################################
var_name = 'd_n_e_40s'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Difference of electron density at 40s scale'
var.label = r'$\delta n_e$@40s'
var.unit = r'cm$^{-3}$'
var.unit_label = r'cm$^{-3}$'
var.group = r'$\delta n_e$'
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

###########################################################################################################################
var_name = 'num_GPS_SATs'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Number of GPS satellites used for observation'
var.label = r'Num GPS SATs'
var.unit = r''
var.unit_label = r''
var.group = r'Num GPS SATs'
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

##########################################################################################################################
var_name = 'VTEC_MEDIAN'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Median of Vertical Total Electron Content'
var.label = r'VTEC MEDIAN'
var.unit = r'TECU'
var.unit_label = r'TECU'
var.group = r'VTEC'
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

##########################################################################################################################
var_name = 'VTEC_STD'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Standard deviation of Vertical Total Electron Content'
var.label = r'VTEC STD'
var.unit = r'TECU'
var.unit_label = r'TECU'
var.group = r'VTEC STD'
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

###########################################################################################################################
var_name = 'ROT_MEDIAN'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Median of Rate of Total Electron Content'
var.label = r'ROT MEDIAN'
var.unit = r'TECU/s'
var.unit_label = r'TECU/s'
var.group = r'ROT'
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

###########################################################################################################################
var_name = 'ROTI_10s_MEDIAN'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Median of Rate of Total Electron Content Index at 10s scale'
var.label = r'ROTI@10s MEDIAN'
var.unit = r'TECU/s'
var.unit_label = r'TECU/s'
var.group = r'ROTI'
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

###########################################################################################################################
var_name = 'ROTI_20s_MEDIAN'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Median of Rate of Total Electron Content Index at 20s scale'
var.label = r'ROTI@20s MEDIAN'
var.unit = r'TECU/s'
var.unit_label = r'TECU/s'
var.group = r'ROTI'
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

############################################################################################################################
var_name = 'IPIR_INDEX'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Ionospheric Plasma Irregularity Index'
var.label = r'IPIR INDEX'
var.unit = r''
var.unit_label = r''
var.group = r'IPIR INDEX'
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

###########################################################################################################################
var_name = 'FLAG_IBI'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag for ionospheric bubble identification'
var.label = r'FLAG IBI'
var.unit = r''
var.unit_label = r''
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
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

###########################################################################################################################
var_name = 'Ionosphere_Region'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Ionospheric region'
var.label = r'Ionos. Region'
var.unit = r''
var.unit_label = r''
var.group = r'Ionos. Region'
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

#############################################################################################################################
var_name = 'FLAG_n_e'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quality flag for electron density'
var.label = r'FLAG $n_e$'
var.unit = r''
var.unit_label = r''
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
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

