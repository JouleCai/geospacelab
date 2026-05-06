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


default_colormap = "turbo"

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

depend_0 = {'UT': 'DATETIME',
            'GEO_LAT': 'GEO_LAT', 'GEO_LON': 'GEO_LON',
            'QD_LAT': 'QD_LAT', 'QD_LON': 'QD_LON', 'QD_MLT': 'QD_MLT',
            # 'AACGM_LAT': 'SC_AACGM_LAT', 'AACGM_LON': 'SC_AACGM_LON', 'AACGM_MLT': 'SC_AACGM_MLT',
            'APEX_LAT': 'APEX_LAT', 'APEX_LON': 'APEX_LON', 'APEX_MLT': 'APEX_MLT',
            }

depend_1_Q = {'IND': 'QUALITY_IND'}
depend_1_ID = {'IND': 'ID_IND'}


# depend_c = {'SPECTRA': 'EMISSION_SPECTRA'}


####################################################################################################################
var_name = 'DATETIME'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Time of observation of MIT minimum'
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
var_name = 'GEO_LAT'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Latitude of MIT minimum'
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
var_name = 'GEO_LON'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Longitude of MIT minimum'
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
var_name = 'GEO_ALT'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Altitude of MIT minimum'
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
var_name = 'SZA'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Solar Zenith Angle of MIT minimum'
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


###################################################################################################
var_name = 'QD_LAT'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Latitude of MIT minimum'
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
var_name = 'QD_LON'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Longitude of MIT minimum'
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
var_name = 'QD_MLT'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Magnetic Local Time of MIT minimum'
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

#########################################################################################################
var_name = 'L'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual) 
# set variable attrs
var.fullname = 'L-shell of MIT minimum'
var.label = r'L'
var.unit = 'Earth radii'
var.unit_label = r'$R_E$'
var.group = r'L-shell'
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
var_name = 'dL'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Change in L-shell across the MIT'
var.label = r'dL'
var.unit = r'$R_E$'
var.unit_label = r'$R_E$'
var.group = r'dL'
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
var_name = 'Sigma'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Standard deviation of the linear fit of S at the boundary'
var.label = r'Sigma'
var.unit = r''
var.unit_label = r''
var.group = r'Sigma'
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
var_name = 'PPI'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Midnight Plasmapause index'
var.label = r'PPI'
var.unit = r'R_E'
var.unit_label = r'$R_E$'
var.group = r'PPI'
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
var_name = 'QUALITY'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quality of MIT minimum detection'
var.label = r'QUALITY'
var.unit = r''
var.unit_label = r''
var.group = r'QUALITY'
# var.error = var_name + '_err'
var.depends = {0: depend_0, 1: depend_1_Q}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
plot_config.pcolormesh.update(cmap='hot')
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.QUALITY_IND"
axis[1].lim = [-0.5, 8.5]
axis[1].label = 'Index'
axis[1].ticks = np.arange(0, 9, 2)
axis[2].value = "@v.value"
axis[2].label = '@v.label'
axis[2].lim = [-1, 3]
axis[2].ticks = np.arange(-1, 4, 1)
configured_variables[var_name] = var

#############################################################################################################################
var_name = "GEO_LAT_ID"
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Latitude of each data point in the MIT detection'
var.label = r'GLAT'
var.unit = r'deg'
var.unit_label = r'$^\circ$'
var.group = r'Latitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0, 1: depend_1_ID}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.ID_IND"
axis[1].lim = [-0.5, 3.5]
axis[1].label = 'Index'
axis[1].ticks = np.arange(0, 4, 1)
axis[2].value = "@v.value"
axis[2].label = '@v.label'
axis[2].lim = [-90, 90]
axis[2].ticks = np.arange(-90, 91, 30)
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

#####################################################################################################################
var_name = "GEO_LON_ID"
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Geographic Longitude of each data point in the MIT detection'
var.label = r'GLON'
var.unit = r'deg'
var.unit_label = r'$^\circ$'
var.group = r'Longitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0, 1: depend_1_ID}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
plot_config.pcolormesh.update(cmap='twilight')
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.ID_IND"
axis[1].lim = [-0.5, 3.5]
axis[1].label = 'Index'
axis[1].ticks = np.arange(0, 4, 1)
axis[2].value = "@v.value"
axis[2].label = '@v.label'
axis[2].lim = [0, 360]
axis[2].ticks = np.arange(0, 361, 90)
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

#################################################################################################################
var_name = "QD_LAT_ID"
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Latitude of each data point in the MIT detection'
var.label = r'QD LAT'
var.unit = r'deg'
var.unit_label = r'$^\circ$'
var.group = r'Latitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0, 1: depend_1_ID}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.ID_IND"
axis[1].lim = [-0.5, 3.5]
axis[1].label = 'Index'
axis[1].ticks = np.arange(0, 4, 1)
axis[2].value = "@v.value"
axis[2].label = '@v.label'
axis[2].lim = [-90, 90]
axis[2].ticks = np.arange(-90, 91, 30)
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var    

##################################################################################################################
var_name = "QD_LON_ID"
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Quasi-Dipole Longitude of each data point in the MIT detection'
var.label = r'QD LON'
var.unit = r'deg'
var.unit_label = r'$^\circ$'
var.group = r'Longitude'
# var.error = var_name + '_err'
var.depends = {0: depend_0, 1: depend_1_ID}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
plot_config.pcolormesh.update(cmap='twilight')
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.ID_IND"
axis[1].lim = [-0.5, 3.5]
axis[1].label = 'Index'
axis[1].ticks = np.arange(0, 4, 1)
axis[2].value = "@v.value"
axis[2].label = '@v.label'
axis[2].lim = [0, 360]
axis[2].ticks = np.arange(0, 361, 90)
axis[2].unit = '@v.unit_label'
configured_variables[var_name] = var

##################################################################################################################3
var_name = "L_ID"
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'L-shell of each data point in the MIT detection'
var.label = r'L'
var.unit = r'$R_E$'
var.unit_label = r'$R_E$'
var.group = r'L-shell'
# var.error = var_name + '_err'
var.depends = {0: depend_0, 1: depend_1_ID}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.ID_IND"
axis[1].lim = [-0.5, 3.5]
axis[1].label = 'Index'
axis[1].ticks = np.arange(0, 4, 1)
axis[2].value = "@v.value"
axis[2].label = '@v.label'
axis[2].lim = [None, None]
configured_variables[var_name] = var



############################################################################################################
var_name = "Position_Quality_ID"
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Position Quality Index of each data point in the MIT detection'
var.label = r'Position Quality'
var.unit = r''
var.unit_label = r''
var.group = r'QUALITY'
# var.error = var_name + '_err'
var.depends = {0: depend_0, 1: depend_1_ID}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
plot_config.pcolormesh.update(cmap='hot')
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.ID_IND"
axis[1].lim = [-0.5, 3.5]
axis[1].label = 'Index'
axis[1].ticks = np.arange(0, 4, 1)
axis[2].value = "@d.ID_IND"
axis[2].label = 'Position Quality'
axis[2].lim = [-1, 3]
axis[2].ticks = np.arange(-1, 4, 1)
configured_variables[var_name] = var
