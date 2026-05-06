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
            # 'QD_LAT': 'QD_LAT', 'QD_LON': 'QD_LON', 'QD_MLT': 'QD_MLT',
            # 'AACGM_LAT': 'SC_AACGM_LAT', 'AACGM_LON': 'SC_AACGM_LON', 'AACGM_MLT': 'SC_AACGM_MLT',
            # 'APEX_LAT': 'APEX_LAT', 'APEX_LON': 'APEX_LON', 'APEX_MLT': 'APEX_MLT',
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
var_name = 'Distance'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual) 
# set variable attrs
var.fullname = 'Averaged distance value from all dipoles used for the estimation of NEGIX values'
var.label = r'Distance'
var.unit = 'm'
var.unit_label = r'm'
var.group = r'Distance'
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

###########################################################################################################
var_name = 'Azimuth'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual) 
# set variable attrs
var.fullname = 'Averaged azimuth value from all dipoles used for the estimation of NEGIX values'
var.label = r'Azimuth'
var.unit = 'degrees'
var.unit_label = r'$^\circ$'
var.group = r'Azimuth'
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

#############################################################################################################
var_name = 'Tegix_X'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual) 
# set variable attrs
var.fullname = 'Mean value of TEGIX for the West-East component'
var.label = r'Tegix X'
var.unit = r'#/cm$^{3}$/m'
var.unit_label = r'#/cm$^{3}$/m'
var.group = r'Tegix X'
var.error = var_name + '_Sigma'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1E'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [None, None]
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

############################################################################################################
var_name = 'Tegix_X_Sigma'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual) 
# set variable attrs
var.fullname = 'Standard deviation value of TEGIX for the West-East component'
var.label = r'Tegix X Sigma'
var.unit = r'#/cm$^{3}$/m'
var.unit_label = r'#/cm$^{3}$/m'
var.group = r'Tegix X Sigma'
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

#############################################################################################################
var_name = 'Tegix_X_P95'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual) 
# set variable attrs
var.fullname = '95-percentile value of TEGIX for the West-East component'
var.label = r'Tegix X P95'
var.unit = r'#/cm$^{3}$/m'
var.unit_label = r'#/cm$^{3}$/m'
var.group = r'Tegix X P95'
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

############################################################################################################
var_name = 'Tegix_Y'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual) 
# set variable attrs
var.fullname = 'Mean value of TEGIX for the South-North component'
var.label = r'Tegix Y'
var.unit = r'#/cm$^{3}$/m'
var.unit_label = r'#/cm$^{3}$/m'
var.group = r'Tegix Y'
var.error = var_name + '_Sigma'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1E'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [None, None]
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var    

############################################################################################################
var_name = 'Tegix_Y_Sigma'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual) 
# set variable attrs
var.fullname = 'Standard deviation value of TEGIX for the South-North component'
var.label = r'Tegix Y Sigma'
var.unit = r'#/cm$^{3}$/m'
var.unit_label = r'#/cm$^{3}$/m'
var.group = r'Tegix Y Sigma'
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

############################################################################################################
var_name = 'Tegix_Y_P95'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual) 
# set variable attrs
var.fullname = '95-percentile value of TEGIX for the South-North component'
var.label = r'Tegix Y P95'
var.unit = r'#/cm$^{3}$/m'
var.unit_label = r'#/cm$^{3}$/m'
var.group = r'Tegix Y P95'
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

###############################################################################################################
var_name = 'Tegix_Total'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual) 
# set variable attrs
var.fullname = 'Total vector value of TEGIX, derived from the mean West-East and South-North components'
var.label = r'Tegix Total'
var.unit = r'#/cm$^{3}$/m'
var.unit_label = r'#/cm$^{3}$/m'
var.group = r'Tegix Total'
var.error = 'Tegix_Sigma'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1E'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [None, None]
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'  
configured_variables[var_name] = var

##############################################################################################################
var_name = 'Tegix_Sigma'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual) 
# set variable attrs
var.fullname = 'Vector value of the standard deviation of TEGIX, derived from the standard deviation values of West-East and South-North components'
var.label = r'Tegix Sigma'
var.unit = r'#/cm$^{3}$/m'
var.unit_label = r'#/cm$^{3}$/m'
var.group = r'Tegix Sigma'
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

##########################################################################################################
var_name = 'Tegix_P95'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual) 
# set variable attrs
var.fullname = 'Vector value of the 95-percentile of TEGIX, derived from the 95-percentile values of West-East and South-North components'
var.label = r'Tegix P95'
var.unit = r'#/cm$^{3}$/m'
var.unit_label = r'#/cm$^{3}$/m'
var.group = r'Tegix P95'
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
var_name = 'N_Measurements'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual) 
# set variable attrs
var.fullname = 'Number of elements for a data record from which TEGIX is determined'
var.label = r'N Measure.'
var.unit = '-'
var.unit_label = r'-'
var.group = r'N Measure.'
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
var_name = 'Flag_Tegix'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual) 
# set variable attrs
var.fullname = 'Flags characterising the dispersion of gradient values that are used to compute the total vector'
var.label = r'Flag Tegix'
var.unit = '-'
var.unit_label = r'-'
var.group = r'Flag Tegix'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].lim = [-1, 3]
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var    

##################################################################################################################################
var_name = 'Orbit_Label'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual) 
# set variable attrs
var.fullname = 'Value defining whether the magnitude belongs to an ascending or descending orbit of the Swarm satellites'
var.label = r'Orbit Label'
var.unit = '-'
var.unit_label = r'-'
var.group = r'Orbit Label'
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


