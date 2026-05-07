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



#################################################################################################################
var_name = 'B_VFM_x'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'B in x direction of VFM frame'
var.label = r'B$_{VFM,x}$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'B in VFM'
var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1E'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

#################################################################################################################
var_name = 'B_VFM_y'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'B in y direction of VFM frame'
var.label = r'B$_{VFM,y}$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'B in VFM'
var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1E'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

#################################################################################################################
var_name = 'B_VFM_z'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'B in z direction of VFM frame'
var.label = r'B$_{VFM,z}$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'B in VFM'
var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1E'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

#################################################################################################################
var_name = 'B_N'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'B in northward direction of NEC frame'
var.label = r'B$_{N}$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'B in NEC'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

#################################################################################################################
var_name = 'B_E'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'B in eastward direction of NEC frame'
var.label = r'B$_{E}$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'B in NEC'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

#################################################################################################################
var_name = 'B_C'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'B in downward direction of NEC frame'
var.label = r'B$_{C}$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'B in NEC'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var


#################################################################################################################
var_name = 'dB_Sun_VFM_x'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'dB due to Sun induced perturbation in x direction of VFM frame'
var.label = r'$\delta$B$_{Sun,x}$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'$\delta$B$_{Sun}$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

##################################################################################################################
var_name = 'dB_Sun_VFM_y'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'dB due to Sun induced perturbation in y direction of VFM frame'
var.label = r'$\delta$B$_{Sun,y}$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'$\delta$B$_{Sun}$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

##################################################################################################################
var_name = 'dB_Sun_VFM_z'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'dB due to Sun induced perturbation in z direction of VFM frame'
var.label = r'$\delta$B$_{Sun,z}$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'$\delta$B$_{Sun}$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var


#################################################################################################################
var_name = 'dB_AOCS_VFM_x'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'dB due to AOCS induced perturbation in x direction of VFM frame'
var.label = r'$\delta$B$_{AOCS,x}$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'$\delta$B$_{AOCS}$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

##################################################################################################################
var_name = 'dB_AOCS_VFM_y'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'dB due to AOCS induced perturbation in y direction of VFM frame'
var.label = r'$\delta$B$_{AOCS,y}$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'$\delta$B$_{AOCS}$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

##################################################################################################################
var_name = 'dB_AOCS_VFM_z'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'dB due to AOCS induced perturbation in z direction of VFM frame'
var.label = r'$\delta$B$_{AOCS,z}$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'$\delta$B$_{AOCS}$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var


##################################################################################################################
var_name = 'dB_other_VFM_x'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'dB due to all other sources of perturbation in x direction of VFM frame'
var.label = r'$\delta$B$_{other,x}$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'$\delta$B$_{other}$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var    


###################################################################################################################
var_name = 'dB_other_VFM_y'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'dB due to all other sources of perturbation in y direction of VFM frame'
var.label = r'$\delta$B$_{other,y}$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'$\delta$B$_{other}$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var 

###################################################################################################################
var_name = 'dB_other_VFM_z'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'dB due to all other sources of perturbation in z direction of VFM frame'
var.label = r'$\delta$B$_{other,z}$'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'$\delta$B$_{other}$'
# var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1noE'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.group'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var


# ############################################################################################################
# var_name = 'q_NEC_CRF'
# var = Var(name=var_name, ndim=4, variable_type='vector', visual=visual)
# # set variable attrs
# var.fullname = 'Attitude quaternion from NEC frame to CRF'
# var.label = r'q$_{NEC-CRF}$'
# var.unit = ''
# var.unit_label = ''
# var.group = r'q'
# configured_variables[var_name] = var

################################################################################################################
var_name = 'F'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Magnetic field intensity'
var.label = r'F'
var.unit = 'nT'
var.unit_label = r'nT'
var.group = r'F'
var.error = var_name + '_err'
var.depends = {0: depend_0}
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '1E'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@v.value"
axis[1].label = '@v.label'
axis[2].label = '@v.label'
axis[1].unit = '@v.unit_label'
configured_variables[var_name] = var

####################################################################################################################
var_name = 'FLAG_B'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag B'
var.label = r'FLAG B'
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
axis[1].lim = [None, None]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'  
configured_variables[var_name] = var

####################################################################################################################
var_name = 'FLAG_q'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag q'
var.label = r'FLAG q'
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
axis[1].lim = [None, None]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'  
configured_variables[var_name] = var

#######################################################################################################################
var_name = 'FLAG_Platform'
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag Platform'
var.label = r'FLAG Platform'
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
axis[1].lim = [None, None]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'  
configured_variables[var_name] = var

###################################################################################################################
var_name = "FLAG_F"
var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Flag F'
var.label = r'FLAG F'
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
axis[1].lim = [None, None]
axis[1].label = '@v.group'
axis[1].unit = '@v.unit_label'
axis[2].label = '@v.label'  
configured_variables[var_name] = var    


####################################################################################################################
var_name = 'FLAG_B_BIN_AUX'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Binary flag for B'
var.label = r'FLAG B'
var.unit = ''
var.unit_label = ''
var.group = r'FLAG'  
# var.error = var_name + '_err'
var.depends = {
    0: depend_0,
    1: {'Binary Index': 'FLAG_B_BIN_IND'},
    }
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.FLAG_B_BIN_IND"
axis[1].lim = [-0.5, 10.5]
axis[1].label = 'Binary Index'
axis[1].unit = ''
axis[2].value = "@v.value"
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
axis[2].lim = [0, 1]
configured_variables[var_name] = var

####################################################################################################################
var_name = 'FLAG_q_BIN_AUX'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Binary flag for q'
var.label = r'FLAG q'
var.unit = ''
var.unit_label = ''
var.group = r'FLAG'  
# var.error = var_name + '_err'
var.depends = {
    0: depend_0,
    1: {'Binary Index': 'FLAG_q_BIN_IND'},
    }
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.FLAG_q_BIN_IND"
axis[1].lim = [-0.5, 10.5]
axis[1].label = 'Binary Index'
axis[1].unit = ''
axis[2].value = "@v.value"
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
axis[2].lim = [0, 1]
configured_variables[var_name] = var

####################################################################################################################
var_name = 'FLAG_Platform_BIN_AUX'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Binary flag for Platform'
var.label = r'FLAG Platform'
var.unit = ''
var.unit_label = ''
var.group = r'FLAG'  
# var.error = var_name + '_err'
var.depends = {
    0: depend_0,
    1: {'Binary Index': 'FLAG_Platform_BIN_IND'},
    }
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.FLAG_Platform_BIN_IND"
axis[1].lim = [-0.5, 10.5]
axis[1].label = 'Binary Index'
axis[1].unit = ''
axis[2].value = "@v.value"
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
axis[2].lim = [0, 1]
configured_variables[var_name] = var

####################################################################################################################
var_name = 'FLAG_F_BIN_AUX'
var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
# set variable attrs
var.fullname = 'Binary flag for F'
var.label = r'FLAG F'
var.unit = ''
var.unit_label = ''
var.group = r'FLAG'  
# var.error = var_name + '_err'
var.depends = {
    0: depend_0,
    1: {'Binary Index': 'FLAG_F_BIN_IND'},
    }
# set plot attrs
plot_config = var.visual.plot_config
plot_config.config(**default_plot_config)
plot_config.style = '2P'
# set axis attrs
axis = var.visual.axis
axis[1].data = "@d.FLAG_F_BIN_IND"
axis[1].lim = [-0.5, 10.5]
axis[1].label = 'Binary Index'
axis[1].unit = ''
axis[2].value = "@v.value"
axis[2].label = '@v.label'
axis[2].unit = '@v.unit_label'
axis[2].lim = [0, 1]
configured_variables[var_name] = var

