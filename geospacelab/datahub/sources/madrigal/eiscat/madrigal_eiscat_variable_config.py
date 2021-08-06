from geospacelab.datahub import VariableModel as Var

database = 'Madrigal'

timestamps = {
    'DATETIME': 'DATETIME',
}

coords = {
    'GEO': ['GEO_LAT', 'GEO_LON', 'GEO_ALT', 'GEO_ST'],
    'AACGM': ['AACGM_LAT', 'AACGM_LON', 'AACGM_R', 'AACGM_MLT'],
    'APEX': ['APEX_LAT', 'APEX_LON', 'APEX_ALT', 'APEX_MLT']
}

depend_0 = {'UT': 'DATETIME', 'TIME_1': 'DATETIME_1', 'TIME_2': 'DATETIME_2'}
depend_1 = {'height': 'height', 'range': 'range', 'GEO_LAT': 'GEO_LAT', 'GEO_LON': 'GEO_LON'}

default_colormap = "gist_ncar"

plot_config = {
    'linestyle': '-',
    'linewidth': 1.5,
    'marker': '.',
    'markersize': 3,
}


def get_variables_assigned():
    vars = {}
    visual = 'on'

    ####################################################################################################################
    var_name = 'n_e'
    var = Var(ndim=2, variable_type='scalar', visual=visual)
    # set variable attrs
    var.name = var_name
    var.fullname = 'electron density'
    var.label = r'$n_e$'
    var.unit = 'm-3'
    var.unit_label = r'm$^{-3}$'
    var.error = var_name + '_err'
    var.depends = {0: depend_0, 1: depend_1}
    # set plot attrs
    plot_config = var.visual.plot_config
    plot_config.style = '2P'
    plot_config.color = default_colormap
    # set axis attrs
    axis = var.visual.axis
    axis[1].data = "@d.height.value"
    axis[2].data = "@v.value"
    axis[1].lim = [90, 350]
    axis[1].label = 'h'
    axis[1].unit = 'km'
    axis[2].lim = [8e9, 9e11]
    axis[2].scale = 'log'
    axis[2].label = '@v.label'
    axis[2].unit = '@v.unit_label'

    vars[var_name] = var

    ####################################################################################################################
    var_name = 'T_i'
    var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
    # set variable attrs
    var.fullname = 'ion temperature'
    var.label = r'$T_i$'
    var.unit = 'K'
    var.error = var_name + '_err'
    var.depends = {0: depend_0, 1: depend_1}
    # set plot attrs
    plot_config = var.visual.plot_config
    plot_config.style = '2P'
    plot_config.color = default_colormap
    # set axis attrs
    axis = var.visual.axis
    axis[1].data = "@d.height.value"
    axis[2].data = "@v.value"
    axis[1].lim = [90, 350]
    axis[1].label = 'h'
    axis[1].unit = 'km'
    axis[2].lim = [100, 2500]
    axis[2].scale = 'linear'
    axis[2].label = '@v.label'
    axis[2].unit = '@v.unit_label'

    vars[var_name] = var

    ####################################################################################################################
    var_name = 'T_e'
    var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
    # set variable attrs
    var.fullname = 'electron temperature'
    var.label = r'$T_e$'
    var.unit = 'K'
    var.error = var_name + '_err'
    var.depends = {0: depend_0, 1: depend_1}
    # set plot attrs
    plot_config = var.visual.plot_config
    plot_config.style = '2P'
    plot_config.color = default_colormap
    # set axis attrs
    axis = var.visual.axis
    axis[1].data = "@d.height.value"
    axis[2].data = "@v.value"
    axis[1].lim = [90, 350]
    axis[1].label = 'h'
    axis[1].unit = 'km'
    axis[2].lim = [100, 3500]
    axis[2].scale = 'linear'
    axis[2].label = '@v.label'
    axis[2].unit = '@v.unit_label'

    vars[var_name] = var

    ####################################################################################################################
    var_name = 'v_i_los'
    var = Var(name=var_name, ndim=2, variable_type='scalar', visual=visual)
    # set variable attrs
    var.fullname = 'Line-of-sight ion velocity'
    var.label = r'$v_i^{los}$'
    var.unit = 'm/s'
    var.error = var_name + '_err'
    var.depends = {0: depend_0, 1: depend_1}
    # set plot attrs
    plot_config = var.visual.plot_config
    plot_config.style = '2P'
    plot_config.color = default_colormap
    # set axis attrs
    axis = var.visual.axis
    axis[1].data = "@d.height.value"
    axis[2].data = "@v.value"
    axis[1].lim = [90, 350]
    axis[1].label = 'h'
    axis[1].unit = 'km'
    axis[2].lim = [-200, 200]
    axis[2].scale = 'linear'
    axis[2].label = '@v.label'
    axis[2].unit = '@v.unit_label'

    vars[var_name] = var

    ####################################################################################################################
    var_name = 'az'
    var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
    # set variable attrs
    var.fullname = 'azimuthal angle'
    var.label = 'az'
    var.depends = {0: depend_0, 1: depend_1}
    # set plot attrs
    plot_config = var.visual.plot_config
    plot_config.style = '1noE'
    # set axis attrs
    axis = var.visual.axis
    axis[1].data = "@v.value"
    axis[1].lim = [0, 360]
    axis[1].label = '@v.group'
    axis[1].unit = ''
    axis[2].label = '@v.label'
    axis[2].unit = '@v.unit_label'

    vars[var_name] = var

    ####################################################################################################################
    var_name = 'el'
    var = Var(name=var_name, ndim=1, variable_type='scalar', visual=visual)
    # set variable attrs
    var.fullname = 'elevation angle'
    var.label = 'el'
    var.depends = {0: depend_0, 1: depend_1}
    # set plot attrs
    plot_config = var.visual.plot_config
    plot_config.style = '1noE'
    # set axis attrs
    axis = var.visual.axis
    axis[1].data = "@v.value"
    axis[1].lim = [0, 180]
    axis[1].label = '@v.group'
    axis[1].unit = ''
    axis[2].label = '@v.label'
    axis[2].unit = '@v.unit_label'

    vars[var_name] = var

    ####################################################################################################################

    return vars

