
database = 'Madrigal'

timestamps = {
    'DATETIME': 'DATETIME',
}

coords = {
    'GEO': ['GEO_LAT', 'GEO_LON', 'GEO_ALT', 'GEO_ST'],
    'AACGM': ['AACGM_LAT', 'AACGM_LON', 'AACGM_R', 'AACGM_MLT'],
    'APEX': ['APEX_LAT', 'APEX_LON', 'APEX_ALT', 'APEX_MLT']
}

default_colormap = "gist_ncar"

plot_config = {
    'linestyle':        '-',
    'linewidth':        1.5,
    'marker':           '.',
    'markersize':       3,
}

visual_config_1 = {
    'plot_type':     '1',
    'x_data':        None,
    'x_data_res':     None,  # in seconds
    'x_label':       None,
    'y_data':        None,
    'y_label':       ('label', ),
    'y_unit':        ('unit_label', ),
    'y_lim':         None,
    'y_scale':       'linear',
    'plot_config':  dict(plot_config)
}

visual_config_2 = {
    'plot_type':     '2',
    'x_data':        None,
    'x_data_res':     None,  # in seconds
    'x_label':       None,
    'y_data':        ('height', ),
    'z_label':       tuple(['label']),
    'z_unit':        tuple(['unit_label']),
    'z_lim':         [1e8, 1e13],
    'z_scale':       'log',
    'color':        default_colormap,
    'plot_config':  dict(plot_config)
}

depend_0 = {'UT': 'DATETIME', 'TIME_1': 'DATETIME_1', 'TIME_2': 'DATETIME_2'}
depend_1 = {'height': 'height', 'range': 'range', 'GEO_LAT': 'GEO_LAT', 'GEO_LON': 'GEO_LON'}

##################################################################################
items = {}
##################################################################################
var_name = 'n_e'
visual_in = dict(visual_config_2)
visual_in['z_label'] = 'Electron density'
visual_in['z_lim'] = [8e9, 9e11]
visual_in['y_label'] = 'h'
visual_in['y_unit'] = 'km'
visual_in['y_lim'] = [90, 350]
items[var_name] = {
    'name':     var_name,
    'fullname':     'electron density',
    'label':        r'$n_e$',
    'unit':         'm-3',
    'unit_label':   r'm$^{-3}$',
    'group':        '',
    'value':        var_name,
    'error':        var_name + '_err',
    'dim':          2,
    'depends':      {0: depend_0, 1: depend_1},
    'timestamps':   timestamps,
    'positions':    coords,
    'visual_config':       visual_in
}

##################################################################################
var_name = 'T_e'
visual_in = dict(visual_config_2)
visual_in['z_label'] = 'Electron temperature'
visual_in['z_lim'] = [100, 3000]
visual_in['z_scale'] = 'linear'
visual_in['y_label'] = 'h'
visual_in['y_unit'] = 'km'
visual_in['y_lim'] = [90, 350]
items[var_name] = {
    'name':     var_name,
    'fullname':     'electron temperature',
    'label':        r'$T_e$',
    'unit':         'K',
    'unit_label':   None,
    'group':        '',
    'value':        var_name,
    'error':        var_name + '_err',
    'dim':          2,
    'depends':      {0: depend_0, 1: depend_1},
    'timestamps':   timestamps,
    'positions':    coords,
    'visual_config':       visual_in
}

##################################################################################
var_name = 'T_i'
visual_in = dict(visual_config_2)
visual_in['z_label'] = 'Ion density'
visual_in['z_lim'] = [100, 2500]
visual_in['z_scale'] = 'linear'
visual_in['y_label'] = 'h'
visual_in['y_unit'] = 'km'
visual_in['y_lim'] = [90, 350]
items[var_name] = {
    'name':     var_name,
    'fullname':     'ion temperature',
    'label':        r'$T_i$',
    'unit':         'K',
    'unit_label':   None,
    'group':        '',
    'value':        var_name,
    'error':        var_name + '_err',
    'dim':          2,
    'depends':      {0: depend_0, 1: depend_1},
    'timestamps':   timestamps,
    'positions':    coords,
    'visual_config':       visual_in
}

##################################################################################
var_name = 'v_i_los'
visual_in = dict(visual_config_2)
visual_in['z_lim'] = [-200, 200]
visual_in['z_scale'] = 'linear'
visual_in['y_label'] = 'h'
visual_in['y_unit'] = 'km'
visual_in['y_lim'] = [90, 350]
items[var_name] = {
    'name':     var_name,
    'fullname':     'LOS ion velocity',
    'label':        r'$v_i$',
    'unit':         'm/s',
    'unit_label':   r'm$/$s',
    'group':        '',
    'value':        var_name,
    'error':        var_name + '_err',
    'dim':          2,
    'depends':      {0: depend_0, 1: depend_1},
    'timestamps':   timestamps,
    'positions':    coords,
    'visual_config':       visual_in
}

##################################################################################
var_name = 'az'
visual_in = dict(visual_config_1)
visual_in['z_label'] = 'az'
visual_in['y_label'] = ('group', )
visual_in['z_lim'] = None
visual_in['y_lim'] = [0, 360]
visual_in['plot_type'] = '1noE'
visual_in['plot_config'] = {
    'linestyle':        '',
    'linewidth':        1.5,
    'marker':           '.',
    'markersize':       3,
}
items[var_name] = {
    'name':     var_name,
    'fullname':     'Azimuth',
    'label':        r'az',
    'unit':         None,
    'unit_label':   None,
    'group':        'radar parameters',
    'value':        var_name,
    'error':        None,
    'dim':          1,
    'depends':      {0: depend_0},
    'timestamps':   timestamps,
    'positions':    coords,
    'visual_config':       visual_in
}

##################################################################################
var_name = 'el'
visual_in = dict(visual_config_1)
visual_in['z_label'] = 'el'
visual_in['z_lim'] = None
visual_in['plot_type'] = '1noE'
visual_in['plot_config'] = {
    'linestyle':        '',
    'linewidth':        1.5,
    'marker':           '.',
    'markersize':       3,
}

items[var_name] = {
    'name':     var_name,
    'fullname':     'Elevation',
    'label':        r'el',
    'unit':         None,
    'unit_label':   None,
    'group':        'radar parameters',
    'value':        var_name,
    'error':        None,
    'dim':          1,
    'depends':      {0: depend_0},
    'timestamps':   timestamps,
    'positions':    coords,
    'visual_config':       visual_in
}

