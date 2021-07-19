
database = 'Madrigal'

timestamps = {
    'DATETIME': 'DATETIME',
}

coords = {
    'GEO': ['GEO_LAT', 'GEO_LON', 'GEO_ALT', 'GEO_ST'],
    'AACGM': ['AACGM_LAT', 'AACGM_LON', 'AACGM_R', 'AACGM_MLT'],
    'APEX': ['APEX_LAT', 'APEX_LON', 'APEX_ALT', 'APEX_MLT']
}

plot_config = {
    'linestyle':        '',
    'linewidth':        1.5,
    'marker':           '.',
    'markersize':       3,
}

visual_config_1 = {
    'plottype':     '1',
    'xdata':        ['DATETIME', 'AACGM_MLT'],
    'xdatares':     1,  # in seconds
    'xlabel':       ['UT', 'MLT'],
    'ydata':        tuple(['value']),
    'ylabel':       tuple(['label']),
    'yunit':        tuple(['unit']),
    'ylim':         [1e8, 1e13],
    'yscale':       'log',
    'plot_config':  dict(plot_config)
}

depend_0 = ['DATETIME', 'AACGM_MLT']
depend_1 = ['ALT']

##################################################################################
items = {}
##################################################################################
var_name = 'n_e'
visual_in = dict(visual_config_1)
visual_in['z_label'] = 'Electron density'
visual_in['z_lim'] = [1e9, 1e12]
items[var_name] = {
    'name':     var_name,
    'fullname':     'electron density',
    'label':        r'$n_e$',
    'unit':         'm-3',
    'unit_label':   r'm$^{-3}$',
    'group':        '',
    'value':        var_name,
    'error':        None,
    'dim':          2,
    'depends':      {0: depend_0, 1: depend_1},
    'timestamps':   timestamps,
    'positions':    coords,
    'visual':       visual_in
}

##################################################################################
var_name = 'T_e'
visual_in = dict(visual_config_1)
visual_in['z_label'] = 'Electron temperature'
visual_in['z_lim'] = [0, 3000]
items[var_name] = {
    'name':     var_name,
    'fullname':     'electron temperature',
    'label':        r'$T_e$',
    'unit':         'K',
    'unit_label':   None,
    'group':        '',
    'value':        var_name,
    'error':        None,
    'dim':          2,
    'depends':      {0: depend_0, 1: depend_1},
    'timestamps':   timestamps,
    'positions':    coords,
    'visual':       visual_in
}

##################################################################################
var_name = 'T_i'
visual_in = dict(visual_config_1)
visual_in['z_label'] = 'Ion density'
visual_in['z_lim'] = [0, 2500]
items[var_name] = {
    'name':     var_name,
    'fullname':     'ion temperature',
    'label':        r'$T_i$',
    'unit':         'K',
    'unit_label':   None,
    'group':        '',
    'value':        var_name,
    'error':        None,
    'dim':          2,
    'depends':      {0: depend_0, 1: depend_1},
    'timestamps':   timestamps,
    'positions':    coords,
    'visual':       visual_in
}

##################################################################################
var_name = 'v_i_los'
visual_in = dict(visual_config_1)
visual_in['z_label'] = ''
visual_in['z_lim'] = [-400, 400]
items[var_name] = {
    'name':     var_name,
    'fullname':     'electron density',
    'label':        r'$n_e$',
    'unit':         'm-3',
    'unit_label':   r'm$^{-3}$',
    'group':        '',
    'value':        var_name,
    'error':        None,
    'dim':          2,
    'depends':      {0: depend_0, 1: depend_1},
    'timestamps':   timestamps,
    'positions':    coords,
    'visual':       visual_in
}

##################################################################################
var_name = 'beam_az'
visual_in = dict(visual_config_1)
visual_in['z_label'] = ''
visual_in['z_lim'] = None
items[var_name] = {
    'name':     var_name,
    'fullname':     'Azimuth',
    'label':        r'az',
    'unit':         None,
    'unit_label':   None,
    'group':        '',
    'value':        var_name,
    'error':        None,
    'dim':          1,
    'depends':      {0: depend_0},
    'timestamps':   timestamps,
    'positions':    coords,
    'visual':       visual_in
}

##################################################################################
var_name = 'beam_el'
visual_in = dict(visual_config_1)
visual_in['z_label'] = ''
visual_in['z_lim'] = None
items[var_name] = {
    'name':     var_name,
    'fullname':     'Elevation',
    'label':        r'el',
    'unit':         None,
    'unit_label':   None,
    'group':        '',
    'value':        var_name,
    'error':        None,
    'dim':          1,
    'depends':      {0: depend_0},
    'timestamps':   timestamps,
    'positions':    coords,
    'visual':       visual_in
}

