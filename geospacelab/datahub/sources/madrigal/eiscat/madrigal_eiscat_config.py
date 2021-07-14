
database = 'Madrigal'

timestamps = {
    'DATETIME': 'DATETIME',
    'SECTIME': 'SECTIME'
}

coords = {
    'GEO': ['GEO_LAT', 'GEO_LON', 'GEO_ALT', 'GEO_ST'],
    'AACGM': ['AACGM_LAT', 'AACGM_LON', 'AACGM_R', 'AACGM_MLT']
}

plot_config = {
    'linestyle':        '',
    'linewidth':        1.5,
    'marker':           '.',
    'markersize':       3,
}

visual_config_1 = {
    'plottype':     '1',
    'xdata':        ['SC_DATETIME', 'SC_GEO_LAT', 'SC_GEO_LON', 'SC_AACGM_LAT', 'SC_AACGM_MLT'],
    'xdatares':     1,  # in seconds
    'xlabel':       ['UT', 'GLAT', 'GLON', 'MLAT', 'MLT'],
    'ydata':        tuple(['value']),
    'ylabel':       tuple(['label']),
    'yunit':        tuple(['unit']),
    'ylim':         [1e8, 1e13],
    'yscale':       'log',
    'plot_config':  dict(plot_config)
}

##################################################################################
items = {}
##################################################################################
var_name = 'n_e'
visual_in = dict(visual_config_1)
visual_in['zlabel'] = 'Electron'
visual_in['ylim'] = [1e4, 3e9]
visual_in['ylabel'] = tuple(['group'])
items[var_name] = {
    'paraname':     var_name,
    'fullname':     'integrated energy flux-electron',
    'label':        'electron flux',
    'unit':         r'$\#/$cm$^{2}/$s$/$ster',
    'group':        'number flux',
    'value':        var_name,
    'error':        None,
    'dim':          1,
    'timestamps':   timestamps,
    'positions':    coords,
    'visual':       visual_in
}
