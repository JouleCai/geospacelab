

class Loader:
    def __init__(self, file_path, file_type='cdf', load_data=True):
        self.file_path = file_path
        self.file_type = file_type
        self.variables = {}
        self.metadata = {}
        if load_data:
            self.load()

    def load(self):
        pass

variable_name_dict = {
    'EPOCH':    'Epoch',
    'YEAR':     'YR',
    'DAY':      'Day',
    'HOUR':     'HR',
    'MIN':      'Minute',
    'SC_ID_IMF':    'IMF',
    'SC_ID_PLS':    'PLS',
    'IMF_PTS':   'IMF_PTS',
    'PLS_PTS':   'PLS_PTS',
    'PCT_INTERP': 'percent_interp',
    'Timeshift':      'Timeshift',
    'Timeshift_RMS':  'RMS_Timeshift',
    'B_x_GSE':      'BX_GSE',
    'B_y_GSE':      'BY_GSE',
    'B_z_GSE':      'B_z_GSE',
    'B_y_GSM':      'B_y_GSM',
    'B_z_GSM':      'B_z_GSM',
    'v_sw':         'flow_speed',
    'v_x':          'Vx',
    'v_y':          'Vy',
    'v_z':          'Vz',
    'n_p':          'proton_density',
    'T':            'T',
    'p':            'Pressure',
    'E':            'E',
    'beta':         'Beta',
    'Ma_A':         'Mach_num',
    'Ma_MSP':       'Mgs_mach_num',
    'BSN_x':        'BSN_x',
    'BSN_y':        'BSN_y',
    'BSN_z':        'BSN_z',
    
}

def load_cdaweb_cdf(filepath):
    vars = {}