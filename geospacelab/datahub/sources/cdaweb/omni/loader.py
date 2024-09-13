# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import cdflib
import numpy as np
import datetime


class Loader:
    def __init__(self, file_path, file_type='cdf', load_data=True):
        self.file_path = file_path
        self.file_type = file_type
        self.variables = {}
        self.metadata = {}
        self.done = False
        if load_data:
            self.load()

    def load(self):
        if self.file_type == 'cdf' or self.file_type == 'hres-cdf':
            self.load_omni_cdf()

    def load_omni_cdf(self):
        f_cdf = cdflib.CDF(self.file_path)
        f_info = f_cdf.cdf_info()
        variables = {}
        self.metadata['var_attrs'] = {}
        for var_name, var_name_cdf in cdf_variable_name_dict.items():
            var = f_cdf.varget(var_name_cdf)
            var_attr = f_cdf.varattsget(var_name_cdf)
            fillval = var_attr['FILLVAL']
            var = np.where(var == fillval, np.nan, var)
            variables[var_name] = np.reshape(var, (var.size, 1))
            self.metadata['var_attrs'].update(var_name=f_cdf.varattsget(var_name_cdf))

        dts_str = cdflib.cdfepoch.encode(variables['EPOCH'].flatten())
        dts = np.empty_like(variables['EPOCH'], dtype=datetime.datetime)
        for ind, dt_str in enumerate(dts_str):
            dts[ind, 0] = datetime.datetime.strptime(dt_str + '000', '%Y-%m-%dT%H:%M:%S.%f')
        variables['DATETIME'] = dts
        variables['B_x_GSM'] = variables['B_x_GSE']
        variables['B_T_GSM'] = np.sqrt(variables['B_z_GSM']**2 + variables['B_y_GSM']**2)
        variables['B_TOTAL'] = np.sqrt(variables['B_z_GSM']**2 + variables['B_y_GSM']**2 + variables['B_x_GSM']**2)

        self.variables = variables
        self.metadata.update(f_cdf.globalattsget())
        self.done = True


cdf_variable_name_dict = {
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
    'B_z_GSE':      'BZ_GSE',
    'B_y_GSM':      'BY_GSM',
    'B_z_GSM':      'BZ_GSM',
    'v_sw':         'flow_speed',
    'v_x':          'Vx',
    'v_y':          'Vy',
    'v_z':          'Vz',
    'n_p':          'proton_density',
    'T':            'T',
    'p_dyn':        'Pressure',
    'E':            'E',
    'beta':         'Beta',
    'Ma_A':         'Mach_num',
    'Ma_MSP':       'Mgs_mach_num',
    'BSN_x':        'BSN_x',
    'BSN_y':        'BSN_y',
    'BSN_z':        'BSN_z',
    'AE':           'AE_INDEX',
    'AL':           'AL_INDEX',
    'AU':           'AU_INDEX',
    'SYM_H':        'SYM_H',
    'SYM_D':        'SYM_D',
    'ASY_D':        'ASY_D',
    'ASY_H':        'ASY_H',
}