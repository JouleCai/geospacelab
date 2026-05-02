# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import numpy as np

from geospacelab.datahub.sources.esa_eo.swarm.loader import LoaderModel

# define the default variable name dictionary
default_variable_name_dict = {
    'CDF_EPOCH': 'Timestamp',
    'Sync_Status': 'SyncStatus',
    'SC_GEO_LAT': 'Latitude',
    'SC_GEO_LON': 'Longitude',
    'SC_GEO_R': 'Radius',
    'v_SC_ITRF': 'U_orbit',
    'n_i': 'N_ion',
    'CALIB_n_i': 'dN_ion',
    'n_i_err': 'N_ion_error',
    'n_e': 'N_elec',
    'n_e_err': 'N_elec_error',
    'T_e': 'T_elec',
    'T_e_err': 'T_elec_error',
    'CALIB_T_e': 'dT_elec',
    'V_SC': 'Vs',
    'V_SC_err': 'Vs_error',
    'FLAG_LP': 'Flags_LP',
    'FLAG_n_i': 'Flags_N_ion',
    'FLAG_n_e': 'Flags_N_elec',
    'FLAG_T_e': 'Flags_T_elec',
    'FLAG_V_SC': 'Flags_Vs',
    'FLAG_BITS_1': 'Flagbits1',
    'FLAG_BITS_2': 'Flagbits2',
    'Gamma_1': 'Gamma1',
    'Gamma_2': 'Gamma2',    
}


class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['SC_GEO_r'] = self.variables['SC_GEO_R'] / 6371.2e3
        self.variables['SC_GEO_LON'] = self.variables['SC_GEO_LON'] % 360
        
        fb = self.variables['FLAG_BITS_1'].flatten()
        fb = (((fb[:,None] & (1 << np.arange(32)))) > 0).astype(int)
        self.variables['FLAG_1_BIN_AUX'] = fb
        self.variables['FLAG_1_BIN_IND'] = np.arange(32)[np.newaxis, :]
        
        fb = self.variables['FLAG_BITS_2'].flatten()
        fb = (((fb[:,None] & (1 << np.arange(17)))) > 0).astype(int)
        self.variables['FLAG_2_BIN_AUX'] = fb
        self.variables['FLAG_2_BIN_IND'] = np.arange(17)[np.newaxis, :]

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)