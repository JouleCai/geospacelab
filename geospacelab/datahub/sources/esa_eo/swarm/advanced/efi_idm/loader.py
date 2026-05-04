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
    'SC_GEO_LAT': 'Latitude',
    'SC_GEO_LON': 'Longitude',
    'SC_GEO_R': 'Radius',
    'SC_GEO_ALT': 'Height',
    'SC_QD_LAT': 'QDLatitude',
    'SC_QD_MLT': 'MLT',
    'v_SC_NEC': 'V_sat_nec',
    'M_i_eff': 'M_i_eff',
    'M_i_eff_err': 'M_i_eff_err',
    'FLAG_M_i_eff': 'M_i_eff_Flags',
    'M_i_eff_model': 'M_i_eff_tbt_model',
    'v_i': 'V_i',
    'v_i_err': 'V_i_err',
    'FLAG_v_i': 'V_i_Flags',
    'v_i_raw': 'V_i_raw',
    'n_i': 'N_i',
    'n_i_err': 'N_i_err',
    'FLAG_n_i': 'N_i_Flags',
    'A_FP': 'A_fp',
    'R_LP': 'R_p',
    'T_e': 'T_e',
    'Phi_SC': 'Phi_sc',
}


class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        if self.product_version < '0301':
            raise NotImplementedError(f"Currently only support SWARM EFI TCT02 data products with version higher than '0301'.")
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['SC_GEO_r'] = self.variables['SC_GEO_R'] / 6371.2e3  # in Earth radius
        self.variables['SC_GEO_LON'] = self.variables['SC_GEO_LON'] % 360
        self.variables['SC_GEO_ALT'] = self.variables['SC_GEO_ALT'] / 1e3  # in km
        
        fb = self.variables['FLAG_M_i_eff'].flatten()
        fb = (((fb[:,None] & (1 << np.arange(18)))) > 0).astype(int)
        self.variables['FLAG_M_i_eff_BIN_AUX'] = fb
        self.variables['FLAG_M_i_eff_BIN_IND'] = np.arange(19)[np.newaxis, :] -0.5
        
        fb = self.variables['FLAG_v_i'].flatten()
        fb = (((fb[:,None] & (1 << np.arange(18)))) > 0).astype(int)
        self.variables['FLAG_v_i_BIN_AUX'] = fb
        self.variables['FLAG_v_i_BIN_IND'] = np.arange(19)[np.newaxis, :] -0.5
        
        fb = self.variables['FLAG_n_i'].flatten()
        fb = (((fb[:,None] & (1 << np.arange(18)))) > 0).astype(int)
        self.variables['FLAG_n_i_BIN_AUX'] = fb
        self.variables['FLAG_n_i_BIN_IND'] = np.arange(19)[np.newaxis, :] -0.5
        
        
    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)