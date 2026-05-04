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
    'SC_GEO_ALT': 'Height',
    'SC_GEO_R': 'Radius',
    'SC_QD_LAT': 'QDLatitude',
    'SC_QD_MLT': 'MLT',
    'T_i_DRIFT': 'Ti_meas_drift',
    'T_i_DRIFT_MODEL': 'Ti_model_drift',
    'T_e_LP': 'Te_adj_LP',
    'T_n_MSIS': 'Tn_msis',
    'FLAG_T_i_DRIFT': 'Flag_ti_meas',
    'FLAG_T_i_DRIFT_MODEL': 'Flag_ti_model',
}


class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['SC_GEO_r'] = self.variables['SC_GEO_R'] / 6371.2e3
        self.variables['SC_GEO_ALT'] = self.variables['SC_GEO_ALT'] / 1e3
        self.variables['SC_GEO_LON'] = self.variables['SC_GEO_LON'] % 360
        
        fb = self.variables['FLAG_T_i_DRIFT'].flatten()
        fb = (((fb[:,None] & (1 << np.arange(4)))) > 0).astype(int)
        self.variables['FLAG_T_i_DRIFT_BIN_AUX'] = fb
        self.variables['FLAG_T_i_DRIFT_BIN_IND'] = np.arange(fb.shape[1]+1)[np.newaxis,:] - 0.5
        
        fb = self.variables['FLAG_T_i_DRIFT_MODEL'].flatten()
        fb = (((fb[:,None] & (1 << np.arange(4)))) > 0).astype(int)
        self.variables['FLAG_T_i_DRIFT_MODEL_BIN_AUX'] = fb
        self.variables['FLAG_T_i_DRIFT_MODEL_BIN_IND'] = np.arange(fb.shape[1]+1)[np.newaxis,:] - 0.5

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)