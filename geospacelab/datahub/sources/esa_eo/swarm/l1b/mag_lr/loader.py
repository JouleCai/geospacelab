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
    'SYNC_STATUS': 'SyncStatus',
    'SC_GEO_LAT': 'Latitude',
    'SC_GEO_LON': 'Longitude',
    'SC_GEO_R': 'Radius',
    'B_VFM': 'B_VFM',
    'B_NEC': 'B_NEC',
    'dB_Sun_VFM': 'dB_Sun',
    'dB_AOCS_VFM': 'dB_AOCS',
    'dB_other_VFM': 'dB_other',
    'B_VFM_err': 'B_error',
    'q_NEC_CRF': 'q_NEC_CRF',
    'Att_error': 'Att_error',
    'FLAG_B': 'Flags_B',
    'FLAG_q': 'Flags_q',
    'FLAG_Platform': 'Flags_Platform',
    'FLAG_F': 'Flags_F',
    'ASM_Freq_Dev': 'ASM_Freq_Dev',
    'F': 'F',
    'F_err': 'F_error',
}


class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['SC_GEO_r'] = self.variables['SC_GEO_R'] / 6371.2e3
        self.variables['SC_GEO_LON'] = self.variables['SC_GEO_LON'] % 360
        
        self.variables['B_N'] = self.variables['B_NEC'][:, 0][:, np.newaxis]
        self.variables['B_E'] = self.variables['B_NEC'][:, 1][:, np.newaxis]
        self.variables['B_C'] = self.variables['B_NEC'][:, 2][:, np.newaxis]
        
        self.variables['B_VFM_x'] = self.variables['B_VFM'][:, 0][:, np.newaxis]
        self.variables['B_VFM_y'] = self.variables['B_VFM'][:, 1][:, np.newaxis]
        self.variables['B_VFM_z'] = self.variables['B_VFM'][:, 2][:, np.newaxis]
        
        self.variables['B_VFM_x_err'] = self.variables['B_VFM_err'][:, 0][:, np.newaxis]
        self.variables['B_VFM_y_err'] = self.variables['B_VFM_err'][:, 1][:, np.newaxis]
        self.variables['B_VFM_z_err'] = self.variables['B_VFM_err'][:, 2][:, np.newaxis]
        
        self.variables['dB_Sun_VFM_x'] = self.variables['dB_Sun_VFM'][:, 0][:, np.newaxis]
        self.variables['dB_Sun_VFM_y'] = self.variables['dB_Sun_VFM'][:, 1][:, np.newaxis]
        self.variables['dB_Sun_VFM_z'] = self.variables['dB_Sun_VFM'][:, 2][:, np.newaxis]
        
        self.variables['dB_AOCS_VFM_x'] = self.variables['dB_AOCS_VFM'][:, 0][:, np.newaxis]
        self.variables['dB_AOCS_VFM_y'] = self.variables['dB_AOCS_VFM'][:, 1][:, np.newaxis]
        self.variables['dB_AOCS_VFM_z'] = self.variables['dB_AOCS_VFM'][:, 2][:, np.newaxis]
        
        self.variables['dB_other_VFM_x'] = self.variables['dB_other_VFM'][:, 0][:, np.newaxis]
        self.variables['dB_other_VFM_y'] = self.variables['dB_other_VFM'][:, 1][:, np.newaxis]
        self.variables['dB_other_VFM_z'] = self.variables['dB_other_VFM'][:, 2][:, np.newaxis]
        
        fb = self.variables['FLAG_B'].flatten()
        fb = (((fb[:,None] & (1 << np.arange(9)))) > 0).astype(int)
        self.variables['FLAG_B_BIN_AUX'] = fb
        self.variables['FLAG_B_BIN_IND'] = np.arange(fb.shape[1])[np.newaxis, :] - 0.5
        
        fb = self.variables['FLAG_q'].flatten()
        fb = (((fb[:,None] & (1 << np.arange(8)))) > 0).astype(int)
        self.variables['FLAG_q_BIN_AUX'] = fb
        self.variables['FLAG_q_BIN_IND'] = np.arange(fb.shape[1])[np.newaxis, :] - 0.5
        
        fb = self.variables['FLAG_Platform'].flatten()
        fb = (((fb[:,None] & (1 << np.arange(9)))) > 0).astype(int)
        self.variables['FLAG_Platform_BIN_AUX'] = fb
        self.variables['FLAG_Platform_BIN_IND'] = np.arange(fb.shape[1])[np.newaxis, :] - 0.5
        
        fb = self.variables['FLAG_F'].flatten()
        fb = (((fb[:,None] & (1 << np.arange(9)))) > 0).astype(int)
        self.variables['FLAG_F_BIN_AUX'] = fb
        self.variables['FLAG_F_BIN_IND'] = np.arange(fb.shape[1])[np.newaxis, :] - 0.5

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)
   
    
    def load_from_VirES(self, collection=None, kwargs_products=None):
        if self.from_FAST:
            collection 
            
        data = super().load_from_VirES(collection=collection, kwargs_products=kwargs_products)
        df = data.as_dataframe()