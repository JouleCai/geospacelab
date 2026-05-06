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
    'ORBIT_ID': 'ORB_C',
    'ORBIT_DIR': 'DIR_C',
    'EVENT_ID': 'ID_C',
    'CDF_EPOCH': 'Timestamp_C',
    'GEO_LAT': 'Latitude_C',
    'GEO_LON': 'Longitude_C',
    'GEO_R': 'Radius_C',
    'QD_LAT': 'Latitude_QD_C',
    'QD_LON': 'Longitude_QD_C',
    'QD_MLT': 'MLT_C',
    'SZA': 'SZA_C',
    'Frequency': 'Frequency_C',
    'Halfwidth': 'Halfwidth_C',
    'Power': 'Power_C',
    'Prominence': 'Prominence_C',
    'QUALITY_B': 'Quality_B_C',
    'QUALITY_p': 'Quality_p_C',
    'QUALITY_n': 'Quality_n_C',
    'ORBIT_ID_MEAN': 'ORB_m_C',
    'ORBIT_DIR_MEAN': 'DIR_m_C',
    'EVENT_ID_MEAN': 'ID_m_C',
    'CDF_EPOCH_MEAN': 'Timestamp_m_C',
    'GEO_LAT_MEAN': 'Latitude_m_C',
    'GEO_LON_MEAN': 'Longitude_m_C',
    'GEO_R_MEAN': 'Radius_m_C',
    'QD_LAT_MEAN': 'Latitude_QD_m_C',
    'QD_LON_MEAN': 'Longitude_QD_m_C',
    'QD_MLT_MEAN': 'MLT_m_C',
    'SZA_MEAN': 'SZA_m_C',
    'Duration_MEAN': 'Duration_m_C',
    'Frequency_MEAN': 'Frequency_m_C',
    'Freq_STD_MEAN': 'Freq_std_m_C',
    'Halfwidth_MEAN': 'Halfwidth_m_C',
    'Power_MEAN': 'Power_m_C',
    'Prominence_MEAN': 'Prominence_m_C',
    'ROFC_MEAN': 'ROFC_m_C',
    'QUALITY_MEAN': 'Quality_m_C',  
}


class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['DATETIME'] = self.variables['SC_DATETIME']
        self.variables['GEO_LON'] = self.variables['GEO_LON'] % 360
        self.variables['GEO_r'] = self.variables['GEO_R'] / 6371.2e3
        self.variables['QD_LON'] = self.variables['QD_LON'] % 360
        
        self.variables['DATETIME_MEAN'] = self.variables['SC_DATETIME_MEAN']
        self.variables['GEO_LON_MEAN'] = self.variables['GEO_LON_MEAN'] % 360
        self.variables['GEO_r_MEAN'] = self.variables['GEO_R_MEAN'] / 6371.2e3
        self.variables['QD_LON_MEAN'] = self.variables['QD_LON_MEAN'] % 360
        
        fb = self.variables['QUALITY_B'].flatten()
        fb = (((fb[:,None] & (1 << np.arange(10)))) > 0).astype(int)
        self.variables['QUALITY_B_BIN_AUX'] = fb
        self.variables['QUALITY_BIN_IND'] = np.arange(fb.shape[1])[np.newaxis, :] - 0.5
        

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)