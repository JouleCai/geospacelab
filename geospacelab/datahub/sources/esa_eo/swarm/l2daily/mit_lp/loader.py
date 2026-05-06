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
    'Counter': 'Counter',
    'GEO_LAT': 'Latitude',
    'GEO_LON': 'Longitude',
    'GEO_R': 'Radius',
    'QD_LAT': 'Latitude_QD',
    'QD_LON': 'Longitude_QD',
    'QD_MLT': 'MLT',
    'L': 'L_value',
    'SZA': 'SZA',
    'n_e': 'Ne',
    'T_e': 'Te',
    'Depth': 'Depth',
    'DR': 'DR',
    'Width': 'Width',
    'dL': 'dL',
    'PW_Gradient': 'PW_Gradient',
    'EW_Gradient': 'EW_Gradient',
    'QUALITY': 'Quality',
    'CDF_EPOCH_ID': 'Timestamp_ID',
    'GEO_LAT_ID': 'Latitude_ID',
    'GEO_LON_ID': 'Longitude_ID',
    'GEO_R_ID': 'Radius_ID',
    'QD_LAT_ID': 'Latitude_QD_ID',
    'QD_LON_ID': 'Longitude_QD_ID',
    'QD_MLT_ID': 'MLT_ID',
    'L_ID': 'L_value_ID',
    'SZA_ID': 'SZA_ID',
    'n_e_ID': 'Ne_ID',
    'T_e_ID': 'Te_ID',
    'Position_Quality_ID': 'Position_Quality_ID',
}

class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['DATETIME'] = self.variables['SC_DATETIME']
        self.variables['DATETIME_ID'] = self.variables['SC_DATETIME_ID']
        
        self.variables['GEO_r'] = self.variables['GEO_R'] / 6371.2e3
        self.variables['GEO_LON'] = self.variables['GEO_LON'] % 360
        self.variables['QD_LON'] = self.variables['QD_LON'] % 360
       
        self.variables['GEO_r_ID'] = self.variables['GEO_R_ID'] / 6371.2e3   
        self.variables['GEO_LON_ID'] = self.variables['GEO_LON_ID'] % 360
        self.variables['QD_LON_ID'] = self.variables['QD_LON_ID'] % 360
        
        self.variables['QUALITY_IND'] = np.arange(9)[np.newaxis, :] - 0.5
        self.variables['ID_IND'] = np.arange(8)[np.newaxis, :] - 0.5
        

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)