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
    'n_e': 'Ne',
    'n_e_BKG': 'Background_Ne',
    'n_e_FRG': 'Foreground_Ne',
    'T_e': 'Te',
    'FLAG_PCP': 'PCP_flag',
    'GRAD_n_e_100km': 'Grad_Ne_at_100km',
    'GRAD_n_e_50km': 'Grad_Ne_at_50km',
    'GRAD_n_e_20km': 'Grad_Ne_at_20km',
    'GRAD_n_e_PCP_EDGE': 'Grad_Ne_at_PCP_edge',
    'ROD': 'ROD',
    'RODI_10s': 'RODI10s',
    'RODI_20s': 'RODI20s',
    'd_n_e_10s': 'delta_Ne10s',
    'd_n_e_20s': 'delta_Ne20s',
    'd_n_e_40s': 'delta_Ne40s',
    'num_GPS_SATs': 'Num_GPS_satellites',
    'VTEC_MEDIAN': 'mVTEC',
    'VTEC_STD': 'TEC_STD',
    'ROT_MEDIAN': 'mROT',
    'ROTI_10s_MEDIAN': 'mROTI10s',
    'ROTI_20s_MEDIAN': 'mROTI20s',
    'IPIR_INDEX': 'IPIR_index',
    'FLAG_IBI': 'IBI_flag',
    'Ionosphere_Region': 'Ionosphere_region_flag',
    'FLAG_n_e': 'Ne_quality_flag',
}

class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['SC_GEO_r'] = self.variables['SC_GEO_R'] / 6371.2e3
        self.variables['SC_GEO_LON'] = self.variables['SC_GEO_LON'] % 360
        

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)