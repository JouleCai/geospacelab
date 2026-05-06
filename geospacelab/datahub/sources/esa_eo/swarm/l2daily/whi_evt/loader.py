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
    'CDF_EPOCH_0': 'Timestamp_Whistler',
    'GEO_LAT': 'Latitude',
    'GEO_LON': 'Longitude',
    'GEO_R': 'Radius',
    'LT': 'LT',
    'Whistler_Dispersion': 'Whistler_Dispersion',
    'Whistler_t0': 'Whistler_t0',
    'Whistler_t0_uncertainty': 'Whistler_t0_uncertainty',
    'Intensity': 'Intensity',
    'CDF_EPOCH': 'Timestamp',
    'TimeFrac': 'TimeFrac',
    'F_analysed': 'F_analysed',
    'FLAG': 'Flags',
    'CDF_EPOCH_PSD': 'Timestamp_PSD',
    'Frequencies_PSD': 'Frequencies_PSD',
    'PSD': 'PSD',
}



class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['DATETIME_0'] = self.variables['SC_DATETIME_0']
        
        self.variables['GEO_r'] = self.variables['GEO_R'] / 6371.2e3
        self.variables['GEO_LON'] = self.variables['GEO_LON'] % 360
        

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)