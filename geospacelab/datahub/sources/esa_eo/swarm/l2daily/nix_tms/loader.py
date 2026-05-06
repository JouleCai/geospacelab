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
    'GEO_LAT': 'Latitude',
    'GEO_LON': 'Longitude',
    'Distance': 'Distance',
    'Azimuth': 'Azimuth',
    'Negix_X': 'Negix_X',
    'Negix_X_Sigma': 'Negix_X_Sigma',
    'Negix_X_P95': 'Negix_X_P95',
    'Negix_Y': 'Negix_Y',
    'Negix_Y_Sigma': 'Negix_Y_Sigma',
    'Negix_Y_P95': 'Negix_Y_P95',
    'Negix_Total': 'Negix_Total',
    'Negix_Sigma': 'Negix_Sigma',
    'Negix_P95': 'Negix_P95',
    'N_Measurements': 'N_Measurements',
    'Flag_Negix': 'Flag_Negix',
    'Orbit_Label': 'Orbit_Label',
}


class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['DATETIME'] = self.variables['SC_DATETIME']
        
        self.variables['GEO_LON'] = self.variables['GEO_LON'] % 360

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)