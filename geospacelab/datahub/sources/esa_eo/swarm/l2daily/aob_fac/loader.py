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
    'CDF_EPOCH': 't',
    'SC_GEO_LAT': 'Latitude',
    'SC_GEO_LON': 'Longitude',
    'SC_GEO_Radius': 'Radius',
    'SC_QD_LAT': 'Latitude_QD',
    'SC_QD_LON': 'Longitude_QD',
    'SC_QD_MLT': 'MLT',
    'BOUNDARY_FLAG': 'Boundary_Flag',
    'QUALITY': 'Quality',
    'PAIR_INDICATOR': 'Pair_Indicator',
}


class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['SC_GEO_r'] = self.variables['SC_GEO_Radius'] / 6371.2e3
        self.variables['QUALITY_Pa'] = self.variables['QUALITY'][:, 0][:, np.newaxis]
        self.variables['QUALITY_Sigma'] = self.variables['QUALITY'][:, 1][:, np.newaxis]

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)