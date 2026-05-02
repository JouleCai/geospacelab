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
    'CDF_EPOCH': 'time',
    'rho_n': 'density',
    'rho_n_ORBITMEAN': 'density_orbitmean',
    'SC_GEO_LAT': 'latitude',
    'SC_GEO_LON': 'longitude',
    'SC_GEO_ALT': 'altitude',
    'SC_GEO_LST': 'local_solar_time',
    'FLAG': 'validity_flag',
}


class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['SC_GEO_r'] = self.variables['SC_GEO_ALT'] / 6371.2e3 + 1.0
        self.variables['rho_n'][self.variables['rho_n']>1] = np.nan
        self.variables['rho_n_ORBITMEAN'][self.variables['rho_n_ORBITMEAN']>1] = np.nan
        self.variables['SC_GEO_ALT'] = self.variables['SC_GEO_ALT'] * 1e-3
        self.variables['SC_GEO_LON'] = self.variables['SC_GEO_LON'] % 360

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)