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
default_variable_name_dict_0502 = {
    'CDF_EPOCH': 'Timestamp',
    'GEO_LAT': 'Latitude',
    'GEO_LON': 'Longitude',
    'EF_EQ': 'EEF',
    'EEJ_E': 'EEJ_meast',
    'EEJ_N': 'EEJ_mnorth',
    'Relative_Error': 'RelErr',
    'FLAG': 'Flags',
    'CDF_EPOCH_Track': 'Timestamp_Track',
    'GEO_LAT_Track': 'Latitude_Track',
    'GEO_LON_Track': 'Longitude_Track',
    'GEO_R_Track': 'Radius_Track',
    'K_SQ_Track': 'Ksq_Track',
    'K_EEJ_Track': 'Keej_Track',
    'Length_Track': 'Length_Track',
}

default_variable_name_dict_old = {
    'CDF_EPOCH': 'Timestamp',
    'GEO_LAT': 'Latitude',
    'GEO_LON': 'Longitude',
    'EF_EQ': 'EEF',
    'EEJ_E': 'EEJ',
    'Relative_Error': 'RelErr',
    'FLAG': 'Flags',
}


class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        version = kwargs.get('product_version', '0502')
        if version >= '0502':
            variable_name_dict = default_variable_name_dict_0502
        else:
            variable_name_dict = default_variable_name_dict_old
        
        kwargs.setdefault('variable_name_dict', variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        self.variables['QD_LAT'] = np.tile(np.arange(-20, 20.5, 0.5), (self.variables['SC_DATETIME'].shape[0], 1))
        if self.product_version >= '0502':
            self._process_0502(self.variables)
        else:
            self._process_0501(self.variables)
    
    def _process_0502(self, variables):
        self.variables['DATETIME'] = self.variables['SC_DATETIME']
        self.variables['DATETIME_Track'] = self.variables['SC_DATETIME_Track']
        self.variables['GEO_r_Track'] = self.variables['GEO_R_Track'] / 6371.2
        self.variables['GEO_LON'] = self.variables['GEO_LON'] % 360
        self.variables['GEO_LON_Track'] = self.variables['GEO_LON_Track'] % 360
    
    def _process_0501(self, variables):
        self.variables['DATETIME'] = self.variables['SC_DATETIME']
        self.variables['GEO_LON'] = self.variables['GEO_LON'] % 360

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)