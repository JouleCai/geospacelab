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
    'GEO_R': 'Radius',
    'POS_GPS': 'GPS_Position',
    'POS_LEO': 'LEO_Position',
    'PRN': 'PRN',
    'L1': 'L1',
    'L2': 'L2',
    'P1': 'P1',
    'P2': 'P2',
    'S1': 'S1',
    'S2': 'S2',
    'STEC_ABS': 'Absolute_STEC',
    'STEC_REL': 'Relative_STEC',
    'STEC_REL_err': 'Relative_STEC_RMS',
    'VTEC_ABS': 'Absolute_VTEC',
    'EL': 'Elevation_Angle',
    'DCB': 'DCB',
    'DCB_err': 'DCB_Error',
}


class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['DATETIME'] = self.variables['SC_DATETIME']
        
        self.variables['GEO_r'] = self.variables['GEO_R'] / 6371.2
        self.variables['GEO_LON'] = self.variables['GEO_LON'] % 360
        
        self.variables['DCB'] = np.full_like(self.variables['DATETIME'], self.variables['DCB'])
        self.variables['DCB_err'] = np.full_like(self.variables['DATETIME'], self.variables['DCB_err'])
        

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)