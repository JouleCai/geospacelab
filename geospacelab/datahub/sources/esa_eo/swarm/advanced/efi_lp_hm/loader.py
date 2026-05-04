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
    'SC_GEO_ALT': 'Height',
    'SC_GEO_R': 'Radius',
    'SC_SZA': 'SZA',
    'SC_SAz': 'SAz',
    'SC_GEO_ST': 'ST',
    'SC_QD_LAT': 'Diplat',
    'SC_QD_LON': 'Diplon',
    'SC_QD_MLT': 'MLT',
    'SC_AACGM_LAT': 'AACGMLat',
    'SC_AACGM_LON': 'AACGMLon',
    'n_p': 'n',
    'T_e': 'T_elec',
    'T_e_HG': 'Te_hgn',
    'T_e_LG': 'Te_lgn',
    'V_SC_HG': 'Vs_hgn',
    'V_SC_LG': 'Vs_lgn',
    'v_SC': 'U_SC',
    'FLAG': 'Flagbits'
}


class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['SC_GEO_r'] = self.variables['SC_GEO_ALT'] / 6371.2 + 1.0
        self.variables['SC_GEO_LON'] = self.variables['SC_GEO_LON'] % 360
        self.variables['SC_QD_LON'] = self.variables['SC_QD_LON'] % 360
        self.variables['SC_AACGM_LON'] = self.variables['SC_AACGM_LON'] % 360
        
        fb = self.variables['FLAG'].flatten()
        fb = (((fb[:,None] & (1 << np.arange(24)))) > 0).astype(int)
        self.variables['FLAG_BIN_AUX'] = fb
        self.variables['FLAG_BIN_IND'] = np.arange(25)[np.newaxis, :] -0.5
        
        
    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)