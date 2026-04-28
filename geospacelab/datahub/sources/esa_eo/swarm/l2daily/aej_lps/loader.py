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
    'GEO_LAT': 'Latitude',
    'GEO_LON': 'Longitude',
    'QD_LAT': 'Latitude_QD',
    'QD_LON': 'Longitude_QD',
    'QD_MLT': 'MLT',
    'J_CF': 'J_CF',
    'J_DF': 'J_DF',
    'J_CF_SemiQD': 'J_CF_SemiQD',
    'J_DF_SemiQD': 'J_DF_SemiQD',
    'J_r': 'J_r',
    'CDF_EPOCH_QUAL': 't_qual',
    'RMS_MISFIT': 'RMS_misfit',
    'CONFIDENCE': 'Confidence',
}


class Loader(LoaderModel):
    """
    Load SWARM 2Hz or 16HZ TII data products. Currently support versions higher than "0301".

    The class is a hierarchy of :class:`SWARM data LoaderModel <geospacelab.datahub.sources.esa_eo.swarm.loader.LoaderModel>`

    """
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        self.variables['DATETIME'] = self.variables['SC_DATETIME']
        self.variables['DATETIME_QUAL'] = self.variables['SC_DATETIME_QUAL']
        self.variables['GEO_ALT'] = np.ones_like(self.variables['GEO_LAT']) * 110. # the altitude of the AEJ_LPS product is fixed to 110 km
        self.variables['GEO_r'] = self.variables['GEO_ALT'] / 6371.2 + 1 # the radius of the Earth is assumed to be 6371.2 km
        self.variables['GEO_LON'] = self.variables['GEO_LON'] % 360. # convert the longitude to [0, 360]
        self.variables['QD_LON'] = self.variables['QD_LON'] % 360. # convert the longitude to [0, 360]
        
        self.variables['J_CF_N'] = self.variables['J_CF'][:, 0][:, np.newaxis]
        self.variables['J_CF_E'] = self.variables['J_CF'][:, 1][:, np.newaxis]
        self.variables['J_DF_N'] = self.variables['J_DF'][:, 0][:, np.newaxis]
        self.variables['J_DF_E'] = self.variables['J_DF'][:, 1][:, np.newaxis]

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
        var_names_cdf_epoch = ['t', 't_qual']  # the variable names of the variables that are epoch in the cdf files
        return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)