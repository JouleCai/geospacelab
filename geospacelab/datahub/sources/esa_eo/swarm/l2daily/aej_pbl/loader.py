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
    "CDF_EPOCH_PEAK": "t_Peak",
    'GEO_LAT_PEAK': 'Latitude_Peak',
    'GEO_LON_PEAK': 'Longitude_Peak',
    'QD_LAT_PEAK': 'Latitude_Peak_QD',
    'QD_LON_PEAK': 'Longitude_Peak_QD',
    'QD_MLT_PEAK': 'MLT_Peak',
    'J_PEAK': 'J',
    'CDF_EPOCH_EB': 't_EB',
    'GEO_LAT_EB': 'Latitude_EB',
    'GEO_LON_EB': 'Longitude_EB',
    'QD_LAT_EB': 'Latitude_EB_QD',
    'QD_LON_EB': 'Longitude_EB_QD',
    'QD_MLT_EB': 'MLT_EB',
    'CDF_EPOCH_PB': 't_PB',
    'GEO_LAT_PB': 'Latitude_PB',
    'GEO_LON_PB': 'Longitude_PB',
    'QD_LAT_PB': 'Latitude_PB_QD',
    'QD_LON_PB': 'Longitude_PB_QD',
    'QD_MLT_PB': 'MLT_PB',
    'QUALITY_FLAG': 'Flags',
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
        self.variables['DATETIME_WEJ_PEAK'] = self.variables['SC_DATETIME_PEAK'][:, 0][:, np.newaxis]
        self.variables['GEO_LAT_WEJ_PEAK'] = self.variables['GEO_LAT_PEAK'][:, 0][:, np.newaxis]
        self.variables['GEO_LON_WEJ_PEAK'] = self.variables['GEO_LON_PEAK'][:, 0][:, np.newaxis]
        self.variables['QD_LAT_WEJ_PEAK'] = self.variables['QD_LAT_PEAK'][:, 0][:, np.newaxis]
        self.variables['QD_LON_WEJ_PEAK'] = self.variables['QD_LON_PEAK'][:, 0][:, np.newaxis]
        self.variables['QD_MLT_WEJ_PEAK'] = self.variables['QD_MLT_PEAK'][:, 0][:, np.newaxis]
        self.variables['GEO_ALT_WEJ_PEAK'] = np.full_like(self.variables['GEO_LAT_WEJ_PEAK'], 110.)
        self.variables['GEO_r_WEJ_PEAK'] = np.full_like(self.variables['GEO_LAT_WEJ_PEAK'], (110. + 6371.2)/6371.2)
        self.variables['WEJ_PEAK'] = self.variables['J_PEAK'][:, 0][:, np.newaxis]
        
        self.variables['DATETIME_EEJ_PEAK'] = self.variables['SC_DATETIME_PEAK'][:, 1][:, np.newaxis]
        self.variables['GEO_LAT_EEJ_PEAK'] = self.variables['GEO_LAT_PEAK'][:, 1][:, np.newaxis]
        self.variables['GEO_LON_EEJ_PEAK'] = self.variables['GEO_LON_PEAK'][:, 1][:, np.newaxis]
        self.variables['QD_LAT_EEJ_PEAK'] = self.variables['QD_LAT_PEAK'][:, 1][:, np.newaxis]
        self.variables['QD_LON_EEJ_PEAK'] = self.variables['QD_LON_PEAK'][:, 1][:, np.newaxis]
        self.variables['QD_MLT_EEJ_PEAK'] = self.variables['QD_MLT_PEAK'][:, 1][:, np.newaxis]
        self.variables['GEO_ALT_EEJ_PEAK'] = np.full_like(self.variables['GEO_LAT_EEJ_PEAK'], 110.)
        self.variables['GEO_r_EEJ_PEAK'] = np.full_like(self.variables['GEO_LAT_EEJ_PEAK'], (110. + 6371.2)/6371.2)
        self.variables['EEJ_PEAK'] = self.variables['J_PEAK'][:, 1][:, np.newaxis]
        
        self.variables['DATETIME_WEJ_EB'] = self.variables['SC_DATETIME_EB'][:, 0][:, np.newaxis]
        self.variables['GEO_LAT_WEJ_EB'] = self.variables['GEO_LAT_EB'][:, 0][:, np.newaxis]
        self.variables['GEO_LON_WEJ_EB'] = self.variables['GEO_LON_EB'][:, 0][:, np.newaxis]
        self.variables['QD_LAT_WEJ_EB'] = self.variables['QD_LAT_EB'][:, 0][:, np.newaxis]
        self.variables['QD_LON_WEJ_EB'] = self.variables['QD_LON_EB'][:, 0][:, np.newaxis]
        self.variables['QD_MLT_WEJ_EB'] = self.variables['QD_MLT_EB'][:, 0][:, np.newaxis]
        self.variables['GEO_ALT_WEJ_EB'] = np.full_like(self.variables['GEO_LAT_WEJ_EB'], 110.)
        self.variables['GEO_r_WEJ_EB'] = np.full_like(self.variables['GEO_LAT_WEJ_EB'], (110. + 6371.2)/6371.2)
        
        self.variables['DATETIME_EEJ_EB'] = self.variables['SC_DATETIME_EB'][:, 1][:, np.newaxis]
        self.variables['GEO_LAT_EEJ_EB'] = self.variables['GEO_LAT_EB'][:, 1][:, np.newaxis]
        self.variables['GEO_LON_EEJ_EB'] = self.variables['GEO_LON_EB'][:, 1][:, np.newaxis]
        self.variables['QD_LAT_EEJ_EB'] = self.variables['QD_LAT_EB'][:, 1][:, np.newaxis]
        self.variables['QD_LON_EEJ_EB'] = self.variables['QD_LON_EB'][:, 1][:, np.newaxis]
        self.variables['QD_MLT_EEJ_EB'] = self.variables['QD_MLT_EB'][:, 1][:, np.newaxis]
        self.variables['GEO_ALT_EEJ_EB'] = np.full_like(self.variables['GEO_LAT_EEJ_EB'], 110.)
        self.variables['GEO_r_EEJ_EB'] = np.full_like(self.variables['GEO_LAT_EEJ_EB'], (110. + 6371.2)/6371.2)
        
        self.variables['DATETIME_WEJ_PB'] = self.variables['SC_DATETIME_PB'][:, 0][:, np.newaxis]
        self.variables['GEO_LAT_WEJ_PB'] = self.variables['GEO_LAT_PB'][:, 0][:, np.newaxis]
        self.variables['GEO_LON_WEJ_PB'] = self.variables['GEO_LON_PB'][:, 0][:, np.newaxis]
        self.variables['QD_LAT_WEJ_PB'] = self.variables['QD_LAT_PB'][:, 0][:, np.newaxis]
        self.variables['QD_LON_WEJ_PB'] = self.variables['QD_LON_PB'][:, 0][:, np.newaxis]
        self.variables['QD_MLT_WEJ_PB'] = self.variables['QD_MLT_PB'][:, 0][:, np.newaxis]
        self.variables['GEO_ALT_WEJ_PB'] = np.full_like(self.variables['GEO_LAT_WEJ_PB'], 110.)
        self.variables['GEO_r_WEJ_PB'] = np.full_like(self.variables['GEO_LAT_WEJ_PB'], (110. + 6371.2)/6371.2)
        
        self.variables['DATETIME_EEJ_PB'] = self.variables['SC_DATETIME_PB'][:, 1][:, np.newaxis]
        self.variables['GEO_LAT_EEJ_PB'] = self.variables['GEO_LAT_PB'][:, 1][:, np.newaxis]
        self.variables['GEO_LON_EEJ_PB'] = self.variables['GEO_LON_PB'][:, 1][:, np.newaxis]
        self.variables['QD_LAT_EEJ_PB'] = self.variables['QD_LAT_PB'][:, 1][:, np.newaxis]
        self.variables['QD_LON_EEJ_PB'] = self.variables['QD_LON_PB'][:, 1][:, np.newaxis]
        self.variables['QD_MLT_EEJ_PB'] = self.variables['QD_MLT_PB'][:, 1][:, np.newaxis]
        self.variables['GEO_ALT_EEJ_PB'] = np.full_like(self.variables['GEO_LAT_EEJ_PB'], 110.)
        self.variables['GEO_r_EEJ_PB'] = np.full_like(self.variables['GEO_LAT_EEJ_PB'], (110. + 6371.2)/6371.2)
        
    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
        return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)