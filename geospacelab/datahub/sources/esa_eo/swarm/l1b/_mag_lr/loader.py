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
    # 'SC_GEO_ALT': 'Height',
    'SC_GEO_r': 'Radius',
    # 'SC_SZA': 'SZA',
    # 'SC_SAz': 'SAz',
    # 'SC_ST': 'ST',
    # 'SC_DIP_LAT': 'Diplat',
    # 'SC_DIP_LON': 'Diplon',
    # 'SC_QD_LAT': 'MLat',
    # 'SC_QD_MLT': 'MLT',
    # 'SC_AACGM_LAT': 'AACGMLat',
    # 'SC_AACGM_LON': 'AACGMLon',
    'B_NEC': 'B_NEC',
    'B_VFM': 'B_VFM',
    'q_NEC_CRF': 'q_NEC_CRF',
    'FLAG_B': 'Flags_B',
    'FLAG_q': 'Flags_q',
    'FLAG_Platform': 'Flags_Platform',
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
        super(Loader, self).load_data(**kwargs)
        self.variables['B_N'] = self.variables['B_NEC'][:, 0][:, np.newaxis]
        self.variables['B_E'] = self.variables['B_NEC'][:, 1][:, np.newaxis]
        self.variables['B_C'] = self.variables['B_NEC'][:, 2][:, np.newaxis]
        self.variables['SC_GEO_r'] = self.variables['SC_GEO_r'] * 1e-3
