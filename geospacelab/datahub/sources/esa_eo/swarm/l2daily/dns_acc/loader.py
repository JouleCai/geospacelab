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
    'SC_GEO_LAT': 'latitude',
    'SC_GEO_LON': 'longitude',
    'SC_GEO_ALT': 'altitude',
    'SC_GEO_LST': 'local_solar_time',
    'rho_n': 'density',
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
        self.variables['rho_n'][self.variables['rho_n']>1] = np.nan
        self.variables['SC_GEO_ALT'] = self.variables['SC_GEO_ALT'] * 1e-3
