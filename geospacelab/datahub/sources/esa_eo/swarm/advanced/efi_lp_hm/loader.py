# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

from geospacelab.datahub.sources.esa_eo.swarm.loader import LoaderModel

# define the default variable name dictionary
default_variable_name_dict = {
    'CDF_EPOCH': 'Timestamp',
    'SC_GEO_LAT': 'Latitude',
    'SC_GEO_LON': 'Longitude',
    'SC_GEO_ALT': 'Height',
    'SC_GEO_r': 'Radius',
    'SC_SZA': 'SZA',
    'SC_SAz': 'SAz',
    'SC_ST': 'ST',
    'SC_DIP_LAT': 'Diplat',
    'SC_DIP_LON': 'Diplon',
    'SC_QD_LAT': 'MLat',
    'SC_QD_MLT': 'MLT',
    'SC_AACGM_LAT': 'AACGMLat',
    'SC_AACGM_LON': 'AACGMLon',
    'n_e': 'n',
    'T_e_HGN': 'Te_hgn',
    'T_e_LGN': 'Te_lgn',
    'T_e': 'T_elec',
    'V_s_HGN': 'vs_hgn',
    'V_s_LGN': 'vs_lgn',
    # 'V_s': 'vs',
    'SC_U': 'U_SC',
    'FLAG': 'Flagbits'
}


class Loader(LoaderModel):
    """
    Load SWARM 2Hz or 16HZ TII data products. Currently support versions higher than "0301".

    The class is a hierarchy of :class:`SWARM data LoaderModel <geospacelab.datahub.sources.esa_eo.swarm.loader.LoaderModel>`

    """
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)
