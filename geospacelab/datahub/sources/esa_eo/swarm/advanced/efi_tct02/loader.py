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
    'SC_GEO_r': 'Radius',
    'SC_QD_LAT': 'QDLatitude',
    'SC_QD_MLT': 'MLT',
    'v_i_H_x': 'Vixh',
    'v_i_H_x_err': 'Vixh_error',
    'v_i_V_x': 'Vixv',
    'v_i_V_x_err': 'Vixv_error',
    'v_i_H_y': 'Viy',
    'v_i_H_y_err': 'Viy_error',
    'v_i_V_z': 'Viz',
    'v_i_V_z_err': 'Viz_error',
    'v_SC_N': 'VsatN',
    'v_SC_E': 'VsatE',
    'v_SC_C': 'VsatC',
    'E_H_x': 'Ehx',
    'E_H_y': 'Ehy',
    'E_H_z': 'Ehz',
    'E_V_x': 'Evx',
    'E_V_y': 'Evy',
    'E_V_z': 'Evz',
    'B_x': 'Bx',
    'B_y': 'By',
    'B_z': 'Bz',
    'v_i_CR_x': 'Vicrx',
    'v_i_CR_y': 'Vicry',
    'v_i_CR_z': 'Vicrz',
    'QUALITY_FLAG': 'Quality_flags',
    'CALIB_FLAG': 'Calibration_flags',
}


class Loader(LoaderModel):
    """
    Load SWARM 2Hz or 16HZ TII data products. Currently support versions higher than "0301".

    The class is a hierarchy of :class:`SWARM data LoaderModel <geospacelab.datahub.sources.esa_eo.swarm.loader.LoaderModel>`

    """
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)
        self.variables['SC_GEO_r'] = self.variables['SC_GEO_r'] * 1e-3  # in km
        self.variables['SC_GEO_ALT'] = self.variables['SC_GEO_r'] - 6371.2

