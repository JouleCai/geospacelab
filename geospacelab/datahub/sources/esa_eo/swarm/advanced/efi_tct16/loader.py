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
    'SC_GEO_R': 'Radius',
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
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        if self.product_version < '0301':
            raise NotImplementedError(f"Currently only support SWARM EFI TCT02 data products with version higher than '0301'.")
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['SC_GEO_r'] = self.variables['SC_GEO_R'] / 6371.2e3  # in Earth radius
        self.variables['SC_GEO_LON'] = self.variables['SC_GEO_LON'] % 360
        
        fb = self.variables['QUALITY_FLAG'].flatten()
        fb = (((fb[:,None] & (1 << np.arange(16)))) > 0).astype(int)
        self.variables['QUALITY_FLAG_BIN_AUX'] = fb
        self.variables['QUALITY_FLAG_BIN_IND'] = np.arange(17)[np.newaxis, :] -0.5
        
        fb = self.variables['CALIB_FLAG'].flatten()
        fb = (((fb[:,None] & (1 << np.arange(32)))) > 0).astype(int)
        self.variables['CALIB_FLAG_BIN_AUX'] = fb
        self.variables['CALIB_FLAG_BIN_IND'] = np.arange(33)[np.newaxis, :] -0.5
        
        
    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)