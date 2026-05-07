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
    'j_r': 'IRC',
    'j_r_err': 'IRC_Error',
    'j_FA': 'FAC',
    'j_FA_err': 'FAC_Error',
    'FLAG': 'Flags',
    'FLAG_F': 'Flags_F',
    'FLAG_B': 'Flags_B',
    'FLAG_q': 'Flags_q',
}


class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['SC_GEO_r'] = self.variables['SC_GEO_R'] / 6371.2e3
        self.variables['SC_GEO_LON'] = self.variables['SC_GEO_LON'] % 360
        
        fb = np.int32(np.floor(self.variables['FLAG'].flatten()))
        max_num_digits = 10 # len(str(np.max(fb)))
        fb_arr = [d for fbn in fb for d in str(fbn).zfill(max_num_digits)]
        fb_arr = np.array(fb_arr, dtype=int).reshape(fb.shape[0], max_num_digits)
        fb_arr = np.fliplr(fb_arr)
        self.variables['FLAG_DIGIT_AUX'] = fb_arr
        self.variables['FLAG_DIGIT_IND'] = np.arange(fb_arr.shape[1]+1)[np.newaxis,:] - 0.5
        
        fb = np.int64(np.floor(self.variables['FLAG_F'].flatten()))
        max_num_digits = 10 # len(str(np.max(fb)))
        fb_arr = [d for fbn in fb for d in str(fbn).zfill(max_num_digits)]
        fb_arr = np.array(fb_arr, dtype=int).reshape(fb.shape[0], max_num_digits)
        fb_arr = np.fliplr(fb_arr)
        self.variables['FLAG_F_DIGIT_AUX'] = fb_arr
        self.variables['FLAG_F_DIGIT_IND'] = np.arange(fb_arr.shape[1]+1)[np.newaxis,:] - 0.5
        
        fb = np.int64(np.floor(self.variables['FLAG_B'].flatten()))
        max_num_digits = 10 # len(str(np.max(fb)))
        fb_arr = [d for fbn in fb for d in str(fbn).zfill(max_num_digits)]
        fb_arr = np.array(fb_arr, dtype=int).reshape(fb.shape[0], max_num_digits)
        fb_arr = np.fliplr(fb_arr)
        self.variables['FLAG_B_DIGIT_AUX'] = fb_arr
        self.variables['FLAG_B_DIGIT_IND'] = np.arange(fb_arr.shape[1]+1)[np.newaxis,:] - 0.5
        
        fb = np.int64(np.floor(self.variables['FLAG_q'].flatten()))
        max_num_digits = 10 # len(str(np.max(fb)))
        fb_arr = [d for fbn in fb for d in str(fbn).zfill(max_num_digits)]
        fb_arr = np.array(fb_arr, dtype=int).reshape(fb.shape[0], max_num_digits)
        fb_arr = np.fliplr(fb_arr)
        self.variables['FLAG_q_DIGIT_AUX'] = fb_arr
        self.variables['FLAG_q_DIGIT_IND'] = np.arange(fb_arr.shape[1]+1)[np.newaxis,:] - 0.5

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)