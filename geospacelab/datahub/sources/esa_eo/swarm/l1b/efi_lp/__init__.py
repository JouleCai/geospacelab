# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import numpy as np
import datetime
import copy

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel, InstrumentModel, ProductModel
from geospacelab.datahub.sources.esa_eo import esaeo_database
from geospacelab.datahub.sources.esa_eo.swarm import swarm_facility
from geospacelab.config import prf
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool
from geospacelab.datahub.sources.esa_eo.swarm.l1b.efi_lp.loader import Loader as default_Loader
from geospacelab.datahub.sources.esa_eo.swarm.l1b.efi_lp.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.esa_eo.swarm.l1b.efi_lp.variable_config as var_config
from geospacelab.datahub.sources.esa_eo.swarm.dataset import Dataset as SwarmDataset



default_dataset_attrs = {
    'database': esaeo_database,
    'facility': swarm_facility,
    'instrument': 'EFI',
    'product': 'EFI_LP',
    'data_file_ext': '.cdf',
    'product_version': 'latest',
    'data_root_dir': prf.datahub_data_root_dir / 'ESA' / 'SWARM' / 'Level1b' / 'EFI_LP',
    'allow_load': True,
    'allow_download': True,
    'force_download': False,
    'data_search_recursive': False,
    'add_AACGM': False,
    'add_APEX': False,
    'add_GEO_LST': True,
    'quality_control': False,
    'calib_control': False,
    'label_fields': ['database', 'facility', 'instrument', 'product'],
    'load_mode': 'AUTO',
    'time_clip': True,
}

default_variable_names = [
    'SC_DATETIME',
    'Sync_Status',
    'SC_GEO_LAT',
    'SC_GEO_LON',
    'SC_GEO_r',
    'v_SC_ITRF',
    'n_i',
    'CALIB_n_i',
    'n_i_err',
    'n_e',
    'n_e_err',
    'T_e',
    'T_e_err',
    'CALIB_T_e',
    'V_SC',
    'V_SC_err',
    'FLAG_LP',
    'FLAG_n_i',
    'FLAG_n_e',
    'FLAG_T_e',
    'FLAG_V_SC',
    'FLAG_BITS_1',
    'FLAG_BITS_2',
    'FLAG_1_BIN_AUX',
    'FLAG_2_BIN_AUX',
    'FLAG_1_BIN_IND',
    'FLAG_2_BIN_IND',
    'Gamma_1',
    'Gamma_2',
]


# default_data_search_recursive = True

default_attrs_required = []


class Dataset(SwarmDataset):
    _default_variable_names = default_variable_names
    _default_dataset_attrs = default_dataset_attrs
    _default_downloader = default_Downloader
    _default_loader = default_Loader
    _default_variable_config = var_config
    
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **Dataset._default_dataset_attrs)
        super().__init__(**kwargs)
        
    def load_data(self, **kwargs):
        kwargs.setdefault('omit_join_variables', ['FLAG_1_BIN_IND', 'FLAG_2_BIN_IND'])
        return super().load_data(**kwargs)
    
    def search_data_files(self, file_patterns=None, file_name_by_day=True, archive_yearly=True, **kwargs):
        file_patterns = ['EFI' + self.sat_id.upper(), 'LP']
        return super().search_data_files(
            file_patterns=file_patterns, 
            file_name_by_day=file_name_by_day, 
            archive_yearly=archive_yearly, 
            **kwargs)
    
    def time_filter_by_range(self, **kwargs):
        kwargs.update({'var_datetime_name': 'SC_DATETIME'})
        super().time_filter_by_range(**kwargs)
    
    def calc_GEO_LST(self, var_name_datetime='SC_DATETIME', var_name_glon='SC_GEO_LON'):
        return super().calc_GEO_LST(var_name_datetime, var_name_glon)
    
    def convert_to_APEX(self, var_name_glat='SC_GEO_LAT', var_name_glon='SC_GEO_LON', var_name_gr='SC_GEO_r', var_name_datetime='SC_DATETIME'):
        return super().convert_to_APEX(var_name_glat, var_name_glon, var_name_gr, var_name_datetime)
    
    def convert_to_AACGM(self, var_name_glat='SC_GEO_LAT', var_name_glon='SC_GEO_LON', var_name_gr='SC_GEO_r', var_name_datetime='SC_DATETIME'):
        return super().convert_to_AACGM(var_name_glat, var_name_glon, var_name_gr, var_name_datetime)
    