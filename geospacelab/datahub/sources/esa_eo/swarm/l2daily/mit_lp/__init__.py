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
from geospacelab.datahub.sources.esa_eo.swarm.l2daily.mit_lp.loader import Loader as default_Loader
from geospacelab.datahub.sources.esa_eo.swarm.l2daily.mit_lp.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.esa_eo.swarm.l2daily.mit_lp.variable_config as var_config
from geospacelab.datahub.sources.esa_eo.swarm.dataset import Dataset as SwarmDataset



default_dataset_attrs = {
    'database': esaeo_database,
    'facility': swarm_facility,
    'instrument': 'MAG',
    'product': 'MIT_LP',
    'data_file_ext': '.cdf',
    'product_version': 'latest',
    'data_root_dir': prf.datahub_data_root_dir / 'ESA' / 'SWARM' / 'Level2daily' / 'MIT_LP',
    'allow_load': True,
    'allow_download': True,
    'force_download': False,
    'data_search_recursive': False,
    'add_AACGM': False,
    'add_APEX': False,
    'add_GEO_LST': False,
    'quality_control': False,
    'calib_control': False,
    'label_fields': ['database', 'facility', 'instrument', 'product'],
    'load_mode': 'AUTO',
    'time_clip': True,
}

default_variable_names = [
    'DATETIME',
    'Counter',
    'GEO_LAT',
    'GEO_LON',
    'GEO_r',
    'QD_LAT',
    'QD_LON',
    'QD_MLT',
    'L',
    'SZA',
    'n_e',
    'T_e',
    'Depth',
    'DR',
    'Width',
    'dL',
    'PW_Gradient',
    'EW_Gradient',
    'QUALITY',
    'QUALITY_IND',
    'DATETIME_ID',
    'GEO_LAT_ID',
    'GEO_LON_ID',
    'GEO_R_ID',
    'QD_LAT_ID',
    'QD_LON_ID',
    'QD_MLT_ID',
    'L_ID',
    'SZA_ID',
    'n_e_ID',
    'T_e_ID',
    'Position_Quality_ID',
    'ID_IND',
]

# default_data_search_recursive = True

default_attrs_required = ['sat_id']


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
        kwargs.setdefault('omit_join_variables', ['QUALITY_IND', 'ID_IND'])
        return super().load_data(**kwargs)
    
    def search_data_files(self, file_patterns=None, file_name_by_day=True, archive_yearly=True, **kwargs):
        file_patterns = ['MIT' + self.sat_id.upper(), 'LP']
        return super().search_data_files(
            file_patterns=file_patterns, 
            file_name_by_day=file_name_by_day, 
            archive_yearly=archive_yearly, 
            **kwargs)
    
    def time_filter_by_range(self, **kwargs):
        kwargs.update({'var_datetime_name': 'DATETIME'})
        super().time_filter_by_range(**kwargs)
    
    def calc_GEO_LST(self, var_name_datetime='DATETIME', var_name_glon='GEO_LON'):
        return super().calc_GEO_LST(var_name_datetime, var_name_glon)
    
    def convert_to_APEX(self, var_name_glat='GEO_LAT', var_name_glon='GEO_LON', var_name_gr='GEO_r', var_name_datetime='DATETIME'):
        return super().convert_to_APEX(var_name_glat, var_name_glon, var_name_gr, var_name_datetime)
    
    def convert_to_AACGM(self, var_name_glat='GEO_LAT', var_name_glon='GEO_LON', var_name_gr='GEO_r', var_name_datetime='DATETIME'):
        return super().convert_to_AACGM(var_name_glat, var_name_glon, var_name_gr, var_name_datetime)
    