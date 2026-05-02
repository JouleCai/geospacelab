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
from geospacelab.datahub.sources.esa_eo.swarm.l2daily.aej_pbs.loader import Loader as default_Loader
from geospacelab.datahub.sources.esa_eo.swarm.l2daily.aej_pbs.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.esa_eo.swarm.l2daily.aej_pbs.variable_config as var_config
from geospacelab.datahub.sources.esa_eo.swarm.dataset import Dataset as SwarmDataset



default_dataset_attrs = {
    'database': esaeo_database,
    'facility': swarm_facility,
    'instrument': 'MAG',
    'product': 'AEJ_PBS',
    'data_file_ext': '.cdf',
    'product_version': 'latest',
    'data_root_dir': prf.datahub_data_root_dir / 'ESA' / 'SWARM' / 'Level2daily' / 'AEJ_PBS',
    'allow_load': True,
    'allow_download': True,
    'force_download': False,
    'data_search_recursive': False,
    'add_AACGM': False,
    'add_APEX': False,
    'quality_control': False,
    'calib_control': False,
    'label_fields': ['database', 'facility', 'instrument', 'product'],
    'load_mode': 'AUTO',
    'time_clip': True,
}

default_variable_names = [
    'DATETIME_WEJ_PEAK',
    'GEO_LAT_WEJ_PEAK',
    'GEO_LON_WEJ_PEAK',
    'GEO_ALT_WEJ_PEAK',
    'GEO_r_WEJ_PEAK',
    'QD_LAT_WEJ_PEAK',
    'QD_LON_WEJ_PEAK',
    'QD_MLT_WEJ_PEAK',
    'WEJ_PEAK',
    
    'DATETIME_EEJ_PEAK',
    'GEO_LAT_EEJ_PEAK',
    'GEO_LON_EEJ_PEAK',
    'GEO_ALT_EEJ_PEAK',
    'GEO_r_EEJ_PEAK',
    'QD_LAT_EEJ_PEAK',
    'QD_LON_EEJ_PEAK',
    'QD_MLT_EEJ_PEAK',
    'EEJ_PEAK',
    
    'DATETIME_WEJ_EB',
    'GEO_LAT_WEJ_EB',
    'GEO_LON_WEJ_EB',
    'GEO_ALT_WEJ_EB',
    'GEO_r_WEJ_EB',
    'QD_LAT_WEJ_EB',
    'QD_LON_WEJ_EB',
    'QD_MLT_WEJ_EB',
    
    'DATETIME_EEJ_EB',
    'GEO_LAT_EEJ_EB',
    'GEO_LON_EEJ_EB',
    'GEO_ALT_EEJ_EB',
    'GEO_r_EEJ_EB',
    'QD_LAT_EEJ_EB',
    'QD_LON_EEJ_EB',
    'QD_MLT_EEJ_EB',
    
    'DATETIME_WEJ_PB',
    'GEO_LAT_WEJ_PB',
    'GEO_LON_WEJ_PB',
    'GEO_ALT_WEJ_PB',
    'GEO_r_WEJ_PB',
    'QD_LAT_WEJ_PB',
    'QD_LON_WEJ_PB',
    'QD_MLT_WEJ_PB',
    
    'DATETIME_EEJ_PB',
    'GEO_LAT_EEJ_PB',
    'GEO_LON_EEJ_PB',
    'GEO_ALT_EEJ_PB',
    'GEO_r_EEJ_PB',
    'QD_LAT_EEJ_PB',
    'QD_LON_EEJ_PB',
    'QD_MLT_EEJ_PB',
    
    'QUALITY_FLAG',
    
    'B_N_MAX',
    'B_N_MIN',
    'B_E_MAX',
    'B_E_MIN',
    'GEO_LAT_B_MIN',
    'GEO_LON_B_MIN',
    'GEO_LAT_B_MAX',
    'GEO_LON_B_MAX',
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
        
    def search_data_files(self, file_patterns=None, file_name_by_day=False, archive_yearly=True, **kwargs):
        file_patterns = ['AEJ' + self.sat_id.upper(), 'PBS']
        return super().search_data_files(
            file_patterns=file_patterns, 
            file_name_by_day=file_name_by_day, 
            archive_yearly=archive_yearly, 
            **kwargs)
    
    def time_filter_by_range(self, **kwargs):
        kwargs.update({'var_datetime_name': 'DATETIME_WEJ_PEAK'})
        kwargs.update({'var_names': [
            'DATETIME_WEJ_PEAK', 
            'GEO_LAT_WEJ_PEAK', 'GEO_r_WEJ_PEAK', 'GEO_ALT_WEJ_PEAK', 'GEO_LON_WEJ_PEAK', 
            'QD_LAT_WEJ_PEAK', 'QD_LON_WEJ_PEAK', 'QD_MLT_WEJ_PEAK', 
            'WEJ_PEAK', 'QUALITY_FLAG', 
            'B_N_MAX', 'B_N_MIN', 'B_E_MAX', 'B_E_MIN', 
            'GEO_LAT_B_MIN', 'GEO_LON_B_MIN', 'GEO_LAT_B_MAX', 'GEO_LON_B_MAX']})
        super().time_filter_by_range(**kwargs)
        
        kwargs.update({'var_datetime_name': 'DATETIME_EEJ_PEAK'})
        kwargs.update({'var_names': [
            'DATETIME_EEJ_PEAK', 
            'GEO_LAT_EEJ_PEAK', 'GEO_r_EEJ_PEAK', 'GEO_ALT_EEJ_PEAK', 'GEO_LON_EEJ_PEAK', 
            'QD_LAT_EEJ_PEAK', 'QD_LON_EEJ_PEAK', 'QD_MLT_EEJ_PEAK', 
            'EEJ_PEAK']})
        super().time_filter_by_range(**kwargs)
        
        kwargs.update({'var_datetime_name': 'DATETIME_WEJ_EB'})
        kwargs.update({'var_names': [
            'DATETIME_WEJ_EB', 
            'GEO_LAT_WEJ_EB', 'GEO_r_WEJ_EB', 'GEO_ALT_WEJ_EB', 'GEO_LON_WEJ_EB',
            'QD_LAT_WEJ_EB', 'QD_LON_WEJ_EB', 'QD_MLT_WEJ_EB']})
        super().time_filter_by_range(**kwargs)
        
        kwargs.update({'var_datetime_name': 'DATETIME_EEJ_EB'})
        kwargs.update({'var_names': [
            'DATETIME_EEJ_EB', 
            'GEO_LAT_EEJ_EB', 'GEO_r_EEJ_EB', 'GEO_ALT_EEJ_EB', 'GEO_LON_EEJ_EB',
            'QD_LAT_EEJ_EB', 'QD_LON_EEJ_EB', 'QD_MLT_EEJ_EB']})
        super().time_filter_by_range(**kwargs)
        
        kwargs.update({'var_datetime_name': 'DATETIME_WEJ_PB'})
        kwargs.update({'var_names': [
            'DATETIME_WEJ_PB', 
            'GEO_LAT_WEJ_PB', 'GEO_r_WEJ_PB', 'GEO_ALT_WEJ_PB', 'GEO_LON_WEJ_PB',
            'QD_LAT_WEJ_PB', 'QD_LON_WEJ_PB', 'QD_MLT_WEJ_PB']})
        super().time_filter_by_range(**kwargs)
        
        kwargs.update({'var_datetime_name': 'DATETIME_EEJ_PB'})
        kwargs.update({'var_names': [
            'DATETIME_EEJ_PB', 
            'GEO_LAT_EEJ_PB', 'GEO_r_EEJ_PB', 'GEO_ALT_EEJ_PB', 'GEO_LON_EEJ_PB',
            'QD_LAT_EEJ_PB', 'QD_LON_EEJ_PB', 'QD_MLT_EEJ_PB']})
        super().time_filter_by_range(**kwargs)
    
    def calc_GEO_LST(self, var_name_datetime='DATETIME', var_name_glon='GEO_LON'):
        
        super().calc_GEO_LST(var_name_datetime='DATETIME_WEJ_PEAK', var_name_glon='GEO_LON_WEJ_PEAK')
        super().calc_GEO_LST(var_name_datetime='DATETIME_EEJ_PEAK', var_name_glon='GEO_LON_EEJ_PEAK')
        super().calc_GEO_LST(var_name_datetime='DATETIME_WEJ_EB', var_name_glon='GEO_LON_WEJ_EB')
        super().calc_GEO_LST(var_name_datetime='DATETIME_EEJ_EB', var_name_glon='GEO_LON_EEJ_EB')
        super().calc_GEO_LST(var_name_datetime='DATETIME_WEJ_PB', var_name_glon='GEO_LON_WEJ_PB')
        super().calc_GEO_LST(var_name_datetime='DATETIME_EEJ_PB', var_name_glon='GEO_LON_EEJ_PB')
        
    
    def convert_to_APEX(self, var_name_glat='GEO_LAT', var_name_glon='GEO_LON', var_name_gr='GEO_r', var_name_datetime='DATETIME'):
        super().convert_to_APEX(
            var_name_glat='GEO_LAT_WEJ_PEAK', 
            var_name_glon='GEO_LON_WEJ_PEAK', 
            var_name_gr='GEO_r_WEJ_PEAK', 
            var_name_datetime='DATETIME_WEJ_PEAK')
        super().convert_to_APEX(
            var_name_glat='GEO_LAT_EEJ_PEAK', 
            var_name_glon='GEO_LON_EEJ_PEAK', 
            var_name_gr='GEO_r_EEJ_PEAK', 
            var_name_datetime='DATETIME_EEJ_PEAK')
        super().convert_to_APEX(
            var_name_glat='GEO_LAT_WEJ_EB', 
            var_name_glon='GEO_LON_WEJ_EB', 
            var_name_gr='GEO_r_WEJ_EB', 
            var_name_datetime='DATETIME_WEJ_EB')
        super().convert_to_APEX(
            var_name_glat='GEO_LAT_EEJ_EB', 
            var_name_glon='GEO_LON_EEJ_EB', 
            var_name_gr='GEO_r_EEJ_EB', 
            var_name_datetime='DATETIME_EEJ_EB')
        super().convert_to_APEX(
            var_name_glat='GEO_LAT_WEJ_PB', 
            var_name_glon='GEO_LON_WEJ_PB', 
            var_name_gr='GEO_r_WEJ_PB', 
            var_name_datetime='DATETIME_WEJ_PB')
        super().convert_to_APEX(
            var_name_glat='GEO_LAT_EEJ_PB', 
            var_name_glon='GEO_LON_EEJ_PB', 
            var_name_gr='GEO_r_EEJ_PB', 
            var_name_datetime='DATETIME_EEJ_PB')
    
    def convert_to_AACGM(self, var_name_glat='GEO_LAT', var_name_glon='GEO_LON', var_name_gr='GEO_r', var_name_datetime='DATETIME'):
        super().convert_to_AACGM(
            var_name_glat='GEO_LAT_WEJ_PEAK', 
            var_name_glon='GEO_LON_WEJ_PEAK', 
            var_name_gr='GEO_r_WEJ_PEAK', 
            var_name_datetime='DATETIME_WEJ_PEAK')
        super().convert_to_AACGM(
            var_name_glat='GEO_LAT_EEJ_PEAK', 
            var_name_glon='GEO_LON_EEJ_PEAK', 
            var_name_gr='GEO_r_EEJ_PEAK', 
            var_name_datetime='DATETIME_EEJ_PEAK')
        super().convert_to_AACGM(
            var_name_glat='GEO_LAT_WEJ_EB', 
            var_name_glon='GEO_LON_WEJ_EB', 
            var_name_gr='GEO_r_WEJ_EB', 
            var_name_datetime='DATETIME_WEJ_EB')
        super().convert_to_AACGM(
            var_name_glat='GEO_LAT_EEJ_EB', 
            var_name_glon='GEO_LON_EEJ_EB', 
            var_name_gr='GEO_r_EEJ_EB', 
            var_name_datetime='DATETIME_EEJ_EB')
        super().convert_to_AACGM(
            var_name_glat='GEO_LAT_WEJ_PB', 
            var_name_glon='GEO_LON_WEJ_PB', 
            var_name_gr='GEO_r_WEJ_PB', 
            var_name_datetime='DATETIME_WEJ_PB')
        super().convert_to_AACGM(
            var_name_glat='GEO_LAT_EEJ_PB', 
            var_name_glon='GEO_LON_EEJ_PB', 
            var_name_gr='GEO_r_EEJ_PB', 
            var_name_datetime='DATETIME_EEJ_PB')
    