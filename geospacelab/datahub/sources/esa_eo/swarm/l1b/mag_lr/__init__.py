# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import numpy as np
import datetime
import copy
import re

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel, InstrumentModel, ProductModel
from geospacelab.datahub.sources.esa_eo import esaeo_database
from geospacelab.datahub.sources.esa_eo.swarm import swarm_facility
from geospacelab.config import prf
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool
from geospacelab.datahub.sources.esa_eo.swarm.l1b.mag_lr.loader import Loader as default_Loader
from geospacelab.datahub.sources.esa_eo.swarm.l1b.mag_lr.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.esa_eo.swarm.l1b.mag_lr.variable_config as var_config
from geospacelab.datahub.sources.esa_eo.swarm.dataset import Dataset as SwarmDataset



default_dataset_attrs = {
    'database': esaeo_database,
    'facility': swarm_facility,
    'instrument': 'MAG',
    'product': 'MAG_LR',
    'data_file_ext': '.cdf',
    'product_version': 'latest',
    'data_root_dir': prf.datahub_data_root_dir / 'ESA' / 'SWARM' / 'Level1b' / 'MAG_LR',
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
    'SYNC_STATUS',
    'SC_GEO_LAT',
    'SC_GEO_LON',
    'SC_GEO_r',
    'B_VFM',
    'B_VFM_x',
    'B_VFM_y',
    'B_VFM_z',
    'B_VFM_x_err',
    'B_VFM_y_err',
    'B_VFM_z_err',
    'B_NEC',
    'B_N',
    'B_E',
    'B_C',
    'dB_Sun_VFM',
    'dB_Sun_VFM_x',
    'dB_Sun_VFM_y',
    'dB_Sun_VFM_z',
    'dB_AOCS_VFM',
    'dB_AOCS_VFM_x',
    'dB_AOCS_VFM_y',
    'dB_AOCS_VFM_z',
    'dB_other_VFM',
    'dB_other_VFM_x',
    'dB_other_VFM_y',
    'dB_other_VFM_z',   
    'B_VFM_err',
    'q_NEC_CRF',
    'Att_error',
    'FLAG_B',
    'FLAG_q',
    'FLAG_Platform',
    'FLAG_B_BIN_AUX',
    'FLAG_B_BIN_IND',
    'FLAG_q_BIN_AUX',
    'FLAG_q_BIN_IND',
    'FLAG_Platform_BIN_AUX',
    'FLAG_Platform_BIN_IND',
    'FLAG_F',
    'FLAG_F_BIN_AUX',
    'FLAG_F_BIN_IND',
    'ASM_Freq_Dev',
    'F',
    'F_err',
    'dF_Sun',
    'dF_AOCS',
    'dF_other',
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
        kwargs.setdefault('omit_join_variables', ['FLAG_B_BIN_IND', 'FLAG_q_BIN_IND', 'FLAG_Platform_BIN_IND', 'FLAG_F_BIN_IND'])
        return super().load_data(**kwargs)
    
    def search_data_files(self, file_patterns=None, file_name_by_day=True, archive_yearly=True, **kwargs):
        file_patterns = ['MAG' + self.sat_id.upper(), 'LR']
        super().search_data_files(
            file_patterns=file_patterns, 
            file_name_by_day=file_name_by_day, 
            archive_yearly=archive_yearly, 
            **kwargs)
        file_paths = []
        versions = []
        for fp, version in zip(self.data_file_paths, self.data_file_versions):
            if 'ASM_VFM_IC' in fp.name:
                continue
            file_paths.append(fp)
            versions.append(version)
        self.data_file_paths = file_paths
        self.data_file_versions = versions

    def time_filter_by_range(self, **kwargs):
        kwargs.update({'var_datetime_name': 'SC_DATETIME'})
        super().time_filter_by_range(**kwargs)
    
    def calc_GEO_LST(self, var_name_datetime='SC_DATETIME', var_name_glon='SC_GEO_LON'):
        return super().calc_GEO_LST(var_name_datetime, var_name_glon)
    
    def convert_to_APEX(self, var_name_glat='SC_GEO_LAT', var_name_glon='SC_GEO_LON', var_name_gr='SC_GEO_r', var_name_datetime='SC_DATETIME'):
        return super().convert_to_APEX(var_name_glat, var_name_glon, var_name_gr, var_name_datetime)
    
    def convert_to_AACGM(self, var_name_glat='SC_GEO_LAT', var_name_glon='SC_GEO_LON', var_name_gr='SC_GEO_r', var_name_datetime='SC_DATETIME'):
        return super().convert_to_AACGM(var_name_glat, var_name_glon, var_name_gr, var_name_datetime)
    
    def _load_from_HAPI(self, **kwargs):
        return super()._load_from_HAPI(**kwargs)
    
    def _load_from_VirES(self, **kwargs):
        kwargs.update(kwargs_VirES=self.kwargs_VirES)
        
        load_obj = self.loader(
            dt_fr=self.dt_fr, dt_to=self.dt_to, 
            sat_id=self.sat_id, 
            from_VirES=True, 
            from_FAST=self.from_FAST, 
            **kwargs,)
        
        variables = load_obj.variables
        configured_variables = self._default_variable_config.configured_variables
        
        for vn in variables:
            if vn in self._default_variable_names:
                self.add_variable(vn, configured_variables=configured_variables)
                self[vn].value = variables[vn]
            else:
                if vn.endswith('_N') or vn.endswith('_E') or vn.endswith('_C'):
                    pattern = re.sub(r'_[NEC]$', '', vn)
                    pattern = pattern.replace('B_res_', '')
                    pattern = re.sub('^B_', '', pattern)
                elif vn.endswith('_N_err') or vn.endswith('_E_err') or vn.endswith('_C_err'):
                    pattern = re.sub(r'_[NEC]_err$', '', vn)
                    pattern = re.sub('^B_', '', pattern)
                elif vn.endswith('_VFM_x') or vn.endswith('_VFM_y') or vn.endswith('_VFM_z'):
                    pattern = re.sub(r'_VFM_[xyz]$', '', vn)
                    pattern = re.sub('^B_', '', pattern)
                    pattern = re.sub('^dB_Sun_', '', pattern)
                    pattern = re.sub('^dB_AOCS_', '', pattern)
                    pattern = re.sub('^dB_other_', '', pattern)
                elif vn.endswith('_VFM_x_err') or vn.endswith('_VFM_y_err') or vn.endswith('_VFM_z_err'):
                    pattern = re.sub(r'_VFM_[xyz]_err$', '', vn)
                    pattern = re.sub('^B_', '', pattern)
                    pattern = re.sub('^dB_Sun_', '', pattern)
                    pattern = re.sub('^dB_AOCS_', '', pattern)
                    pattern = re.sub('^dB_other_', '', pattern)
                elif vn.startswith('F_') or vn.startswith('dF_'):
                    pattern = vn.replace('F_res_', '')
                    pattern = re.sub(r'^F_', '', pattern)
                    pattern = re.sub(r'^dF_Sun_', '', pattern)
                    pattern = re.sub(r'^dF_AOCS_', '', pattern)
                    pattern = re.sub(r'^dF_other_', '', pattern)
                else:
                    mylog.StreamLogger.warning(f"Variable {vn} is not in the default variable names and does not match the patterns for automatically assigning configured variable names. It will be added without a configured variable name, and may not be included in the default panels for plotting and analysis. Please check if this variable should be included in the default variable names or if it follows the naming patterns for automatic assignment of configured variable names.")
                    self.add_variable(vn, configured_variables=configured_variables)
                    self[vn].value = variables[vn]
                    continue
                configured_variable_name = vn.replace(pattern, '').replace('__', '_').strip('_')   
                if '_res' in configured_variable_name:
                    configured_variable_name = configured_variable_name.replace('_res', '')     
                self.add_variable(vn, configured_variable_name=configured_variable_name, configured_variables=configured_variables)
                self[vn].value = variables[vn]
                
                if '_res_' in vn:
                    vn = vn.replace('_NEC', '')
                    self[vn].label = pattern + ' ' +  r'$\Delta$' + self[vn].label
                    self[vn].group = r'$\Delta$' + self[vn].group
                else:
                    self[vn].label = pattern + ' ' + self[vn].label
                
        return
    
    def _load_from_HAPI(self, **kwargs):
        
        kwargs.update(kwargs_VirES=self.kwargs_VirES)
        
        load_obj = self.loader(
            dt_fr=self.dt_fr, dt_to=self.dt_to, 
            sat_id=self.sat_id, 
            from_HAPI=True, 
            from_FAST=self.from_FAST, 
            **kwargs,)
        
        variables = load_obj.variables
        configured_variables = self._default_variable_config.configured_variables
        
        for vn in variables:
            if vn in self._default_variable_names:
                self.add_variable(vn, configured_variables=configured_variables)
                self[vn].value = variables[vn]
            else:
                if vn.endswith('_N') or vn.endswith('_E') or vn.endswith('_C'):
                    pattern = re.sub(r'_[NEC]$', '', vn)
                    pattern = pattern.replace('B_res_', '')
                    pattern = re.sub('^B_', '', pattern)
                elif vn.endswith('_N_err') or vn.endswith('_E_err') or vn.endswith('_C_err'):
                    pattern = re.sub(r'_[NEC]_err$', '', vn)
                    pattern = re.sub('^B_', '', pattern)
                elif vn.endswith('_VFM_x') or vn.endswith('_VFM_y') or vn.endswith('_VFM_z'):
                    pattern = re.sub(r'_VFM_[xyz]$', '', vn)
                    pattern = re.sub('^B_', '', pattern)
                    pattern = re.sub('^dB_Sun_', '', pattern)
                    pattern = re.sub('^dB_AOCS_', '', pattern)
                    pattern = re.sub('^dB_other_', '', pattern)
                elif vn.endswith('_VFM_x_err') or vn.endswith('_VFM_y_err') or vn.endswith('_VFM_z_err'):
                    pattern = re.sub(r'_VFM_[xyz]_err$', '', vn)
                    pattern = re.sub('^B_', '', pattern)
                    pattern = re.sub('^dB_Sun_', '', pattern)
                    pattern = re.sub('^dB_AOCS_', '', pattern)
                    pattern = re.sub('^dB_other_', '', pattern)
                elif vn.startswith('F_') or vn.startswith('dF_'):
                    pattern = vn.replace('F_res_', '')
                    pattern = re.sub(r'^F_', '', pattern)
                    pattern = re.sub(r'^dF_Sun_', '', pattern)
                    pattern = re.sub(r'^dF_AOCS_', '', pattern)
                    pattern = re.sub(r'^dF_other_', '', pattern)
                else:
                    mylog.StreamLogger.warning(f"Variable {vn} is not in the default variable names and does not match the patterns for automatically assigning configured variable names. It will be added without a configured variable name, and may not be included in the default panels for plotting and analysis. Please check if this variable should be included in the default variable names or if it follows the naming patterns for automatic assignment of configured variable names.")
                    self.add_variable(vn, configured_variables=configured_variables)
                    self[vn].value = variables[vn]
                    continue
                configured_variable_name = vn.replace(pattern, '').replace('__', '_').strip('_')   
                if '_res' in configured_variable_name:
                    configured_variable_name = configured_variable_name.replace('_res', '')     
                self.add_variable(vn, configured_variable_name=configured_variable_name, configured_variables=configured_variables)
                self[vn].value = variables[vn]
                
                if '_res_' in vn:
                    vn = vn.replace('_NEC', '')
                    self[vn].label = pattern + ' ' +  r'$\Delta$' + self[vn].label
                    self[vn].group = r'$\Delta$' + self[vn].group
                else:
                    self[vn].label = pattern + ' ' + self[vn].label