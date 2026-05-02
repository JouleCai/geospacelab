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
from geospacelab.datahub.sources.esa_eo.swarm.l2daily.aob_fac.loader import Loader as default_Loader
from geospacelab.datahub.sources.esa_eo.swarm.l2daily.aob_fac.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.esa_eo.swarm.l2daily.aob_fac.variable_config as var_config
from geospacelab.datahub.sources.esa_eo.swarm.dataset import Dataset as SwarmDataset



default_dataset_attrs = {
    'database': esaeo_database,
    'facility': swarm_facility,
    'instrument': 'MAG',
    'product': 'AOB_FAC',
    'data_file_ext': '.cdf',
    'product_version': 'latest',
    'data_root_dir': prf.datahub_data_root_dir / 'ESA' / 'SWARM' / 'Level2daily' / 'AOB_FAC',
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
    'SC_DATETIME',
    'SC_GEO_LAT',
    'SC_GEO_LON',
    'SC_GEO_r',
    'SC_QD_LAT',
    'SC_QD_LON',
    'SC_QD_MLT',
    'BOUNDARY_FLAG',
    'QUALITY_Pa',
    'QUALITY_Sigma',
    'PAIR_INDICATOR',
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
        super().load_data(**kwargs)
        self._get_boundaries()
     
    def _get_boundaries(self):
        
        inds_PB = np.array([], dtype=int)
        inds_EB = np.array([], dtype=int)
        skip = False
        bfs = self['BOUNDARY_FLAG'].flatten()
        pis = self['PAIR_INDICATOR'].flatten()
        for i, (bf, pi) in enumerate(zip(bfs, pis)):
            
            if skip:
                skip = False
                continue
       
            if bf == 1:
                inds_EB = np.append(inds_EB, i)
                wait_PB = True
            elif bf == 2:
                inds_PB = np.append(inds_PB, i)
                wait_PB = False
            else:
                continue
            if pi == 1 and i < len(pis) - 1:
                if pis[i+1] == -1:
                    skip = True
                    if wait_PB:
                        inds_PB = np.append(inds_PB, i+1)
                    else:
                        inds_EB = np.append(inds_EB, i+1)   
            else:
                if wait_PB:
                    inds_PB = np.append(inds_PB, -1)
                else:
                    inds_EB = np.append(inds_EB, -1)
                skip = False
        dts_EB = np.empty_like(inds_EB, dtype=object,)
        glats_EB = np.full_like(inds_EB, np.nan, dtype=float)
        glons_EB = np.full_like(inds_EB, np.nan, dtype=float)
        grs_EB = np.full_like(inds_EB, np.nan, dtype=float)
        qd_lats_EB = np.full_like(inds_EB, np.nan, dtype=float)
        qd_lons_EB = np.full_like(inds_EB, np.nan, dtype=float)
        qd_mlts_EB = np.full_like(inds_EB, np.nan, dtype=float)
        q_pa_EB = np.full_like(inds_EB, np.nan, dtype=float)
        q_sigma_EB = np.full_like(inds_EB, np.nan, dtype=float)
        
        dts_PB = np.empty_like(inds_PB, dtype=object,)
        glats_PB = np.full_like(inds_PB, np.nan, dtype=float)
        glons_PB = np.full_like(inds_PB, np.nan, dtype=float)
        grs_PB = np.full_like(inds_PB, np.nan, dtype=float)
        qd_lats_PB = np.full_like(inds_PB, np.nan, dtype=float)
        qd_lons_PB = np.full_like(inds_PB, np.nan, dtype=float)
        qd_mlts_PB = np.full_like(inds_PB, np.nan, dtype=float)
        q_pa_PB = np.full_like(inds_PB, np.nan, dtype=float)
        q_sigma_PB = np.full_like(inds_PB, np.nan, dtype=float)
        
        for ii, (i_EB, i_PB) in enumerate(zip(inds_EB, inds_PB)):
            if i_EB != -1:
                dts_EB[ii] = self['SC_DATETIME'].flatten()[i_EB]
                glats_EB[ii] = self['SC_GEO_LAT'].flatten()[i_EB]
                glons_EB[ii] = self['SC_GEO_LON'].flatten()[i_EB]
                grs_EB[ii] = self['SC_GEO_r'].flatten()[i_EB]
                qd_lats_EB[ii] = self['SC_QD_LAT'].flatten()[i_EB]
                qd_lons_EB[ii] = self['SC_QD_LON'].flatten()[i_EB]
                qd_mlts_EB[ii] = self['SC_QD_MLT'].flatten()[i_EB]
                q_pa_EB[ii] = self['QUALITY_Pa'].flatten()[i_EB]
                q_sigma_EB[ii] = self['QUALITY_Sigma'].flatten()[i_EB]
            if i_PB != -1:
                dts_PB[ii] = self['SC_DATETIME'].flatten()[i_PB]
                glats_PB[ii] = self['SC_GEO_LAT'].flatten()[i_PB]
                glons_PB[ii] = self['SC_GEO_LON'].flatten()[i_PB]
                grs_PB[ii] = self['SC_GEO_r'].flatten()[i_PB]
                qd_lats_PB[ii] = self['SC_QD_LAT'].flatten()[i_PB]
                qd_lons_PB[ii] = self['SC_QD_LON'].flatten()[i_PB]
                qd_mlts_PB[ii] = self['SC_QD_MLT'].flatten()[i_PB]
                q_pa_PB[ii] = self['QUALITY_Pa'].flatten()[i_PB]
                q_sigma_PB[ii] = self['QUALITY_Sigma'].flatten()[i_PB]
            if not isinstance(dts_EB[ii], datetime.datetime):
                dts_EB[ii] = dts_PB[ii]
            if not isinstance(dts_PB[ii], datetime.datetime):
                dts_PB[ii] = dts_EB[ii]
        var = var_config.configured_variables['SC_DATETIME_EB'].clone()
        var.value = dts_EB[:, np.newaxis]
        self['SC_DATETIME_EB'] = var
        var = var_config.configured_variables['SC_GEO_LAT_EB'].clone()
        var.value = glats_EB[:, np.newaxis]
        self['SC_GEO_LAT_EB'] = var
        var = var_config.configured_variables['SC_GEO_LON_EB'].clone()
        var.value = glons_EB[:, np.newaxis]
        self['SC_GEO_LON_EB'] = var
        var = self['SC_GEO_r'].clone()
        var.value = grs_EB[:, np.newaxis]
        self['SC_GEO_r_EB'] = var
        var = var_config.configured_variables['SC_QD_LAT_EB'].clone()
        var.value = qd_lats_EB[:, np.newaxis]
        self['SC_QD_LAT_EB'] = var
        var = var_config.configured_variables['SC_QD_LON_EB'].clone()
        var.value = qd_lons_EB[:, np.newaxis]
        self['SC_QD_LON_EB'] = var
        var = var_config.configured_variables['SC_QD_MLT_EB'].clone()
        var.value = qd_mlts_EB[:, np.newaxis]
        self['SC_QD_MLT_EB'] = var
        var = var_config.configured_variables['QUALITY_Pa_EB'].clone()
        var.value = q_pa_EB[:, np.newaxis]
        self['QUALITY_Pa_EB'] = var
        var = var_config.configured_variables['QUALITY_Sigma_EB'].clone()
        var.value = q_sigma_EB[:, np.newaxis]
        self['QUALITY_Sigma_EB'] = var  
        
        var = var_config.configured_variables['SC_DATETIME_PB'].clone()
        var.value = dts_PB[:, np.newaxis]
        self['SC_DATETIME_PB'] = var
        var = var_config.configured_variables['SC_GEO_LAT_PB'].clone()
        var.value = glats_PB[:, np.newaxis]
        self['SC_GEO_LAT_PB'] = var
        var = var_config.configured_variables['SC_GEO_LON_PB'].clone()
        var.value = glons_PB[:, np.newaxis] 
        self['SC_GEO_LON_PB'] = var
        var = self['SC_GEO_r'].clone()
        var.value = grs_PB[:, np.newaxis]
        self['SC_GEO_r_PB'] = var
        var = var_config.configured_variables['SC_QD_LAT_PB'].clone()
        var.value = qd_lats_PB[:, np.newaxis]
        self['SC_QD_LAT_PB'] = var
        var = var_config.configured_variables['SC_QD_LON_PB'].clone()
        var.value = qd_lons_PB[:, np.newaxis]
        self['SC_QD_LON_PB'] = var
        var = var_config.configured_variables['SC_QD_MLT_PB'].clone()
        var.value = qd_mlts_PB[:, np.newaxis]
        self['SC_QD_MLT_PB'] = var
        var = var_config.configured_variables['QUALITY_Pa_PB'].clone()
        var.value = q_pa_PB[:, np.newaxis]
        self['QUALITY_Pa_PB'] = var
        var = var_config.configured_variables['QUALITY_Sigma_PB'].clone()
        var.value = q_sigma_PB[:, np.newaxis]
        self['QUALITY_Sigma_PB'] = var
        
        self.calc_GEO_LST(var_name_datetime='SC_DATETIME_EB', var_name_glon='SC_GEO_LON_EB')
        self.calc_GEO_LST(var_name_datetime='SC_DATETIME_PB', var_name_glon='SC_GEO_LON_PB')
        
        if self.add_APEX:
            self.convert_to_APEX(
                var_name_glat='SC_GEO_LAT_EB', 
                var_name_glon='SC_GEO_LON_EB', 
                var_name_gr='SC_GEO_r_EB', 
                var_name_datetime='SC_DATETIME_EB',
            )
            self.convert_to_APEX(
                var_name_glat='SC_GEO_LAT_PB', 
                var_name_glon='SC_GEO_LON_PB', 
                var_name_gr='SC_GEO_r_PB', 
                var_name_datetime='SC_DATETIME_PB',
            )
        if self.add_AACGM:
            self.convert_to_AACGM(
                var_name_glat='SC_GEO_LAT_EB', 
                var_name_glon='SC_GEO_LON_EB', 
                var_name_gr='SC_GEO_r_EB', 
                var_name_datetime='SC_DATETIME_EB',
            )
            self.convert_to_AACGM(
                var_name_glat='SC_GEO_LAT_PB', 
                var_name_glon='SC_GEO_LON_PB', 
                var_name_gr='SC_GEO_r_PB', 
                var_name_datetime='SC_DATETIME_PB',
            )    
        
    def search_data_files(self, file_patterns=None, file_name_by_day=False, archive_yearly=True, **kwargs):
        file_patterns = ['AOB' + self.sat_id.upper(), 'FAC']
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
    