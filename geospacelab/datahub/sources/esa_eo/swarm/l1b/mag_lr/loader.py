# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import numpy as np
import pandas as pd
import datetime

from geospacelab.datahub.sources.esa_eo.swarm.loader import LoaderModel

import geospacelab.toolbox.utilities.pylogging as mylog

# define the default variable name dictionary
default_variable_name_dict = {
    'CDF_EPOCH': 'Timestamp',
    'SYNC_STATUS': 'SyncStatus',
    'SC_GEO_LAT': 'Latitude',
    'SC_GEO_LON': 'Longitude',
    'SC_GEO_R': 'Radius',
    'B_VFM': 'B_VFM',
    'B_NEC': 'B_NEC',
    'dB_Sun_VFM': 'dB_Sun',
    'dB_AOCS_VFM': 'dB_AOCS',
    'dB_other_VFM': 'dB_other',
    'B_VFM_err': 'B_error',
    'q_NEC_CRF': 'q_NEC_CRF',
    'Att_error': 'Att_error',
    'FLAG_B': 'Flags_B',
    'FLAG_q': 'Flags_q',
    'FLAG_Platform': 'Flags_Platform',
    'FLAG_F': 'Flags_F',
    'ASM_Freq_Dev': 'ASM_Freq_Dev',
    'F': 'F',
    'F_err': 'F_error',
    'dF_Sun': 'dF_Sun',
    'dF_AOCS': 'dF_AOCS',
    'dF_other': 'dF_other',
}


class Loader(LoaderModel):
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variable_name_dict', default_variable_name_dict)
        super(Loader, self).__init__(*args, **kwargs)

    def load_data(self, **kwargs):
        
        super(Loader, self).load_data(**kwargs, )
        
        self.variables['SC_GEO_r'] = self.variables['SC_GEO_R'] / 6371.2e3
        self.variables['SC_GEO_LON'] = self.variables['SC_GEO_LON'] % 360
        
        variables_extend = {}
        for vn in self.variables:
            if vn.startswith('B_') or vn.startswith('dB_'):
                if 'NEC' in vn:
                    vn_new_0 = vn.replace('_NEC', '')
                    vn_new = vn_new_0 + '_N'
                    variables_extend[vn_new] = self.variables[vn][:, 0][:, np.newaxis]
                    vn_new = vn_new_0 + '_E'
                    variables_extend[vn_new] = self.variables[vn][:, 1][:, np.newaxis]
                    vn_new = vn_new_0 + '_C'
                    variables_extend[vn_new] = self.variables[vn][:, 2][:, np.newaxis]
                elif 'VFM' in vn:
                    vn_new_x = vn.replace('_VFM', '_VFM_x')
                    variables_extend[vn_new_x] = self.variables[vn][:, 0][:, np.newaxis]
                    vn_new_y = vn.replace('_VFM', '_VFM_y')
                    variables_extend[vn_new_y] = self.variables[vn][:, 1][:, np.newaxis]
                    vn_new_z = vn.replace('_VFM', '_VFM_z')
                    variables_extend[vn_new_z] = self.variables[vn][:, 2][:, np.newaxis]
        self.variables.update(variables_extend)
        
        if 'FLAG_B' in self.variables:
            fb = self.variables['FLAG_B'].flatten()
            fb = (((fb[:,None] & (1 << np.arange(9)))) > 0).astype(int)
            self.variables['FLAG_B_BIN_AUX'] = fb
            self.variables['FLAG_B_BIN_IND'] = np.arange(fb.shape[1]+1)[np.newaxis, :] - 0.5
        
        if 'FLAG_q' in self.variables:
            fb = self.variables['FLAG_q'].flatten()
            fb = (((fb[:,None] & (1 << np.arange(8)))) > 0).astype(int)
            self.variables['FLAG_q_BIN_AUX'] = fb
            self.variables['FLAG_q_BIN_IND'] = np.arange(fb.shape[1]+1)[np.newaxis, :] - 0.5
        
        if 'FLAG_Platform' in self.variables:
            fb = self.variables['FLAG_Platform'].flatten()
            fb = (((fb[:,None] & (1 << np.arange(9)))) > 0).astype(int)
            self.variables['FLAG_Platform_BIN_AUX'] = fb
            self.variables['FLAG_Platform_BIN_IND'] = np.arange(fb.shape[1]+1)[np.newaxis, :] - 0.5
        
        if 'FLAG_F' in self.variables:
            fb = self.variables['FLAG_F'].flatten()
            fb = (((fb[:,None] & (1 << np.arange(9)))) > 0).astype(int)
            self.variables['FLAG_F_BIN_AUX'] = fb
            self.variables['FLAG_F_BIN_IND'] = np.arange(fb.shape[1]+1)[np.newaxis, :] - 0.5

    def load_cdf_data(self, var_names_cdf_epoch=None, var_names_independent_time=None):
       return super().load_cdf_data(var_names_cdf_epoch=var_names_cdf_epoch, var_names_independent_time=var_names_independent_time)
   
    
    def load_from_VirES(self, collection=None, kwargs_products=None):
        
        if collection is None:
            if self.from_FAST:
                collection = "SW_FAST_MAG__LR_1B"  # example collection for FAST data
            else:
                collection = "SW_OPER_MAG__LR_1B"  # example collection for Swarm L1b data
            
            collection = collection.replace("MAG_", "MAG"+self.sat_id)
        
        kwargs_products_default = {
            "measurements": [
                'B_VFM', 'B_NEC', 'dB_Sun', 'dB_AOCS', 'dB_other', 'B_error',
                'q_NEC_CRF', 'Att_error',
                'Flags_B', 'Flags_q', 'Flags_Platform', 'Flags_F',
                'ASM_Freq_Dev', 'F', 'F_error', 'dF_Sun', 'dF_AOCS', 'dF_other',
            ],
            "models": [
                'CHAOS-Core',
            ],
            "residuals": False,
        }
        
        if kwargs_products is not None:
            kwargs_products_default.update(kwargs_products)
        
        data = super().load_from_VirES(collection=collection, kwargs_products=kwargs_products_default)
        ds = data.as_xarray()
        
        var_names_xarray = list(ds.keys())
        
        for vn_xarray in var_names_xarray:
            # print(vn_xarray)
            if vn_xarray in self.variable_name_dict.values():
                vn_c = [vn for vn, vn_x in self.variable_name_dict.items() if vn_x == vn_xarray][0]
                self.variables[vn_c] = ds[vn_xarray].values
            else:
                res = [ (vn, vn_s) for vn, vn_s in self.variable_name_dict.items() if vn_s in vn_xarray]
                if len(res) == 1:
                    vn_c, vn_s = res[0]
                    vn_c_new = vn_xarray.replace(vn_s, vn_c)
                    self.variables[vn_c_new] = ds[vn_xarray].values
                else:
                    mylog.StreamLogger.warning(f"Variable name {vn_xarray} not found in the variable name dictionary, and no unique match found. Skipping this variable.")
        
        for vn, value in self.variables.items():
            if value.ndim == 1:
                self.variables[vn] = value[:, np.newaxis]
        self.variables['SC_DATETIME'] = pd.to_datetime(ds['Timestamp'].values).to_pydatetime()[:, np.newaxis]
        
    def load_from_HAPI(
            self, 
            server="https://vires.services/hapi", 
            dataset=None,   
            parameters=None):

        parameters_default = ["Timestamp,Latitude,Longitude,Radius",
            "F,dF_Sun,dF_AOCS,dF_other,F_error,B_VFM,B_NEC,dB_Sun,dB_AOCS,dB_other,B_error",
            "q_NEC_CRF,Att_error,Flags_F,Flags_B,Flags_q,Flags_Platform,ASM_Freq_Dev",
            "SyncStatus,B_NEC_Model,F_Model,F_res_Model,B_NEC_res_Model"]
        if dataset is None:
            if self.from_FAST:
                dataset = "SW_FAST_MAG__LR_1B"  # example collection for FAST data
            else:
                dataset = "SW_OPER_MAG__LR_1B"  # example collection for Swarm L1b data
            
            dataset = dataset.replace("MAG_", "MAG"+self.sat_id)

        if parameters is None:
            parameters = parameters_default
        if isinstance(parameters, str):
            parameters = [parameters]
            
        for pp in parameters:
        
            data, meta = super().load_from_HAPI(server=server, dataset=dataset, parameters=pp)
        
            for vn_hapi in data.dtype.names:
                if vn_hapi in self.variable_name_dict.values():
                    vn_c = [vn for vn, vn_x in self.variable_name_dict.items() if vn_x == vn_hapi][0]
                    self.variables[vn_c] = data[vn_hapi]
                else:
                    res = [ (vn, vn_s) for vn, vn_s in self.variable_name_dict.items() if vn_s in vn_hapi]
                    if len(res) == 1:
                        vn_c, vn_s = res[0]
                        vn_c_new = vn_hapi.replace(vn_s, vn_c)
                        self.variables[vn_c_new] = data[vn_hapi]
                    else:
                        mylog.StreamLogger.warning(f"Variable name {vn_hapi} not found in the variable name dictionary, and no unique match found. Skipping this variable.")
            if 'Timestamp' in data.dtype.names:
                dts = [t.decode('utf-8') for t in data['Timestamp']]
                dts = pd.to_datetime(dts).to_pydatetime()
                dts = [dt.replace(tzinfo=None) for dt in dts]
                self.variables['SC_DATETIME'] = np.array(dts)[:, np.newaxis]
        
        for vn, value in self.variables.items():
            if value.ndim == 1:
                self.variables[vn] = value[:, np.newaxis]