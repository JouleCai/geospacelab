# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import datetime
import numpy as np
import cftime
import requests
import bs4
import pathlib
import re
import netCDF4 as nc
import pandas as pd
import ftplib
from contextlib import closing

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.config import prf
import geospacelab.datahub.sources.supermag.supermag_api as smapi

basekeys = ["sme", "sml", "smu", "mlat", "mlt", "glat", "glon", "stid", "num"]
# sunkeys: alias allowed of SUN___ -> ___s
sunkeys = ["smes", "smls", "smus", "mlats", "mlts", "glats", "glons", "stids", "nums"]
# darkkeys: alias allowed of DARK___ -> ___d
darkkeys = ["smed", "smld", "smud", "mlatd", "mltd", "glatd", "glond", "stidd", "numd"]
# regkeys: alias allowed of REGIONAL___ -> ___r
regkeys = ["smer", "smlr", "smur", "mlatr", "mltr", "glatr", "glonr", "stidr", "numr"]
pluskeys = ["smr", "ltsmr", "ltnum", "nsmr"]
indiceskeys = basekeys + sunkeys + darkkeys + regkeys + pluskeys
# 'all' means all the above

imfkeys = ["bgse", "bgsm", "vgse", "vgsm"]  # or imfall for all these
swikeys = ["pdyn", "epsilon", "newell", "clockgse", "clockgsm",
           "density"]  # % or swiall for all these

var_name_dict = {
    'sme': 'SME',
    "sml": 'SML',
    "smu": 'SMU',
    "mlat": 'AACGM_LAT',
    "mlt": 'AACGM_MLT',
    "glat": 'GEO_LAT',
    "glon": 'GEO_LON',
    "stid": 'STATION_ID',
    "num": 'STATION_NUM',
    'smes': 'SME_S',
    "smls": 'SML_S',
    "smus": 'SMU_S',
    "mlats": 'AACGM_LAT_S',
    "mlts": 'AACGM_MLT_S',
    "glats": 'GEO_LAT_S',
    "glons": 'GEO_LON_S',
    "stids": 'STATION_ID_S',
    "nums": 'STATION_NUM_S',
    'smed': 'SME_D',
    "smld": 'SML_D',
    "smud": 'SMU_D',
    "mlatd": 'AACGM_LAT_D',
    "mltd": 'AACGM_MLT_D',
    "glatd": 'GEO_LAT_D',
    "glond": 'GEO_LON_D',
    "stidd": 'STATION_ID_D',
    "numd": 'STATION_NUM_D',
    'smer': 'SME_R',
    "smlr": 'SML_R',
    "smur": 'SMU_R',
    "mlatr": 'AACGM_LAT_R',
    "mltr": 'AACGM_MLT_R',
    "glatr": 'GEO_LAT_R',
    "glonr": 'GEO_LON_R',
    "stidr": 'STATION_ID_R',
    "numr": 'STATION_NUM_R',
    "smr": 'SMR',
    "ltsmr": 'SMR_LT',
    "ltnum": 'LT_NUM',
    "nsmr": 'SMR_STATION_NUM',
}

class Downloader(object):

    def __init__(
            self,
            dt_fr: datetime.datetime,
            dt_to: datetime.datetime,
            user_id: str = None,
            products: str = None, # 'indicesall', 'swiall', 'imfall'
            force_download=False,
            **kwargs
    ):
        if products is None:
            products = ['indicesall']
        self.data_file_root_dir = prf.datahub_data_root_dir / 'SuperMAG' / 'INDICES'
        self.dt_fr = dt_fr
        self.dt_to = dt_to
        if user_id is None:
            self.user_id = prf.user_config['datahub']['supermag']['username']
        self.products = products

        self.force_download = force_download
        self.done = False
        self.download()

    def download(self):
        num_days = dttool.get_diff_days(self.dt_fr, self.dt_to)
        for nd in range(num_days+1):
            this_day = dttool.get_start_of_the_day(self.dt_fr) + datetime.timedelta(days=nd)
            extent = 86400.

            if self.products in [['all'], ['indicesall']]:
                product_name = 'indices'
            else:
                product_name = '_'.join(self.products)
            product_name = product_name.upper()
            file_name = '_'.join([
                'SuperMAG',
                product_name,
                this_day.strftime('%Y%m%d')
            ]) + '.nc'
            file_path = self.data_file_root_dir / this_day.strftime('%Y') / file_name
            if file_path.is_file() and not self.force_download:
                mylog.StreamLogger.info(f'The requested data file already exists! See "{file_path}".')
                self.done = True
                continue
            
            (status, idxdata) = smapi.SuperMAGGetIndices(
             self.user_id,
             this_day,
             extent,
             ','.join(self.products),
             FORMAT='list'
            )
            if status == 1:
                self.save_to_nc(idxdata, file_path)
            else:
                self.done = False
                mylog.StreamLogger.error(f'The requested data cannot be downloaded!')
                return
            self.done = True

    def save_to_nc(self, idxdata, file_path):
        def sm_t_to_datetime(tval):
            jd = (tval / 86400.0) + 2440587.5
            timestamp = pd.to_datetime(jd, unit='D', origin='julian')  # format YYYY-MM-DD HH:MM:SS.ssssss
            return timestamp.to_pydatetime()
        vars = {}
        dts = []
        dt_0 = dttool.get_start_of_the_day(sm_t_to_datetime(idxdata[0]['tval']))
        for item in idxdata:
            dt_c = sm_t_to_datetime(item['tval'])
            dt_c = dt_0 + datetime.timedelta(minutes=np.around((dt_c - dt_0).total_seconds() / 60))
            dts.append(dt_c)

            for key in item.keys():
                if key == 'tval':
                    continue
                value = item[key]
                if type(value) is list:
                    nelem = len(value)
                else:
                    nelem = 1
                    value = [value]
                value = np.array(value)[np.newaxis, :]
                vars.setdefault(key, np.empty((0, nelem), dtype=type(value[0, 0])))
                vars[key] = np.vstack((vars[key], value))
        keys = list(vars.keys())
        for key in keys:
            if 'mlat' in key:
                new_key = key.replace('mlat', '_AACGM_LAT')
            elif 'mlt' in key:
                new_key = key.replace('mlt', '_AACGM_MLT')
            elif 'glat' in key:
                new_key = key.replace('glat', '_GEO_LAT')
            elif 'glon' in key:
                new_key = key.replace('glon', '_GEO_LON')
            elif 'stid' in key:
                new_key = key.replace('stid', '_STATION')
            elif 'smr' in key:
                new_key = key.replace('smr', 'SMR')
            else:
                new_key = key
            if 'num' in new_key:
                new_key = key.replace('num', '_NUM')
            vars[new_key] = vars.pop(key)

        file_path.parent.resolve().mkdir(parents=True, exist_ok=True)
        fnc = nc.Dataset(file_path, 'w')
        fnc.title = "SuperMAG Indices (Daily)"

        num_rows = len(dts)
        time_array = np.array(cftime.date2num(dts, units='seconds since 1970-01-01 00:00:00.0'))
        fnc.createDimension('UNIX_TIME', num_rows)
        fnc.createDimension('SME_R_NUM', vars['SMEr_NUM'].shape[1])
        time = fnc.createVariable('UNIX_TIME', np.float64, ('UNIX_TIME',))
        time.units = 'seconds since 1970-01-01 00:00:00.0'
        time[::] = time_array[::]

        for key in vars.keys():
            var = vars[key]
            if var.shape[1] == 1:
                dim = ('UNIX_TIME', )
            else:
                dim = ('UNIX_TIME', 'SME_R_NUM')
            var_nc = fnc.createVariable(key, var.dtype, dim)
            var_nc[::] = var[::]
        mylog.StreamLogger.info(
            "The requested SuperMAG data has been saved in the file {}.".format(file_path))
        fnc.close()
        self.done = True




