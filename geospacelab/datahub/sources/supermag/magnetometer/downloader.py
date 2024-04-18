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

var_name_dict = {
    'mlat': 'AACGM_LAT',
    'mlon': 'AACGM_LON',
    'mlt': 'AACGM_MLT',
    'sza': 'SZA',
    'decl': 'DECLINATION',
    'N_GEO': 'N_GEO',
    'E_GEO': 'E_GEO',
    'Z_GEO': 'Z_GEO',
    'N_MAG': 'N_MAG',
    'E_MAG': 'E_MAG',
    'Z_MAG': 'Z_MAG',
}


class Downloader(object):

    def __init__(
            self,
            dt_fr: datetime.datetime,
            dt_to: datetime.datetime,
            user_id: str = None,
            site_name: str = None,
            force_download=False,
            baseline='all',
            **kwargs
    ):

        self.data_file_root_dir = prf.datahub_data_root_dir / 'SuperMAG' / 'sites'
        self.dt_fr = dt_fr
        self.dt_to = dt_to
        if user_id is None:
            self.user_id = prf.user_config['datahub']['supermag']['username']
        self.site_name = site_name
        self.baseline = baseline

        self.force_download = force_download
        self.done = False

        self.download()


    def download(self):
        diff_month = dttool.get_diff_months(self.dt_fr, self.dt_to)
        for nm in range(diff_month+1):
            this_month = dttool.get_next_n_months(self.dt_fr, nm)
            now = datetime.datetime.now()
            if this_month.year == now.year and this_month.month == now.month:
                self.force_download = True
            next_month = dttool.get_next_n_months(self.dt_fr, nm + 1)
            extent = (next_month - this_month).total_seconds()
            flags = ['MLT', 'MAG', 'GEO', 'DECL', 'SZA',]
            if self.baseline == 'DELTA':
                flags.append('DELTA')
            elif self.baseline in ['all', 'yearly', 'none']:
                bstr = f"baseline={self.baseline}"
                flags.append(bstr)
            flags = ','.join(flags)
            file_name = '_'.join([
                'SuperMAG',
                self.site_name,
                this_month.strftime('%Y%m'),
                'baseline',
                self.baseline.lower()
            ]) + '.nc'
            file_path = self.data_file_root_dir / self.site_name / this_month.strftime('%Y') / file_name
            if file_path.is_file() and not self.force_download:
                mylog.StreamLogger.info(f'The requested data file already exists! See "{file_path}".')
                self.done = True
                continue
            mylog.simpleinfo.info(f"Downloading the {self.site_name} data from SuperMAG ...")
            (status, sm_data) = smapi.SuperMAGGetData(
                self.user_id,
                this_month,
                extent,
                flags,
                self.site_name,
                FORMAT='list')

            if status == 1:
                self.save_to_nc(sm_data, file_path)
            else:
                self.done = False
                mylog.StreamLogger.error(f'The requested data cannot be downloaded!')
                return
            self.done = True

    def save_to_nc(self, sm_data, file_path):
        def sm_t_to_datetime(tval):
            jd = (tval / 86400.0) + 2440587.5
            timestamp = pd.to_datetime(jd, unit='D', origin='julian')
            t = timestamp.to_pydatetime()       # format YYYY-MM-DD HH:MM:SS.ssssss
            if t.microsecond >= 500_000:
                t += datetime.timedelta(seconds=1)
            return t.replace(microsecond=0)

        res = list(map(lambda x: list(x.values()), sm_data))
        keys = list(sm_data[0].keys())
        res = list(zip(*res))

        keys_1d = ['tval', 'mlat', 'mlon', 'mlt', 'decl', 'sza', 'N', 'E', 'Z']

        data = {}
        for k in keys_1d:
            id = keys.index(k)
            if k not in ['N', 'E', 'Z']:
                data[k] = np.array(res[id])
                continue

            rr = list(map(lambda x: list(x.values()), res[id]))
            rr = list(zip(*rr))

            for ik, kk in enumerate(res[id][0].keys()):
                if kk == 'geo':
                    name_key = k + '_GEO'
                elif kk == 'nez':
                    name_key = k + '_MAG'
                else:
                    raise NotImplementedError
                data[name_key] = np.array(rr[ik])

        dts = np.array([sm_t_to_datetime(tval) for tval in data['tval']])

        file_path.parent.resolve().mkdir(parents=True, exist_ok=True)
        fnc = nc.Dataset(file_path, 'w')
        fnc.title = f"SuperMAG magnetometer at {self.site_name} (Daily)"
        fnc.site = self.site_name
        fnc.start_time = dttool.get_start_of_the_day(dts[0]).strftime("%Y-%m-%d %H:%M:%S")
        fnc.end_time = dttool.get_end_of_the_day(dts[-1]).strftime("%Y-%m-%d %H:%M:%S")
        fnc.GEO_LAT = sm_data[0]['glat']
        fnc.GEO_LON = sm_data[0]['glon']
        fnc.time_res = sm_data[0]['ext']
        num_rows = len(dts)
        time_array = np.array(cftime.date2num(dts, units='seconds since 1970-01-01 00:00:00.0'))
        fnc.createDimension('UNIX_TIME', num_rows)
        time = fnc.createVariable('UNIX_TIME', np.float64, ('UNIX_TIME',))
        time.units = 'seconds since 1970-01-01 00:00:00.0'
        time[::] = time_array[::]

        for k, v in var_name_dict.items():
            var = data[k]
            var_nc = fnc.createVariable(v, np.float32, ('UNIX_TIME', ))
            var_nc[::] = var[::]
        mylog.StreamLogger.info(
            "The requested SuperMAG magnetometer data has been saved in the file {}.".format(file_path))
        fnc.close()
        self.done = True


if __name__ == "__main__":
    dt_fr1 = datetime.datetime(2016, 3, 15, 15)
    dt_to1 = datetime.datetime(2016, 3, 15, 22)
    Downloader(dt_fr1, dt_to1, site_name='SKT', force_download=True)



