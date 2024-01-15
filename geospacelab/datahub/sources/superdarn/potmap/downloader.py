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
import requests
import bs4
import pathlib
import re
import netCDF4 as nc
import cftime
import ftplib
from contextlib import closing

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.datahub.sources.wdc as wdc
from geospacelab.config import prf


class Downloader(object):

    def __init__(self, dt_fr,  dt_to, data_res=None, pole='N', data_file_root_dir=None):

        self.dt_fr = dt_fr
        self.dt_to = dt_to

        self.done = False
        if data_res is None:
            data_res = 2    # in minutes
        self.data_res = data_res
        self.pole = pole
        if data_file_root_dir is None:
            self.data_file_root_dir = prf.datahub_data_root_dir / 'SuperDARN' / 'PotentialMap'
        else:
            self.data_file_root_dir = pathlib.Path(data_file_root_dir)

        self.download()

    def download(self):
        diff_days = dttool.get_diff_days(self.dt_fr, self.dt_to)

        for i in range(diff_days + 1):
            dt1 = self.dt_fr + datetime.timedelta(days=i)

            fn = '_'.join(['SuperDARN', 'POTMAP', str(self.data_res) + 'min', dt1.strftime('%Y%m%d'), self.pole]) + '.dat'

            file_path = self.data_file_root_dir / dt1.strftime('%Y') / fn
            self.save_to_netcdf(file_path)

    def save_to_netcdf(self, file_path):
        with open(file_path, 'r') as f:
            text = f.read()

            results = re.findall(
                r'^\s*(\d+)\s*\[(\d+),(\d+)]\s*([-\d.]+)\s*' +
                r'([-\d.]+)\s*([-\d.]+)\s*([-\d.]+)\s*([-\d.]+)\s*([-\d.]+)\s*([-\d.]+)\s*' +
                r'([\S]+)',
                text,
                re.M
            )
            results = list(zip(*results))
            nlat = 40
            nlon = 180
            ntime = len(results[0]) / nlon / nlat
            if ntime != int(ntime):
                raise ValueError
            ntime = int(ntime)
            mlat_arr = np.array(results[3]).reshape([ntime, nlat, nlon], order='C').transpose((0, 2, 1)).astype(np.float32)
            mlon_arr = np.array(results[4]).reshape([ntime, nlat, nlon], order='C').transpose((0, 2, 1)).astype(np.float32)
            EF_N_arr = np.array(results[5]).reshape([ntime, nlat, nlon], order='C').transpose((0, 2, 1)).astype(np.float32)
            EF_E_arr = np.array(results[6]).reshape([ntime, nlat, nlon], order='C').transpose((0, 2, 1)).astype(np.float32)
            v_N_arr = np.array(results[7]).reshape([ntime, nlat, nlon], order='C').transpose((0, 2, 1)).astype(np.float32)
            v_E_arr = np.array(results[8]).reshape([ntime, nlat, nlon], order='C').transpose((0, 2, 1)).astype(np.float32)
            phi_arr = np.array(results[9]).reshape([ntime, nlat, nlon], order='C').transpose((0, 2, 1)).astype(np.float32)

            dts = np.array(results[10])[::nlon * nlat]
            dts = [datetime.datetime.strptime(dtstr, "%Y-%m-%d/%H:%M:%S") for dtstr in dts]
            time_array = np.array(cftime.date2num(dts, units='seconds since 1970-01-01 00:00:00.0'))

            import aacgmv2
            mlt_arr = np.empty_like(mlat_arr)
            for i in range(ntime):
                mlt1 = aacgmv2.convert_mlt(mlon_arr[i].flatten(), dts[i]).reshape((nlon, nlat))
                mlt_arr[i, ::] = mlt1[::]

            fp = pathlib.Path(file_path.with_suffix('.nc'))
            fp.parent.resolve().mkdir(parents=True, exist_ok=True)
            fnc = nc.Dataset(fp, 'w')
            fnc.createDimension('UNIX_TIME', ntime)
            fnc.createDimension('MLAT', nlat)
            fnc.createDimension('MLON', nlon)

            fnc.title = "SuperDARN Potential maps"

            time = fnc.createVariable('UNIX_TIME', np.float64, ('UNIX_TIME',))
            time.units = 'seconds since 1970-01-01 00:00:00.0'
            time[::] = time_array[::]

            mlat = fnc.createVariable('MLAT', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            mlat[::] = mlat_arr[::]
            mlon = fnc.createVariable('MLON', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            mlon[::] = mlon_arr[::]
            mlt = fnc.createVariable('MLT', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            mlt[::] = mlt_arr[::]
            EF_N = fnc.createVariable('E_N', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            EF_N[::] = EF_N_arr[::]
            EF_E = fnc.createVariable('E_E', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            EF_E[::] = EF_E_arr[::]
            v_N = fnc.createVariable('v_i_N', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            v_N[::] = v_N_arr[::]
            v_E = fnc.createVariable('v_i_E', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            v_E[::] = v_E_arr[::]
            phi = fnc.createVariable('phi', np.float32, ('UNIX_TIME', 'MLON', 'MLAT'))
            phi[::] = phi_arr[::]

            print('From {} to {}.'.format(
                datetime.datetime.utcfromtimestamp(time_array[0]),
                datetime.datetime.utcfromtimestamp(time_array[-1]))
            )
            mylog.StreamLogger.info(
                "The requested SuperDARN map potential data has been saved in the file {}.".format(fp))
            fnc.close()

            self.done = True


if __name__ == "__main__":
    dt_fr1 = datetime.datetime(2016, 3, 15)
    dt_to1 = datetime.datetime(2016, 3, 15)
    Downloader(dt_fr1, dt_to1)



