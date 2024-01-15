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
import ftplib
from contextlib import closing

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.datahub.sources.gfz.downloader as downloader
import geospacelab.datahub.sources.wdc as wdc


class Downloader(downloader.Downloader):

    def __init__(self, data_res=60, data_file_root_dir=None, force=False, download_method='html'):

        self.data_res = data_res

        ftp_sub_dir = 'Hpo/' + 'Hp' + str(self.data_res)
        ftp_filename_prefix = 'ap' + str(self.data_res) + '_'

        dt_fr = datetime.datetime.now()
        dt_to = datetime.datetime.now()
        super().__init__(
            dt_fr,  dt_to,
            data_file_root_dir=data_file_root_dir, force=force,
            ftp_sub_dir=ftp_sub_dir, ftp_filename_prefix=ftp_filename_prefix
        )
        self.data_file_root_dir = self.data_file_root_dir / 'Hpo' / f'Hp{self.data_res}'
        self.download_method = download_method
        self.download()

    def download(self):
        if self.download_method == 'html':
            self.download_from_web()
        elif self.download_method == 'ftp':
            super().download()

    def download_from_web(self):
        mylog.StreamLogger.info("Downloading the nowcast data from the GFZ server ...")
        r = requests.get(f'https://www-app3.gfz-potsdam.de/kp_index/Hp{self.data_res}_ap{self.data_res}_nowcast.txt')
        r.raise_for_status()
        file_path = self.data_file_root_dir / f'Hp{self.data_res}_ap{self.data_res}_nowcast.txt'
        with open(file_path, "w") as file:
            file.write(r.text)
        self.save_to_netcdf(file_path)

    def save_to_netcdf(self, file_path):
        with open(file_path, 'r') as f:
            text = f.read()

            results = re.findall(
                r'^(\d{4} \d{2} \d{2})\s*([-+]?\d*\.\d+|\d+)' +
                r'\s*([-+]?\d*\.\d+|\d+)\s*([-+]?\d*\.\d+|\d+)\s*([-+]?\d*\.\d+|\d+)' +
                r'\s*([-+]?\d*\.\d+|\d+)\s*([-+]?\d*\.\d+|\d+)\s*([-+]?\d*\.\d+|\d+)',
                text,
                re.M
            )

            results = list(zip(*results))

            dts = [
                datetime.datetime.strptime(dtstr, "%Y %m %d") + datetime.timedelta(hours=float(hs))
                for dtstr, hs in zip(results[0], results[2])
            ]
            time_array = np.array(cftime.date2num(dts, units='seconds since 1970-01-01 00:00:00.0'))

            hp_array = np.array(results[5]).astype(np.float32)
            hp_array = np.where(hp_array == -1, np.nan, hp_array)
            ap_array = np.array(results[6]).astype(np.float32)
            ap_array = np.where(ap_array == -1, np.nan, ap_array)

            flag_array = np.array(results[7]).astype(np.int32)

            num_rows = len(results[0])

            fp = file_path.parent.resolve() / (f"GFZ_Hpo_Hp{self.data_res}_nowcast" + '.nc')
            fp.parent.resolve().mkdir(parents=True, exist_ok=True)
            fnc = nc.Dataset(fp, 'w')
            fnc.createDimension('UNIX_TIME', num_rows)

            fnc.title = f"GFZ Hpo/Hp{self.data_res} index"
            time = fnc.createVariable('UNIX_TIME', np.float64, ('UNIX_TIME',))
            time.units = 'seconds since 1970-01-01 00:00:00.0'
            hp = fnc.createVariable('Hp', np.float32, ('UNIX_TIME',))
            ap = fnc.createVariable('ap', np.float32, ('UNIX_TIME',))

            flag = fnc.createVariable('FLAG', np.float32, ('UNIX_TIME',))
            time[::] = time_array[::]
            hp[::] = hp_array[::]
            ap[::] = ap_array[::]
            flag[::] = flag_array
            print('From {} to {}.'.format(
            datetime.datetime.utcfromtimestamp(time_array[0]),
            datetime.datetime.utcfromtimestamp(time_array[-1]))
            )
            mylog.StreamLogger.info(
                "The requested Hpo/Hp{} data has been downloaded and saved in the file {}.".format(self.data_res, fp))
            fnc.close()

            self.done = True


if __name__ == "__main__":
    dt_fr1 = datetime.datetime(2020, 1, 1)
    dt_to1 = datetime.datetime(2020, 12, 16)
    Downloader(dt_fr1, dt_to1)



