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
import netCDF4
import cftime

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.datahub.sources.wdc as wdc
from geospacelab.config import prf


class Downloader(object):

    def __init__(self, dt_fr,  dt_to, data_file_root_dir=None, user_email=wdc.default_user_email):

        self.dt_fr = dt_fr
        self.dt_to = dt_to

        self.user_email = user_email

        self.done = False
        if data_file_root_dir is None:
            self.data_file_root_dir = prf.datahub_data_root_dir / 'WDC' / 'ASYSYM'
        else:
            self.data_file_root_dir = pathlib.Path(data_file_root_dir)

        self.url_base = "http://wdc.kugi.kyoto-u.ac.jp"

        self.download()

    def download(self):
        diff_months = dttool.get_diff_months(self.dt_fr, self.dt_to)
        dt0 = datetime.datetime(self.dt_fr.year, self.dt_fr.month, 1)
        r = requests.get(self.url_base + '/aeasy/')
        soup = bs4.BeautifulSoup(r.text, 'html.parser')
        form_tag = soup.find_all('form')
        r_method = form_tag[0].attrs['method']
        r_action_url = self.url_base + form_tag[0].attrs['action']
        for i in range(diff_months + 1):
            dt_fr = dttool.get_next_n_months(dt0, i)
            dt_to = dttool.get_next_n_months(dt0, i + 1) - datetime.timedelta(seconds=1)
            delta_seconds = (dt_to - dt_fr).total_seconds()

            file_name = 'WDC_ASYSYM_' + dt_fr.strftime('%Y%m') + '.nc'
            file_path = self.data_file_root_dir / '{:4d}'.format(dt_fr.year) / file_name
            if file_path.is_file():
                mylog.simpleinfo.info(
                    "The file {} exists in the directory {}.".format(file_path.name, file_path.parent.resolve()))
                self.done = True
                continue
            else:
                file_path.parent.resolve().mkdir(parents=True, exist_ok=True)
            form_asy = {
                'Tens': str(int(dt_fr.year/10)),
                'Year': str(int(dt_fr.year - np.floor(dt_fr.year/10)*10)),
                'Month': '{:02d}'.format(dt_fr.month),
                'Day_Tens': '0',
                'Days':     '1',
                'Hour':    '00',
                'min':     '00',
                'Dur_Day_Tens': '{:02d}'.format(int(np.floor(np.ceil(delta_seconds/86400.)/10))),
                'Dur_Day': str(int(np.ceil(delta_seconds/86400.) - np.floor(np.ceil(delta_seconds/86400.)/10)*10)),
                'Dur_Hour': '00',
                "Dur_Min": '00',
                "Image Type": "GIF",
                "COLOR": "COLOR",
                "AE Sensitivity": "100",
                "ASY/SYM Sensitivity": "100",
                "Output": 'ASY',
                "Out format": "IAGA2002",
                "Email": self.user_email,
            }

            if r_method.lower() == 'get':
                mylog.StreamLogger.info("Requesting data from WDC ...")
                r_file = requests.get(r_action_url, params=form_asy)

            if "No data for your request" in r_file.text or "DATE       TIME         DOY" not in r_file.text:
                mylog.StreamLogger.warning("No data for your request!")
                return

            with open(file_path.with_suffix('.dat'), 'w') as f:
                f.write(r_file.text)

            mylog.StreamLogger.info("Preparing to save the data in the netcdf format ...")
            self.save_to_netcdf(r_file.text, file_path)

    def save_to_netcdf(self, r_text, file_path):

        results = re.findall(
            r'^(\d+-\d+-\d+ \d+:\d+:\d+.\d+)\s*(\d+)\s*([+\-\d.]+)\s*([+\-\d.]+)\s*([+\-\d.]+)\s*([+\-\d.]+)',
            r_text,
            re.M
        )
        results = list(zip(*results))
        # time_array = np.array([(datetime.datetime.strptime(dtstr+'000', "%Y-%m-%d %H:%M:%S.%f")
        #                      - datetime.datetime(1970, 1, 1)) / datetime.timedelta(seconds=1)
        #                      for dtstr in results[0]])
        dts = [datetime.datetime.strptime(dtstr+'000', "%Y-%m-%d %H:%M:%S.%f") for dtstr in results[0]]
        time_array = np.array(cftime.date2num(dts, units='seconds since 1970-01-01 00:00:00.0'))
        print('From {} to {}.'.format(
            datetime.datetime.fromtimestamp(time_array[0], tz=datetime.timezone.utc),
            datetime.datetime.fromtimestamp(time_array[-1], tz=datetime.timezone.utc),)
        )
        asy_d_array = np.array(results[2])
        asy_d_array.astype(np.float32)
        asy_h_array = np.array(results[3])
        asy_h_array.astype(np.float32)
        sym_d_array = np.array(results[4])
        sym_d_array.astype(np.float32)
        sym_h_array = np.array(results[5])
        sym_h_array.astype(np.float32)

        num_rows = len(results[0])

        fnc = netCDF4.Dataset(file_path, 'w')
        fnc.createDimension('UNIX_TIME', num_rows)

        fnc.title = "WDC ASY/SYM indices"
        time = fnc.createVariable('UNIX_TIME', np.float64, ('UNIX_TIME',))
        time.units = 'seconds since 1970-01-01 00:00:00.0'
        asy_d = fnc.createVariable('ASY_D', np.float32, ('UNIX_TIME',))
        asy_h = fnc.createVariable('ASY_H', np.float32, ('UNIX_TIME',))
        sym_d = fnc.createVariable('SYM_D', np.float32, ('UNIX_TIME',))
        sym_h = fnc.createVariable('SYM_H', np.float32, ('UNIX_TIME',))

        time[::] = time_array[::]
        asy_d[::] = asy_d_array[::]
        asy_h[::] = asy_h_array[::]
        sym_d[::] = sym_d_array[::]
        sym_h[::] = sym_h_array[::]
        # for i, res in enumerate(results):
        #     dt = datetime.datetime.strptime(res[0]+'000', "%Y-%m-%d %H:%M:%S.%f")
        #     time[i] = (dt - datetime.datetime(1970, 1, 1)) / datetime.timedelta(seconds=1)
        #     asy_d[i] = float(res[2])
        #     asy_h[i] = float(res[3])
        #     sym_d[i] = float(res[4])
        #     sym_h[i] = float(res[5])

        fnc.close()
        mylog.StreamLogger.info("The requested data has been downloaded and saved in the file {}.".format(file_path))
        self.done = True


if __name__ == "__main__":
    dt_fr1 = datetime.datetime(2000, 1, 14)
    dt_to1 = datetime.datetime(2000, 6, 16)
    Downloader(dt_fr1, dt_to1, user_email="lei.cai@oulu.fi")