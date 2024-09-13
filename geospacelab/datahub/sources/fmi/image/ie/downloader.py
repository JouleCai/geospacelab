# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLAB"
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
from geospacelab.config import prf


class Downloader(object):

    def __init__(self, dt_fr,  dt_to, data_file_root_dir=None, force_download=False):

        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.force_download = force_download

        self.done = False
        if data_file_root_dir is None:
            self.data_file_root_dir = prf.datahub_data_root_dir / 'FMI' / 'IMAGE' / 'IE'
        else:
            self.data_file_root_dir = pathlib.Path(data_file_root_dir)

        self.url_base = "https://space.fmi.fi/image/www/data_download.php"

        self.download()

    def download(self):
        diff_days = dttool.get_diff_days(self.dt_fr, self.dt_to)
        dt0 = dttool.get_start_of_the_day(self.dt_fr)
        
        for i in range(diff_days + 1):
            starttime = dt0 + datetime.timedelta(days=i)
             
            params = {
                'starttime': starttime.strftime('%Y%m%d'),
                'length': '1440',
                'format': 'ie-index'
            }
            
            file_name = 'FMI_IMAGE_IE_' + starttime.strftime('%Y%m%d') + '.dat'
            file_dir = self.data_file_root_dir / '{:4d}'.format(starttime.year) 
            file_dir.mkdir(parents=True, exist_ok=True)
            file_path = file_dir / file_name
            
            if file_path.exists() & (not self.force_download):
                mylog.simpleinfo.info(f"The file has existed: {file_path}.")
                self.done = True
                continue
            mylog.simpleinfo.info(f'Downloading the data on {starttime.strftime("%Y-%m-%d")} ...') 
            r = requests.get(self.url_base, params=params)
            if "No data" in r.text:
                mylog.StreamLogger.warning(f"No data found on {starttime.strftime('%Y-%m-%d')}")
                self.done = (self.done or False)
                continue
            with open(file_path.with_suffix('.dat'), 'w') as f:
                f.write(r.text) 
            self.done = True
            mylog.simpleinfo.info('Done.')
        
        return


if __name__ == '__main__':
    dt1 = datetime.datetime(2009, 1, 1)
    dt2 = datetime.datetime(2009, 12, 31)
    
    Downloader(dt1, dt2)




# form_dst = {'SCent': 20, 'STens': 1, 'SYear': 1, 'SMonth': '01', 'ECent': 20, 'ETens': 1, 'EYear': 1, 'EMonth': 12, "Image Type": "GIF", "COLOR": "COLOR", "AE Sensitivity": "100", "Dst Sensitivity": "20", "Output": 'DST', "Out format": "IAGA2002", "Email": "lei.cai@oulu.fi"}