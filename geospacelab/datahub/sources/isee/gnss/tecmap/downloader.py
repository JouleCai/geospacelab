# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import datetime
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as dtparse
import numpy as np
import re
import requests
import bs4
import os
import pathlib

from geospacelab.config import prf
import geospacelab.datahub.sources.madrigal as madrigal
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool


def test():
    dt_fr = datetime.datetime(2021, 3, 10)
    dt_to = datetime.datetime(2021, 3, 10)
    download_obj = Downloader(dt_fr, dt_to)


data_type_dict = {
    'TEC-MAP': 'TEC binned',
    'TEC-LOS': 'Line of sight',
    'TEC-Sites': 'List of sites',
}


class Downloader(object):
    """Download the GNSS TEC data
    """

    def __init__(self, dt_fr, dt_to, data_file_root_dir=None, file_type='TEC-MAP', force_download=False):
        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.force_download = force_download
        if data_file_root_dir is None:
            self.data_file_root_dir: pathlib.Path = prf.datahub_data_root_dir / 'ISEE' / 'GNSS' / 'TEC'
        else:
            self.data_file_root_dir = data_file_root_dir
        self.done = False

        self.root_url = "https://stdb2.isee.nagoya-u.ac.jp/GPS/shinbori/AGRID2/nc/"
        self.download()

    def download(self):
        ndays = dttool.get_diff_days(self.dt_fr, self.dt_to)
        dt_0 = dttool.get_start_of_the_day(self.dt_fr)
        for nd in range(ndays + 1):
            thisday = dt_0 + datetime.timedelta(days=nd)
            doy = dttool.get_doy(thisday)
            for hh in range(24):
                this_time = thisday + datetime.timedelta(hours=hh)
                url = self.root_url + \
                    "{:04d}/{:03d}/".format(thisday.year, doy) + \
                    this_time.strftime("%Y%m%d%H") + '_atec.nc'

                file_dir = self.data_file_root_dir / str(thisday.year) / thisday.strftime("%Y%m%d")
                file_dir.mkdir(parents=True, exist_ok=True)
                file_name = this_time.strftime("%Y%m%d%H") + '_atec.nc'
                file_path = file_dir / file_name
                if file_path.is_file():
                    mylog.simpleinfo.info("The file {} has been downloaded.".format(file_path.name))
                    self.done = True
                    continue
                mylog.simpleinfo.info("Downloading  {} from the ISEE GNSS database ...".format(file_name))
                r = requests.get(url)
                with open(file_dir / file_name, 'wb') as f:
                    f.write(r.content)
                self.done = True


if __name__ == "__main__":
    test()
