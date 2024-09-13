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
import ftplib
from contextlib import closing

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.datahub.sources.wdc as wdc
from geospacelab.config import prf


class Downloader(object):

    def __init__(
            self, dt_fr,  dt_to,
            data_file_root_dir=None, force=False, ftp_sub_dir=None, ftp_filename_prefix=None,
            ftp_filename_ext='txt'
    ):

        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.force = force
        self.done = False
        if data_file_root_dir is None:
            self.data_file_root_dir = prf.datahub_data_root_dir / 'GFZ' / 'Indices'
        else:
            self.data_file_root_dir = pathlib.Path(data_file_root_dir)

        self.ftp_host = "ftp.gfz-potsdam.de"
        self.ftp_subdir = ftp_sub_dir
        self.ftp_filename_prefix = ftp_filename_prefix
        self.ftp_filename_ext = ftp_filename_ext

    def download(self):
        diff_years = self.dt_to.year - self.dt_fr.year

        for i in range(diff_years + 1):
            dt1 = datetime.datetime(self.dt_fr.year + i, 1, 1)

            ystr = dt1.strftime('%Y')

            with closing(ftplib.FTP()) as ftp:
                try:
                    ftp.connect(self.ftp_host, 21, 30)  # 30 timeout
                    ftp.login()
                    ftp.cwd('/pub/home/obs/' + self.ftp_subdir)

                    file_list = ftp.nlst()

                    file_name_patterns = list([self.ftp_filename_prefix+ystr])

                    file_name = self.search_files(file_list=file_list, file_name_patterns=file_name_patterns)

                    file_path = self.data_file_root_dir / file_name

                    if file_path.is_file() and not self.force:
                        mylog.simpleinfo.info(
                            "The file {} exists in the directory {}.".format(file_path.name, file_path.parent.resolve()))
                        self.done = True
                    else:
                        file_path.parent.resolve().mkdir(parents=True, exist_ok=True)
                        with open(file_path, 'w+b') as f:
                            res = ftp.retrbinary('RETR ' + file_name, f.write)

                            if not res.startswith('226 Transfer complete'):
                                print('Downloaded of file {0} is not compile.'.format(file_name))
                                pathlib.Path.unlink(file_path)
                                self.done = False
                                return None
                except:
                    print('Error during download from FTP')
                    self.done = False
                    return

            mylog.StreamLogger.info("Preparing to save the data in the netcdf format ...")
            self.save_to_netcdf(ystr, file_path)

    @staticmethod
    def search_files(file_list=None, file_name_patterns=None):

        if file_name_patterns is None:
            file_name_patterns = []

        # filter with file name patterns
        if list(file_name_patterns):
            search_pattern = '.*' + '.*'.join(file_name_patterns) + '.*'
            fn_regex = re.compile(search_pattern)
            file_list = list(filter(fn_regex.match, file_list))

        if not list(file_list):
            return None

        if len(file_list) > 1:
            raise ValueError

        return file_list[0]

    def save_to_netcdf(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    dt_fr1 = datetime.datetime(2020, 1, 1)
    dt_to1 = datetime.datetime(2020, 12, 16)
    Downloader(dt_fr1, dt_to1)



