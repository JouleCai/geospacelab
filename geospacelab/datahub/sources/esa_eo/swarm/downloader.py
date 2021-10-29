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
import zipfile
import ftplib
from contextlib import closing

from geospacelab.datahub.dataset_base import DownloaderModel
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.datahub.sources.wdc as wdc
from geospacelab import preferences as prf


class Downloader(DownloaderModel):
    """
    Base downloader for downloading the SWARM data files from ftp://swarm-diss.eo.esa.int/

    :param ftp_host: the FTP host address
    :type ftp_host: str
    :param ftp_port: the FTP port [21].
    :type ftp_port: int
    :param ftp_data_dir: the directory in the FTP that stores the data.
    :type ftp_data_dir: str
    """

    def __init__(self,
                 dt_fr, dt_to,
                 data_file_root_dir=None, ftp_data_dir=None, force=True, direct_download=True, file_version=None,
                 **kwargs):
        self.ftp_host = "swarm-diss.eo.esa_eo.int"
        self.ftp_port = 21
        if ftp_data_dir is None:
            raise ValueError

        self.ftp_data_dir = ftp_data_dir
        if file_version is None:
            file_version = ''
        self.file_version = file_version

        super(Downloader, self).__init__(
            dt_fr, dt_to, data_file_root_dir=data_file_root_dir, force=force, direct_download=direct_download, **kwargs
        )

    def download(self, **kwargs):
        done = False
        with closing(ftplib.FTP()) as ftp:
            try:
                ftp.connect(self.ftp_host, self.ftp_port, 30)  # 30 timeout
                ftp.login()
                ftp.cwd(self.ftp_data_dir)
                file_list = ftp.nlst()

                file_names = self.search_files(file_list)
                file_dir = pathlib.Path(str(self.data_file_root_dir) + self.ftp_data_dir)
                for file_name in file_names:
                    file_path = file_dir / file_name

                    if file_path.is_file():
                        mylog.simpleinfo.info(
                            "The file {} exists in the directory {}.".format(
                                file_path.name, file_path.parent.resolve()
                            )
                        )
                        if not self.force:
                            done = True
                            continue
                    else:
                        file_path.parent.resolve().mkdir(parents=True, exist_ok=True)
                    with open(file_path, 'w+b') as f:
                        done = False
                        res = ftp.retrbinary('RETR ' + file_name, f.write)
                        if not res.startswith('226 Transfer complete'):
                            print('Downloaded of file {0} is not compile.'.format(file_name))
                            pathlib.Path.unlink(file_path)
                            done = False
                            return done
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(file_path.parent.resolve())
                            file_path.unlink()

                    done = True
            except:
                print('Error during download from FTP')
                done = False
        return done

    def search_files(self, file_list=None, file_name_patterns=None):
        def extract_timeline(file_list):
            nf = len(file_list)
            dt_ranges = np.empty((2, nf), dtype=datetime.datetime)
            versions = np.empty((nf, ), dtype=str)
            for ind, fn in enumerate(file_list):
                dt_regex = re.compile(r'(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{4})')
                rm = dt_regex.findall(fn)
                if not list(rm):
                    dt_ranges[:, ind] = np.nan
                    continue
                dt_ranges[0, ind] = datetime.datetime.strptime(rm[0], '%Y%m%dT%H%M%S')
                dt_ranges[1, ind] = datetime.datetime.strptime(rm[1], '%Y%m%dT%H%M%S')
                versions[ind] = rm[2]

            return dt_ranges[0], dt_ranges[1], versions

        if file_name_patterns is None:
            file_name_patterns = []

        if list(file_name_patterns):
            search_pattern = '*' + '*'.join(file_name_patterns) + '*'
            fn_regex = re.compile(search_pattern)
            file_list = list(filter(fn_regex.match, file_list))

        start_dts, stop_dts, versions = extract_timeline(file_list)
        ind_dt = np.where(
            (self.dt_fr > start_dts & self.dt_fr < start_dts) |
            (self.dt_to > start_dts & self.dt_to < stop_dts)
        )[0]
        file_list = file_list[ind_dt]
        versions = versions[ind_dt]

        if not str(self.file_version):
            self.file_version = versions(np.argmax(versions.astype(np.int32)))
        ind_v = np.where(versions == self.file_version)[0]

        file_list = file_list[ind_v]
        return file_list, versions





