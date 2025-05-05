# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import copy
import datetime
import numpy as np
import requests
import bs4
import pathlib
import re
import zipfile
import ftplib
from contextlib import closing

from geospacelab.datahub.__dataset_base__ import DownloaderBase
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.datahub.sources.esa_eo.swarm as swarm


class Downloader(DownloaderBase):
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
                 sat_id=None,
                 data_file_root_dir=None, ftp_data_dir=None, force=True, direct_download=True, file_version=None,
                 username=swarm.default_username,
                 password=swarm.eo_password,
                 file_extension = '.cdf',
                 **kwargs):
        self.ftp_host = "swarm-diss.eo.esa.int"
        self.ftp_port = 21
        self.sat_id = sat_id.upper()
        self.username = username
        self.__password__ = password
        if ftp_data_dir is None:
            raise ValueError

        self.ftp_data_dir = ftp_data_dir
        if file_version is None:
            file_version = 'latest'
        self.file_version = file_version
        self.file_extension = file_extension

        super(Downloader, self).__init__(
            dt_fr, dt_to, data_file_root_dir=data_file_root_dir, force=force, direct_download=direct_download, **kwargs
        )

    def download(self, **kwargs):
        done = False
        diff_month = dttool.get_diff_months(self.dt_fr, self.dt_to)
        default_file_name_patterns = kwargs['file_name_patterns']
        for nm in range(diff_month+1):
            this_month = dttool.get_next_n_months(self.dt_fr, nm)
            file_name_patterns = copy.deepcopy(default_file_name_patterns)
            file_name_patterns.append(this_month.strftime("%Y%m"))
            ftp = ftplib.FTP_TLS()
            ftp.connect(self.ftp_host, self.ftp_port, 30)
            try:
                ftp.login(user=self.username, passwd=self.__password__)
                ftp.cwd(self.ftp_data_dir)
                file_list = ftp.nlst()
               
                file_names, versions = self.search_files(file_list=file_list, file_name_patterns=file_name_patterns)
                if file_names is None:
                    raise FileNotFoundError
                file_dir_root = self.data_file_root_dir
                for ind_f, file_name in enumerate(file_names):
                    dt_regex = re.compile(r'(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{4})')
                    rm = dt_regex.findall(file_name)
                    this_day = datetime.datetime.strptime(rm[0][0], '%Y%m%dT%H%M%S')
                    file_dir = file_dir_root / rm[0][2] / 'Sat_{}'.format(self.sat_id) / this_day.strftime("%Y")
                    file_dir.mkdir(parents=True, exist_ok=True)
                    file_path = file_dir / file_name
                    local_files = file_path.parent.resolve().glob(file_path.stem.split('.')[0] + '*' + self.file_extension)
                    # file_path_cdf = file_path.parent.resolve() / ((file_path.stem.split('.')[0] + self.file_extension))
                    # if file_path_cdf.is_file():
                    if list(local_files):
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
                    mylog.simpleinfo.info(
                        f"Downloading the file {file_name} from the FTP ..."
                    )
                    try:
                        with open(file_path, 'w+b') as f:
                            done = False
                            res = ftp.retrbinary('RETR ' + file_name, f.write)
                            print(res)
                            if not res.startswith('226'):
                                mylog.StreamLogger.warning('The file {0} downloaded is not compile.'.format(file_name))
                                pathlib.Path.unlink(file_path)
                                done = False
                                ftp.quit()
                                return done
                            mylog.simpleinfo.info("Done.")
                    except Exception as e:
                        print(e)
                        pathlib.Path.unlink(file_path)
                        mylog.StreamLogger.warning('The file {0} downloaded is not compile.'.format(file_name))
                        ftp.quit()
                        return False
                    mylog.simpleinfo.info("Uncompressing the file ...")
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(file_path.parent.resolve())
                        file_path.unlink()
                    mylog.simpleinfo.info("Done. The zip file has been removed.")

                done = True
            except Exception as e:
                print('Error during download from FTP')
                print(e)
                done = False
            ftp.quit()
        return done

    def search_files(self, file_list=None, file_name_patterns=None):
        def extract_timeline(files):
            nf = len(files)
            dt_ranges = np.empty((nf, 2), dtype=datetime.datetime)
            # dts = np.empty((nf, ), dtype=datetime.datetime)
            vers = np.empty((nf, ), dtype=object)
            for ind, fn in enumerate(files):
                dt_regex = re.compile(r'(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{4})')
                rm = dt_regex.findall(fn)
                if not list(rm):
                    dt_ranges[ind, :] = [datetime.datetime(1900, 1, 1), datetime.datetime(1900, 1, 1)]
                    continue
                dt_ranges[ind, 0] = datetime.datetime.strptime(rm[0][0], '%Y%m%dT%H%M%S')
                dt_ranges[ind, 1] = datetime.datetime.strptime(rm[0][1], '%Y%m%dT%H%M%S')
                vers[ind] = rm[0][2]

            return dt_ranges[:, 0], dt_ranges[:, 1], vers

        if file_name_patterns is None:
            file_name_patterns = []

        if list(file_name_patterns):
            search_pattern = '.*' + '.*'.join(file_name_patterns) + '.*'
            fn_regex = re.compile(search_pattern)
            file_list = list(filter(fn_regex.match, file_list))

        start_dts, stop_dts, versions = extract_timeline(file_list)

        ind_dt = np.where((self.dt_fr <= stop_dts) & (self.dt_to >= start_dts))[0]
        if not list(ind_dt):
            mylog.StreamLogger.info("No matching files found on the ftp")
            return None, None
        file_list = [file_list[ii] for ii in ind_dt]
        versions = [versions[ii] for ii in ind_dt]

        if self.file_version == 'latest':
            self.file_version = max(versions)
        ind_v = np.where(np.array(versions) == self.file_version)[0]

        file_list = [file_list[ii] for ii in ind_v]
        versions = [versions[ii] for ii in ind_v]
        return file_list, versions





