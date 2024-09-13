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

from geospacelab.datahub.__dataset_base__ import DownloaderBase
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.datahub.sources.wdc as wdc


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
                 data_file_root_dir=None, ftp_data_dir=None, force=True, direct_download=True, file_name_patterns=[],
                 **kwargs):
        self.ftp_host = "thermosphere.tudelft.nl"
        self.ftp_port = 21
        if ftp_data_dir is None:
            raise ValueError

        self.ftp_data_dir = ftp_data_dir

        self.file_name_patterns = file_name_patterns

        super(Downloader, self).__init__(
            dt_fr, dt_to, data_file_root_dir=data_file_root_dir, force=force, direct_download=direct_download, **kwargs
        )

    def download(self, **kwargs):
        done = False
        try:
            ftp = ftplib.FTP()
            ftp.connect(self.ftp_host, self.ftp_port, 30)  # 30 timeout
            ftp.login()
            ftp.cwd(self.ftp_data_dir)
            file_list = ftp.nlst()

            file_names = self.search_files(file_list=file_list, file_name_patterns=self.file_name_patterns)
            file_dir_root = self.data_file_root_dir
            for ind_f, file_name in enumerate(file_names):

                file_path = file_dir_root / file_name

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
                mylog.simpleinfo.info(
                    f"Downloading the file {file_name} from the FTP ..."
                )
                try:
                    with open(file_path, 'w+b') as f:
                        done = False
                        res = ftp.retrbinary('RETR ' + file_name, f.write)
                        print(res)
                        if not res.startswith('226'):
                            mylog.StreamLogger.warning('Downloaded of file {0} is not compile.'.format(file_name))
                            pathlib.Path.unlink(file_path)
                            done = False
                            return done
                        mylog.simpleinfo.info("Done.")
                except:
                    pathlib.Path.unlink(file_path)
                    mylog.StreamLogger.warning('Downloaded of file {0} is not compile.'.format(file_name))
                    return False
                mylog.simpleinfo.info("Uncompressing the file ...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(file_path.parent.resolve())
                    file_path.unlink()
                mylog.simpleinfo.info("Done. The zip file has been removed.")

            done = True
            ftp.quit()
        except Exception as err:
            print(err)
            print('Error during download from FTP')
            done = False
        return done

    def search_files(self, file_list=None, file_name_patterns=None):
        def extract_timeline(files):
            
            nf = len(files)
            dt_list = []
            for ind, fn in enumerate(files):
                dt_regex = re.compile(r'_(\d{4}_\d{2})_')
                rm = dt_regex.findall(fn)
                if not list(rm):
                    continue
                dt_list.append(datetime.datetime.strptime(rm[0] + '_01', '%Y_%m_%d'))
            dt_list = np.array(dt_list, dtype=datetime.datetime)

            return dt_list

        if file_name_patterns is None:
            file_name_patterns = []

        if list(file_name_patterns):
            search_pattern = '.*' + '.*'.join(file_name_patterns) + '.*'
            fn_regex = re.compile(search_pattern)
            file_list = list(filter(fn_regex.match, file_list))

        dt_list = extract_timeline(file_list)

        dt_fr = datetime.datetime(self.dt_fr.year, self.dt_fr.month, 1)
        dt_to = datetime.datetime(self.dt_to.year, self.dt_to.month, 1)
        ind_dt = np.where((dt_list >= dt_fr) & (dt_list <= dt_to))[0]
        if not list(ind_dt):
            raise FileExistsError
        file_list = [file_list[ii] for ii in ind_dt]

        return file_list





