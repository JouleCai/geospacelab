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
from geospacelab.config import prf

from geospacelab.datahub.sources.cdaweb.downloader import DownloaderFromHTTPBase as DownloaderBase


class TUDownloader(DownloaderBase):
    
    root_dir_remote = 'data/data'
    
    def __init__(
        self, 
        dt_fr, dt_to, 
        mission='SWARM',
        sat_id=None,
        product='DNS-POD',
        version='v02',
        root_dir_local=None, 
        direct_download=True,
        force_download=False, dry_run=False, 
        **kwargs):
        
        base_url = 'https://thermosphere.tudelft.nl'
        if root_dir_local is None:
            root_dir_local = prf.datahub_data_root_dir / 'TUD'
            
        self.mission = mission
        self.sat_id = sat_id
        self.product = product
        self.version = version
        
        self._validate_mission_name()
        self._validate_sat_id()
        self._validate_product()
        self._validate_version()
        super().__init__(
            dt_fr, dt_to, 
            base_url=base_url, 
            root_dir_local=root_dir_local, 
            direct_download=direct_download, 
            force_download=force_download, 
            dry_run=dry_run, **kwargs)
    
    def search_from_http(self, subdirs = None, file_name_patterns = None, **kwargs):
        
        diff_months = dttool.get_diff_months(self.dt_fr, self.dt_to)
        file_paths_remote = []
        for nm in range(diff_months + 1):
            this_month = dttool.get_next_n_months(self.dt_fr, nm)
            
            subdirs = [version_dict[self.version[:3]]]
            subdirs.append(self.mission + '_data')

            file_name_patterns = [
                self.mission[0] + self.sat_id,
                self.product.replace('-', '_'),
                this_month.strftime('%Y'),
                this_month.strftime('%m'),
                self.version
            ]
            paths = super().search_from_http(subdirs, file_name_patterns, **kwargs)
            if len(paths) > 1:
                mylog.StreamLogger.warning(
                    f"Multiple versions found for {this_month.strftime('%Y-%m')}. All versions will be downloaded.")
            file_paths_remote.extend(paths)
        return file_paths_remote
    
    def save_files_from_http(self, file_paths_local=None, root_dir_remote=None):
        
        file_paths_local = []
        for fp_remote in self.file_paths_remote:
            fn_remote = fp_remote.split('/')[-1]
            
            rc = re.compile(r'_(v[\w]+)')
            rm = rc.search(fn_remote)
            if rm is None:
                raise ValueError(f"Cannot extract version info from file name {fn_remote}.")
            version_in_fn = rm.groups()[0]
            
            replacement = '/'.join([s.strip('/') for s in [
                self.base_url, 
                self.root_dir_remote,
                version_dict[version_in_fn[:3]], 
                self.mission + '_data'
            ]]) + '/'

            fp_local = fp_remote.replace(replacement, '')
            file_paths_local.append(
                self.root_dir_local / self.mission.upper() /
                self.product.upper() / version_in_fn / fp_local)
        super().save_files_from_http(file_paths_local, root_dir_remote)  
        
        for i, (done, file_path) in enumerate(zip(self.done, self.file_paths_local)):
            if not done:
                continue
            else:
                if not file_path.exists():
                    continue
                mylog.simpleinfo.info(f"Uncompressing the file: {file_path} ...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(file_path.parent.resolve())
                    file_path.unlink()
                    mylog.simpleinfo.info("Done. The zip file has been removed.")
                    self.file_paths_local[i] = file_path.with_suffix('.txt') # Update to unzipped file path
        return 
    
    def _to_download(self, file_path, with_suffix='.txt'):
        return super()._to_download(file_path, with_suffix=with_suffix)
    
    def _validate_mission_name(self):
        if self.mission.lower() == 'swarm':
            self.mission = 'Swarm'
        elif self.mission.lower() in ['champ', 'grace', 'goce', 'grace-fo']:
            self.mission = self.mission.upper()
        else:
            raise NotImplementedError(f"Mission {self.mission} is not supported.")
        
    def _validate_product(self):
        if self.mission == 'Swarm' and self.sat_id in ['A', 'B', 'C']:
            valid_products = ['DNS-POD', 'DNS-ACC']
            assert self.product in valid_products, f"Invalid product {self.product}. Valid products are {valid_products}."
        elif self.mission in ['CHAMP', 'GRACE', 'GOCE', 'GRACE-FO']:
            valid_products = ['DNS-ACC', 'WND-ACC']
            assert self.product in valid_products, f"Invalid product {self.product}. Valid products are {valid_products}."
        else:
            raise NotImplementedError(f"Mission {self.mission} with sat_id {self.sat_id} is not supported.")
        return
            
        
    def _validate_sat_id(self):
        
        if self.mission.upper() == 'SWARM':
            valid_sat_ids = ['A', 'B', 'C']
            assert self.sat_id in valid_sat_ids, f"Invalid sat_id {self.sat_id}. Valid sat_ids are {valid_sat_ids}."
        elif self.mission.upper() == 'CHAMP':
            valid_sat_ids = ['']
            assert self.sat_id in valid_sat_ids, f"Invalid sat_id {self.sat_id}. Valid sat_ids are {valid_sat_ids}."
        elif self.mission.upper() == 'GRACE':
            valid_sat_ids = ['A', 'B']
            assert self.sat_id in valid_sat_ids, f"Invalid sat_id {self.sat_id}. Valid sat_ids are {valid_sat_ids}."
        elif self.mission.upper() == 'GOCE':
            valid_sat_ids = ['']
            assert self.sat_id in valid_sat_ids, f"Invalid sat_id {self.sat_id}. Valid sat_ids are {valid_sat_ids}."
        elif self.mission.upper() == 'GRACE-FO':
            valid_sat_ids = ['FO1', '1', 'C']
            assert self.sat_id in valid_sat_ids, f"Invalid sat_id {self.sat_id}. Valid sat_ids are {valid_sat_ids}."
            self.sat_id = 'C'  # Standardize to 'C' for GRACE-FO
        else:
            raise NotImplementedError(f"Mission {self.mission} is not supported.")
        return

    def _validate_version(self):
        if self.mission.upper() in ['SWARM', 'CHAMP', 'GOCE', 'GRACE']:
            valid_versions = ['v01', 'v02']
            assert self.version in valid_versions, f"Invalid version {self.version}. Valid versions are {valid_versions}."
        elif self.mission.upper() == 'GRACE-FO':
            valid_versions = ['v02']
            assert self.version in valid_versions, f"Invalid version {self.version}. Valid versions are {valid_versions}."
        else:
            raise NotImplementedError(f"Mission {self.mission} is not supported.")
        return

version_dict = {
    'v01': 'version_01',
    'v02': 'version_02',
}


# class Downloader(DownloaderBase):
#     """
#     Base downloader for downloading the SWARM data files from ftp://swarm-diss.eo.esa.int/

#     :param ftp_host: the FTP host address
#     :type ftp_host: str
#     :param ftp_port: the FTP port [21].
#     :type ftp_port: int
#     :param ftp_data_dir: the directory in the FTP that stores the data.
#     :type ftp_data_dir: str
#     """

#     def __init__(self,
#                  dt_fr, dt_to,
#                  data_file_root_dir=None, ftp_data_dir=None, force=True, direct_download=True, file_name_patterns=[],
#                  **kwargs):
#         self.ftp_host = "thermosphere.tudelft.nl"
#         self.ftp_port = 21
#         if ftp_data_dir is None:
#             raise ValueError

#         self.ftp_data_dir = ftp_data_dir

#         self.file_name_patterns = file_name_patterns

#         super(Downloader, self).__init__(
#             dt_fr, dt_to, data_file_root_dir=data_file_root_dir, force=force, direct_download=direct_download, **kwargs
#         )


#     def download_from_http(self, **kwargs):
        
#         return
    
    
#     def download_from_ftp(self, **kwargs):
#         """[DEPRECATED] Download data files from FTP server.

#         Returns:
#             _type_: _description_
#         """
#         done = False
#         try:
#             ftp = ftplib.FTP()
#             ftp.connect(self.ftp_host, self.ftp_port, 30)  # 30 timeout
#             ftp.login()
#             ftp.cwd(self.ftp_data_dir)
#             file_list = ftp.nlst()

#             file_names = self.search_files(file_list=file_list, file_name_patterns=self.file_name_patterns)
#             file_dir_root = self.data_file_root_dir
#             for ind_f, file_name in enumerate(file_names):

#                 file_path = file_dir_root / file_name

#                 if file_path.is_file():
#                     mylog.simpleinfo.info(
#                         "The file {} exists in the directory {}.".format(
#                             file_path.name, file_path.parent.resolve()
#                         )
#                     )
#                     if not self.force:
#                         done = True
#                         continue
#                 else:
#                     file_path.parent.resolve().mkdir(parents=True, exist_ok=True)
#                 mylog.simpleinfo.info(
#                     f"Downloading the file {file_name} from the FTP ..."
#                 )
#                 try:
#                     with open(file_path, 'w+b') as f:
#                         done = False
#                         res = ftp.retrbinary('RETR ' + file_name, f.write)
#                         print(res)
#                         if not res.startswith('226'):
#                             mylog.StreamLogger.warning('Downloaded of file {0} is not compile.'.format(file_name))
#                             pathlib.Path.unlink(file_path)
#                             done = False
#                             return done
#                         mylog.simpleinfo.info("Done.")
#                 except:
#                     pathlib.Path.unlink(file_path)
#                     mylog.StreamLogger.warning('Downloaded of file {0} is not compile.'.format(file_name))
#                     return False
#                 mylog.simpleinfo.info("Uncompressing the file ...")
#                 with zipfile.ZipFile(file_path, 'r') as zip_ref:
#                     zip_ref.extractall(file_path.parent.resolve())
#                     file_path.unlink()
#                 mylog.simpleinfo.info("Done. The zip file has been removed.")

#             done = True
#             ftp.quit()
#         except Exception as err:
#             print(err)
#             print('Error during download from FTP')
#             done = False
#         return done

#     def search_files(self, file_list=None, file_name_patterns=None):
#         def extract_timeline(files):
            
#             nf = len(files)
#             dt_list = []
#             for ind, fn in enumerate(files):
#                 dt_regex = re.compile(r'_(\d{4}_\d{2})_')
#                 rm = dt_regex.findall(fn)
#                 if not list(rm):
#                     continue
#                 dt_list.append(datetime.datetime.strptime(rm[0] + '_01', '%Y_%m_%d'))
#             dt_list = np.array(dt_list, dtype=datetime.datetime)

#             return dt_list

#         if file_name_patterns is None:
#             file_name_patterns = []

#         if list(file_name_patterns):
#             search_pattern = '.*' + '.*'.join(file_name_patterns) + '.*'
#             fn_regex = re.compile(search_pattern)
#             file_list = list(filter(fn_regex.match, file_list))

#         dt_list = extract_timeline(file_list)

#         dt_fr = datetime.datetime(self.dt_fr.year, self.dt_fr.month, 1)
#         dt_to = datetime.datetime(self.dt_to.year, self.dt_to.month, 1)
#         ind_dt = np.where((dt_list >= dt_fr) & (dt_list <= dt_to))[0]
#         if not list(ind_dt):
#             raise FileExistsError
#         file_list = [file_list[ii] for ii in ind_dt]

#         return file_list





