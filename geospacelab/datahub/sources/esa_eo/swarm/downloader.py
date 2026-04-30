# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import pickle
from typing import Union, Tuple, List
import copy
import datetime
import numpy as np
import requests
import bs4
import pathlib
import re
import zipfile
import ftplib
import pandas as pd

from geospacelab.datahub.__dataset_base__ import DownloaderBase
from geospacelab.datahub.sources.cdaweb.downloader import DownloaderFromFTPBase
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.datahub.sources.esa_eo.swarm as swarm



FILE_RECORD_REMOTE_MODEL = {
    'id': np.empty((0, ), dtype=int),
    'file_path': np.empty((0, ), dtype=object),
    'file_name': np.empty((0, ), dtype=object),
    'product_version': np.empty((0, ), dtype=object),
    'product_name': np.empty((0, ), dtype=object),
    'product_level': np.empty((0, ), dtype=object),
    'mission': np.empty((0, ), dtype=object),
    'sat_id': np.empty((0, ), dtype=object),
    'datetime_fr': np.empty((0, ), dtype=object),
    'datetime_to': np.empty((0, ), dtype=object),
}


class DownloaderSwarm(DownloaderFromFTPBase):
    _default_sub_dirs_remote = None
    
    def __init__(
        self,
        dt_fr=None, dt_to=None,
        mission=None,
        sat_id=None,
        file_class=None,
        file_extension=None,
        product=None,
        product_pattherns=None,
        product_level=None,
        product_version=None,
        ftp_host=None, ftp_port=21,
        username=swarm.default_username, password=swarm.eo_password, 
        root_dir_local=None,
        root_dir_remote=None,
        sub_dirs_remote=None,
        sub_dirs_local=None,
        force_indexing = False,
        query_mode = 'off',
        direct_download=True, force_download=False, dry_run=False,
        **kwargs
        ):
        if ftp_host is None:
            ftp_host = "swarm-diss.eo.esa.int"
        if root_dir_remote is None:
            root_dir_remote = "/"
        if root_dir_local is None:
            root_dir_local = pathlib.Path.cwd() / 'data'
        
        self.mission = mission.upper() if mission is not None else None
        self.sat_id = sat_id.upper() if sat_id is not None else None
        self.file_class = file_class
        self.file_extension = file_extension
        self.product = product
        if product_pattherns is None:
            product_pattherns = product.split('_')
        self.product_pattherns = product_pattherns
        self.product_level = product_level
        self.product_version = product_version
        self._files_record_remote = None
        if sub_dirs_remote is None:
            sub_dirs_remote = self._default_sub_dirs_remote
        self.sub_dirs_remote = sub_dirs_remote
        if sub_dirs_local is None:
            sub_dirs_local = []
        self.sub_dirs_local = sub_dirs_local
        
        self.force_indexing = force_indexing
        self.query_mode = query_mode
        self._file_dir_indexing = None
        self._from_indexing_record = True
        self._indexing = False
        self._files_record_remote = copy.deepcopy(FILE_RECORD_REMOTE_MODEL)
        
        super().__init__(
            dt_fr, dt_to,
            ftp_host=ftp_host, ftp_port=ftp_port,
            username=username, password=password,
            root_dir_local=root_dir_local,
            root_dir_remote=root_dir_remote,
            direct_download=False, force_download=force_download, dry_run=dry_run,
            **kwargs
        )
        
        self._validate()
        
        if self.query_mode == 'on':
            drect_download = False    
        if direct_download:
            self.direct_download = True
            self.download(with_TLS=True, subdirs=self.sub_dirs_remote, file_name_patterns=self.product_pattherns, **kwargs)
    
    def _validate(self):
        if self._file_dir_indexing is None:
            self._file_dir_indexing = self.root_dir_local 
        file_name_indexing = f'{self.mission}_remote_indexing_{self.product}_Sat-{self.sat_id.upper()}.pickle'
        file_path_indexing = self._file_dir_indexing / file_name_indexing
        
        if not file_path_indexing.is_file():
            self._indexing = True
        else:
            if self.force_indexing:
                self._indexing = True
            else:
                mtime = datetime.datetime.fromtimestamp(file_path_indexing.stat().st_mtime, tz=datetime.timezone.utc)
                if (datetime.datetime.now(tz=datetime.timezone.utc) - mtime).total_seconds() > 86400:  # 1 day
                    self._indexing = True
        if self._indexing:
            mylog.StreamLogger.info(f"Indexing the files for the product {self.product} of the satellite {self.sat_id} ...")
            self._from_indexing_record = False
            self.download(
                with_TLS=True, subdirs=self.sub_dirs_remote, file_name_patterns=self.product_pattherns,
            )
            self._indexing_record = copy.deepcopy(self._files_record_remote)
            self._files_record_remote = copy.deepcopy(FILE_RECORD_REMOTE_MODEL)
            self._save_indexing_result(file_path_indexing)
            self._from_indexing_record = True
            self._indexing = False
        self._load_indexing_result(file_path_indexing)
    
    def _save_indexing_result(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self._indexing_record, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    def _load_indexing_result(self, file_path):
        with open(file_path, 'rb') as f:
            self._indexing_record = pickle.load(f)  
        
    def download(self, with_TLS=True, subdirs = None, file_name_patterns = None, **kwargs):
        
        return super().download(with_TLS, subdirs, file_name_patterns, **kwargs)
    
    def search_from_ftp(
        self, ftp, subdirs: Union[list, dict] = None, file_name_patterns = None, **kwargs
        ):
        if file_name_patterns is None:
            file_name_patterns = self.product_pattherns 
        if self._indexing:
            dt_fr = datetime.datetime(1900, 1, 1)
            dt_to = datetime.datetime(2100, 1, 1)
        else:
            dt_fr = self.dt_fr
            dt_to = self.dt_to
            
        if subdirs is None:
            subdirs = self._default_sub_dirs_remote
        if isinstance(subdirs, list):
            subdirs = {0: subdirs}
                
        if self._from_indexing_record:
            files_record = copy.deepcopy(self._indexing_record)
        else:
            file_paths_remote = []
            for k, subdirs_k in subdirs.items():
                pwd_ = ftp.pwd()
                subdirs_ = subdirs_k + ['Sat_{}'.format(self.sat_id.upper())]
                file_paths_remote_ = super().search_from_ftp(ftp, subdirs_, file_name_patterns, **kwargs)

                if not list(file_paths_remote_):
                    continue
                file_paths_remote.extend(file_paths_remote_)
                ftp.cwd(pwd_)
            files_record = self._parse_searched_files(file_paths_remote, **kwargs)
        if not self._indexing:
            files_record = self._filtering_files_by_time(files_record, dt_fr=dt_fr, dt_to=dt_to)
            files_record = self._filtering_files_by_version(files_record, version=self.product_version)
        self._files_record_remote = files_record
        file_paths_remote = files_record['file_path']
        return file_paths_remote
    
    def _filtering_files_by_time(self, files_record, dt_fr=None, dt_to=None):
        dts_fr = files_record['datetime_fr']
        dts_to = files_record['datetime_to']
        inds_t_invalid = np.where((dt_to < dts_fr) | (dt_fr > dts_to))[0]
        inds_t = np.array([i for i in range(len(dts_fr)) if i not in inds_t_invalid])
        if not list(inds_t):
            mylog.StreamLogger.warning("No matching files found on the ftp after filtering by time.")
            return copy.deepcopy(FILE_RECORD_REMOTE_MODEL)
        files_record_filtered = {key: files_record[key][inds_t] for key in files_record.keys()}
        return files_record_filtered
    
    def _filtering_files_by_version(self, files_record, version=None):
        if version is None:
            version = self.product_version
        files_with_versions = self._check_file_versions(files_record)

        inds_f = []
        for file_with_version in files_with_versions:
            versions = file_with_version['versions']
            if version == 'latest':
                version_latest = max(versions)
                version = version_latest
            ii_v = np.where(versions == version)[0]
            if not list(ii_v):
                mylog.StreamLogger.warning(f"No file with the version {version} found for the file name pattern {file_with_version['file_names'][0]}.")
                continue
            inds_f.append(file_with_version['indices'][ii_v][0])
        files_record_filtered = {key: files_record[key][inds_f] for key in files_record.keys()}
        return files_record_filtered

    def _check_file_versions(self, files_record, ):
        versions = files_record['product_version']
        file_names = files_record['file_name']
        file_names_ = [fn.replace(v, '') for fn, v in zip(file_names, versions)]
        fn_unique, inds_fn_unique, inds_fn_inverse = np.unique(file_names_, return_index=True, return_inverse=True) 
        file_paths = files_record['file_path']
        
        files_with_versions = []
        for fn_u, ifn in zip(fn_unique, inds_fn_unique):
            ii = np.where(inds_fn_inverse == ifn)[0]
            versions_c = versions[ii]
            
            files_with_versions.append(
                {
                    'file_names': file_names[ii],
                    'file_paths': file_paths[ii],
                    'versions': versions_c,
                    'indices': ii,
                }
            )
        return files_with_versions
    
    def save_files_from_ftp(self, ftp, file_paths_local=None, root_dir_remote=None, dry_run=None, **kwargs):
        if self._indexing:
            return
        dts_fr = self._files_record_remote['datetime_fr']
        dts_to = self._files_record_remote['datetime_to']
        file_names = self._files_record_remote['file_name']
        file_versions = self._files_record_remote['product_version']
        if file_paths_local is None:
            file_paths_local = []
        for dt_fr, dt_to, file_name, file_version in zip(dts_fr, dts_to, file_names, file_versions):
            if dt_fr.year == dt_to.year:
                fp_local = self.root_dir_local / file_version / 'Sat_{}'.format(self.sat_id.upper()) / dt_fr.strftime("%Y") / file_name
            else:
                fp_local = self.root_dir_local / file_version / 'Sat_{}'.format(self.sat_id.upper()) / file_name
            file_paths_local.append(fp_local)
        
        super().save_files_from_ftp(ftp, file_paths_local=file_paths_local, root_dir_remote=root_dir_remote, dry_run=dry_run, **kwargs)
        
        if not dry_run:
            self.uncompress_files()
        return
    
    def uncompress_files(self, remove_compressed_file=True):
        file_paths_local = self.file_paths_local
        
        for file_path_local in file_paths_local:
            if not file_path_local.is_file():
                continue
            if zipfile.is_zipfile(file_path_local):
                mylog.simpleinfo.info(f"Uncompressing the file {file_path_local} ...")
                with zipfile.ZipFile(file_path_local, 'r') as zip_ref:
                    zip_ref.extractall(file_path_local.parent.resolve())
                mylog.simpleinfo.info(f"Done. The zip file has been uncompressed.")
                if remove_compressed_file:
                    try:
                        file_path_local.unlink()
                        mylog.simpleinfo.info(f"The zip file has been removed.")
                    except Exception as e:
                        mylog.StreamLogger.warning(f"The zip file cannot be removed. Please check the file and remove it manually if it is not needed. Error message: {e}")
            else:
                mylog.StreamLogger.warning(f"The file {file_path_local} is not a zip file. It cannot be uncompressed."  )
    
    def _to_download(self, file_path, with_suffix=None):
        with_suffix = with_suffix if with_suffix is not None else self.file_extension
        return super()._to_download(file_path, with_suffix)
    
    def _parse_file_name(self, file_name, rc_pattern=None):
        if rc_pattern is None:
            rc_pattern = r'(\d{8}T\d{6})_(\d{8}T\d{6})_(\d+)[\._]+'
        rc = re.compile(rc_pattern)
        rm = rc.findall(file_name)
        if not list(rm):
            raise ValueError(f"Cannot parse the file name {file_name} with the regex pattern {rc_pattern}.")
        dt_fr = datetime.datetime.strptime(rm[0][0], '%Y%m%dT%H%M%S')
        dt_to = datetime.datetime.strptime(rm[0][1], '%Y%m%dT%H%M%S')
        version = rm[0][2]
        return dt_fr, dt_to, version

    def _parse_searched_files(self, file_paths_remote, rc_pattern=None, ):
        
        records = copy.deepcopy(FILE_RECORD_REMOTE_MODEL)
        for i, file_path in enumerate(file_paths_remote):
            file_name = pathlib.Path(file_path).name
            dt_fr, dt_to, version = self._parse_file_name(file_name, rc_pattern)
            
            records['id'] = np.concatenate((records['id'], [i]))
            records['file_path'] = np.concatenate((records['file_path'], [file_path]))
            records['file_name'] = np.concatenate((records['file_name'], [file_name]))
            records['product_version'] = np.concatenate((records['product_version'], [version]))
            records['product_name'] = np.concatenate((records['product_name'], [self.product]))
            records['product_level'] = np.concatenate((records['product_level'], [self.product_level if self.product_level is not None else '']))
            records['mission'] = np.concatenate((records['mission'], [self.mission if self.mission is not None else '']))
            records['sat_id'] = np.concatenate((records['sat_id'], [self.sat_id if self.sat_id is not None else '']))
            records['datetime_fr'] = np.concatenate((records['datetime_fr'], [dt_fr]))
            records['datetime_to'] = np.concatenate((records['datetime_to'], [dt_to]))
        
        return records


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
            raise ValueError("FTP data directory must be specified.")

        self.ftp_data_dir = ftp_data_dir
        if file_version is None:
            file_version = 'latest'
        self.file_version = file_version
        self.file_extension = file_extension

        super(Downloader, self).__init__(
            dt_fr, dt_to, data_file_root_dir=data_file_root_dir, force=force, direct_download=direct_download, **kwargs
        )

    def download(self, long_term=False,**kwargs):
        done = False
        diff_month = dttool.get_diff_months(self.dt_fr, self.dt_to)
        default_file_name_patterns = kwargs['file_name_patterns']
        if not long_term:
            months = [dttool.get_next_n_months(self.dt_fr, nm) for nm in range(diff_month+1)]
        else:           
            months = [self.dt_fr]
        for this_month in months:
            file_name_patterns = copy.deepcopy(default_file_name_patterns)
            if not long_term:
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





