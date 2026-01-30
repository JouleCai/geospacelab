# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import datetime
import pathlib

import ftplib
import requests
import bs4
import re
import tqdm

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.config import prf


class DownloaderBase(object):
    """

    Parameters
    ----------
    dt_fr : datetime.datetime
        The start datetime for downloading data.
    dt_to : datetime.datetime
        The end datetime for downloading data.
    root_dir_local : str or pathlib.Path
        The root directory in the local disk to store the downloaded data files.
    direct_download : bool
        Whether to start downloading once the Downloader object is created.
    force_download : bool
        Whether to force re-download even the data files are already in the local disk.
    dry_run : bool
        Whether to only print the downloading information without actual downloading.
    done : list of bool
        Whether the downloading is done. The length of the list is the number of files to be downloaded.
    file_paths_local : list of pathlib.Path
        The local file paths of the downloaded data files.
    file_paths_remote : list of str
        The remote file paths of the data files to be downloaded.
    
    """
    def __init__(
        self,
        dt_fr=None,
        dt_to=None,
        root_dir_local=None,
        direct_download=False,
        force_download=False,
        dry_run=False,
        download_from = None,
        **kwargs
    ):

        self.force_download = force_download
        self.dry_run = dry_run

        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.file_paths_source = []
        self.file_paths_local = []
        self.done = None
        self.file_paths_local = []
        self.file_paths_remote = []

        if root_dir_local is None:
            self.root_dir_local = prf.datahub_data_root_dir
        else:
            self.root_dir_local = root_dir_local

        if direct_download:
            self.download()
            
    def download(self, *args, **kwargs):
        pass
    
    def _to_download(self, file_path, with_suffix=None):
        
        to_download = True
        
        if with_suffix is not None:
            is_file = file_path.with_suffix(with_suffix).is_file()
        else:
            is_file = file_path.is_file()
        if is_file:
            if self.force_download:
                mylog.simpleinfo.info(
                    "The file {} exists in the directory {}: Forced redownloading the file ...".format(
                        file_path.name, file_path.parent.resolve()
                    )
                )
            else:
                mylog.simpleinfo.info(
                    "The file {} exists in the directory {}: Skipped downloading.".format(
                        file_path.name, file_path.parent.resolve()
                    )
                )
                to_download = False
        return to_download
    
    def _search_files(self, file_list, file_name_patterns):
        search_pattern = '.*' + '.*'.join(file_name_patterns) + '.*'
        fn_regex = re.compile(search_pattern)
        file_names = list(filter(fn_regex.match, file_list))
        return file_names
    


class DownloaderFromFTPBase(DownloaderBase):
    """
    Base downloader for downloading data files from FTP server.

    Parameters
    ----------
    ftp_host : str
        The FTP host address.
    ftp_port : int
        The FTP port [21].
    root_dir_remote : str
        The directory in the FTP that stores the data.
    """
    
    root_dir_remote = None

    def __init__(
        self,
        dt_fr, dt_to,
        ftp_host=None, ftp_port=21,
        username='anonymous', password='', 
        root_dir_local=None,
        root_dir_remote=None,
        direct_download=False, force_download=False, dry_run=False,
        **kwargs
        ):
        
        self.ftp_host = ftp_host
        self.ftp_port = ftp_port
        self.username = username
        self.password = password
        if root_dir_remote is not None:
            self.root_dir_remote = root_dir_remote
        else:
            self.root_dir_remote = '/'
        
        super(DownloaderFromFTPBase, self).__init__(
            dt_fr, dt_to,
            root_dir_local=root_dir_local, 
            direct_download=direct_download,
            force_download=force_download,
            dry_run=dry_run,
            **kwargs
        )
        
    def download(
        self, 
        with_TLS=False, 
        subdirs: None | list=None, 
        file_name_patterns: None | list=None,
        **kwargs
        ):
        
        timeout = kwargs.pop('timeout', 30) # seconds
        encoding = kwargs.pop('encoding', 'utf-8')
        
        if with_TLS:
            FTP_CLASS = ftplib.FTP_TLS
        else:
            FTP_CLASS = ftplib.FTP
            
        with FTP_CLASS(encoding=encoding) as ftp:
            ftp.connect(self.ftp_host, self.ftp_port, timeout)
            ftp.login(user=self.username, passwd=self.password)
            ftp.cwd(self.root_dir_remote)
            
            self.file_paths_remote = self.search_from_ftp(
                ftp, subdirs=subdirs, file_name_patterns=file_name_patterns
            )
            self.save_files_from_ftp(ftp)
            
    def search_from_ftp(self, ftp, subdirs: None | list=None, file_name_patterns: None | list=None):
        if subdirs is not None:
            for subdir in subdirs:
                ftp.cwd(subdir)
        try:
            files = ftp.nlst()
        except ftplib.error_perm as resp:
            if str(resp) == "550 No files found":
                mylog.StreamLogger.warning("No files in this directory.")
            else:
                raise

        file_names = self._search_files(
            file_list=files, file_name_patterns=file_name_patterns
        )        
        
        file_paths_remote = []
        for fn in file_names:
            file_paths_remote.append(ftp.pwd() + '/' + fn)
        return file_paths_remote
    
    def save_files_from_ftp(self, ftp, file_paths_local=None, root_dir_remote=None):
        if root_dir_remote is None:
            root_dir_remote = self.root_dir_remote
            
        if file_paths_local is None:
            self.file_paths_local = []
            for fp_remote in self.file_paths_remote:
                self.file_paths_local.append(
                    self.root_dir_local / fp_remote.replace(root_dir_remote, '')
                )
        self.done = [False] * len(self.file_paths_remote)
        
        if self.dry_run:
            for i, (fp_remote, fp_local) in enumerate(zip(self.file_paths_remote, self.file_paths_local)):
                mylog.simpleinfo.info(
                    f"Dry run: Downloading the file {fp_remote} to {fp_local} ..."
                )
            return
        
        for i, (fp_remote, fp_local) in enumerate(zip(self.file_paths_remote, self.file_paths_local)):
            to_download = self._to_download(fp_local)
            
            if not to_download:
                self.done[i] = True
                continue
            mylog.simpleinfo.info(
                f"Downloading the file: {fp_remote} ..."
            )
            
            file_dir_remote = fp_remote.rsplit('/', 1)[0]
            file_name_remote = fp_remote.split('/')[-1]
            
            file_dir_local = fp_local.parent.resolve()
            file_dir_local.mkdir(exist_ok=True, parents=True)
                
            with open(fp_local, 'w+b') as f:
                if file_dir_remote != ftp.pwd():
                    ftp.cwd(file_dir_remote)
                
                bufsize=1024
                total=ftp.size(fp_remote)
                pbar=tqdm(total=total)
                def bar(data):
                    f.write(data)
                    pbar.update(len(data))
                res = ftp.retrbinary('RETR '+ file_name_remote, bar, bufsize)
                pbar.close()
                # res = ftp.retrbinary('RETR ' + file_name_remote, f.write)
                # mylog.simpleinfo.info(res)
                if not res.startswith('226'):
                    mylog.StreamLogger.warning('The downloading is not compiled...: {}'.format(res))
                    fp_local.unlink(missing_ok=True)
                self.done[i] = True
                mylog.simpleinfo.info("Saved as {}.".format(fp_local))
            if not self.done[i]:
                mylog.StreamLogger.error("Error in downloading the file {}.".format(file_name_remote))
        return


class DownloaderFromHTTPBase(DownloaderBase):
    """
    Base downloader for downloading data files from HTTP server.

    Parameters
    ----------
    base_url : str
        The base URL of the HTTP server.
    root_dir_remote : str
        The directory in the HTTP server that stores the data.
    """
    
    root_dir_remote = None

    def __init__(
        self,
        dt_fr, dt_to,
        base_url=None,
        root_dir_local=None,
        root_dir_remote=None,
        direct_download=False, force_download=False, dry_run=False,
        **kwargs
        ):
        
        self.base_url = base_url
        
        if root_dir_remote is not None:    
            self.root_dir_remote = root_dir_remote
        
        super(DownloaderFromHTTPBase, self).__init__(
            dt_fr, dt_to,
            root_dir_local=root_dir_local,
            direct_download=direct_download,
            force_download=force_download,
            dry_run=dry_run,
            **kwargs
        )
        
    def download(
        self, 
        subdirs: None | list=None, 
        file_name_patterns: None | list=None,
        **kwargs
        ):
        
        self.file_paths_remote = self.search_from_http(
            subdirs=subdirs, file_name_patterns=file_name_patterns
        )
        self.save_files_from_http()
        
        
    def search_from_http(self, subdirs: None | list=None, file_name_patterns: None | list=None, **kwargs):
        url_patterns = [self.base_url]
        if str(self.root_dir_remote):
            url_patterns.append(self.root_dir_remote)
        if subdirs is not None:
            url_patterns.extend(subdirs)
        url = '/'.join(url_patterns)
        r = requests.get(url)
        soup = bs4.BeautifulSoup(r.text, 'html.parser')
        a_tags = soup.find_all('a', href=True)
        hrefs = [a_tag['href'] for a_tag in a_tags]
        
        hrefs = self._search_files(file_list=hrefs, file_name_patterns=file_name_patterns)

        file_paths_remote = []
        for href in hrefs:
            file_paths_remote.append(url + '/' + href)
        r.close()
        return file_paths_remote

    def save_files_from_http(self, file_paths_local=None, root_dir_remote=None):
        if root_dir_remote is None:
            root_dir_remote = self.root_dir_remote
        if str(root_dir_remote):
            root_url = self.base_url + '/' + root_dir_remote
        else:
            root_url = self.base_url
            
        if file_paths_local is None:
            self.file_paths_local = []
            for fp_remote in self.file_paths_remote:
                self.file_paths_local.append(
                    self.root_dir_local / fp_remote.replace(root_url, '')
                )
        else:
            self.file_paths_local = file_paths_local
        self.done = [False] * len(self.file_paths_remote)
        
        if self.dry_run:
            for i, (fp_remote, fp_local) in enumerate(zip(self.file_paths_remote, self.file_paths_local)):
                mylog.simpleinfo.info(
                    f"Dry run: Downloading the file {fp_remote} to {fp_local} ..."
                )
            return
        
        for i, (fp_remote, fp_local) in enumerate(zip(self.file_paths_remote, self.file_paths_local)):
            to_download = self._to_download(fp_local)
            
            if not to_download:
                self.done[i] = True
                continue
            
            mylog.simpleinfo.info(
                f"Downloading the file: {fp_remote} ..."
            )

            res = self._download_by_requests_get(fp_remote, fp_local)
            if res:
                mylog.simpleinfo.info(f"Saved as {fp_local}.")
                self.done[i] = True

        return

    @staticmethod
    def _download_by_requests_get(
        url, file_path_local, 
        params=None, stream=True, allow_redirects=True, 
        file_block_size=1024, 
        **kwargs
        ):
        try: 
            r = requests.get(url, params=params, stream=stream, allow_redirects=allow_redirects, **kwargs)
            if r.status_code != 200:
                return -2

            total_size_in_bytes = int(r.headers.get('content-length', 0))

            progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

            file_path_local.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path_local, 'wb') as file:
                for data in r.iter_content(file_block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                mylog.StreamLogger.error("Something wrong during the download!")
                file_path_local.unlink(missing_ok=True)
                return -1
            r.close()
        except Exception as e:
            mylog.StreamLogger.error(f"Exception during downloading the file from {url}: {e}")
            return -1
        return 1


class CDAWebHTTPDownloader(DownloaderFromHTTPBase):
    """
    Downloader for downloading data files from CDAWeb HTTP server.

    Parameters
    ----------
    base_url : str
        The base URL of the CDAWeb HTTP server.
    root_dir_remote : str
        The directory in the CDAWeb HTTP server that stores the data.
    """

    root_dir_remote = 'pub/data'
    
    def __init__(
        self,
        dt_fr, dt_to,
        root_dir_local=None,
        root_dir_remote=None,
        direct_download=False, force_download=False, dry_run=False,
        **kwargs
        ):
        
        base_url = 'https://cdaweb.gsfc.nasa.gov'
        
        if root_dir_local is None:
            root_dir_local = prf.datahub_data_root_dir / 'CDAWeb'
        
        super().__init__(
            dt_fr, dt_to,
            base_url=base_url,
            root_dir_local=root_dir_local,
            root_dir_remote=root_dir_remote,
            direct_download=direct_download,
            force_download=force_download,
            dry_run=dry_run,
            **kwargs
        )