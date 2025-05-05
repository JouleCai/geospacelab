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

import requests
import bs4
import re
import tqdm

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.config import prf


class Downloader(object):

    def __init__(
        self, 
        dt_fr, 
        dt_to,
        direct_download=True, 
        force_download=False,
        data_file_root_dir=None,
        dry_run=False,
        from_ftp=False,
        ):

        self.url_base = "https://cdaweb.gsfc.nasa.gov/pub/data/"
        self.force_download = force_download
        self.dry_run=dry_run
        self.from_ftp=from_ftp

        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.source_file_paths = []

        if data_file_root_dir is None:
            self.data_file_root_dir = prf.datahub_data_root_dir
        else:
            self.data_file_root_dir = data_file_root_dir

        if direct_download:
            self.download()

    def download(self,):
        if self.from_ftp:
            raise NotImplementedError
        else:
            self.download_from_http()
            
    def search_from_http(self, subdirs=None, file_name_patterns=None):
        url = self.url_base + '/'.join(subdirs)
        r = requests.get(url)

        soup = bs4.BeautifulSoup(r.text, 'html.parser')
        a_tags = soup.find_all('a', href=True)
        hrefs = [a_tag['href'] for a_tag in a_tags]

        search_pattern = '.*' + '.*'.join(file_name_patterns) + '.*'
        fn_regex = re.compile(search_pattern)
        hrefs = list(filter(fn_regex.match, hrefs))

        paths = []
        for href in hrefs:
            paths.append(url + '/' + href)
        return paths

    def download_from_http(self, ):

        source_file_paths = self.search_from_http()

        for url  in source_file_paths:
            if self.dry_run:
                print(f"Dry run: {url}.")
            else:
                self.save_file_from_http(url=url)

    def save_file_from_http(self, url, file_dir=None, file_name=None):
            if file_name is None:
                file_name = url.split('/')[-1]
            file_path = file_dir / file_name
            if file_path.is_file():
                mylog.simpleinfo.info(
                    "The file {} exists in the directory {}.".format(file_path.name, file_path.parent.resolve()))
                if not self.force_download:
                    self.done = True
                    return

            file_path.parent.resolve().mkdir(parents=True, exist_ok=True)
            mylog.simpleinfo.info(f'Downloading {file_name} ...')
            res = self._download_by_requests_get(url, data_file_path=file_path)
            if res:
                mylog.simpleinfo.info(f'Saved in {file_dir}')
                self.done = True
            else:
                mylog.StreamLogger.error(f"Error during downloading. Code: {res}.")

            return

    @staticmethod
    def _download_by_requests_get(url, data_file_path=None, params=None, stream=True, allow_redirects=True, **kwargs):
        r = requests.get(url, params=params, stream=stream, allow_redirects=allow_redirects, **kwargs)
        if r.status_code != 200:
            return -2

        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        with open(data_file_path, 'wb') as file:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            mylog.StreamLogger.error("Something wrong during the download!")
            data_file_path.unlink(missing_ok=True)
            return -1

        return 1

    def search_from_ftp(self, ftp, file_name_patterns=None):
        paths = []

        return paths

    def save_file(self, file_dir, file_name, r_source_file):
        file_path = file_dir / file_name

        if file_path.is_file():
            mylog.simpleinfo.info(
                "The file {} exists in the directory {}.".format(file_path.name, file_path.parent.resolve()))

        if self.force_download:
            file_path.parent.resolve().mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                mylog.simpleinfo.info(
                    "Downloading {} to the directory {} ...".format(file_path.name, file_path.parent.resolve())
                )
                f.write(r_source_file.content)
                mylog.simpleinfo.info("Done")

