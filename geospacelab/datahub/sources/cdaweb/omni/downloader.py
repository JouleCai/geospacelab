# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import datetime
import requests
import bs4
import re
import pathlib

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.config import prf

from geospacelab.datahub.sources.cdaweb.downloader import CDAWebHTTPDownloader as DownloaderBase


class Downloader(DownloaderBase):
    
    def __init__(
        self, 
        dt_fr,  dt_to, 
        time_res='1min', 
        product='OMNI2', 
        version='',
        root_dir_local=None,
        direct_download=True,
        force_download=False,
        dry_run=False,
        ):

        self.time_res = time_res
        self.product = product
        self.version = version
        
        if root_dir_local is None:
            root_dir_local = prf.datahub_data_root_dir / "CDAWeb" / 'OMNI' 
        else:
            root_dir_local = pathlib.Path(root_dir_local)
        root_dir_remote = '/'.join([
            self.root_dir_remote, 'omni', 'omni_cdaweb'
            ])
        super().__init__(
            dt_fr, dt_to,
            root_dir_local=root_dir_local,
            root_dir_remote=root_dir_remote,
            direct_download=direct_download,
            force_download=force_download,
            dry_run=dry_run,
        ) 
        
    def search_from_http(self, *args, **kwargs):
        dt_fr_1 = self.dt_fr
        dt_to_1 = self.dt_to
        diff_months = (dt_to_1.year - dt_fr_1.year) * 12 + dt_to_1.month - dt_fr_1.month
        file_paths_remote = []
        for nm in range(diff_months + 1):
            this_month = dttool.get_next_n_months(dt_fr_1, nm)
            
            subdirs = []
            
            if self.time_res in ['1min', '5min']:
                subdirs.append(f'{omni_product_dict[self.product]}_{self.time_res}')
            elif self.time_res == '1h':
                subdirs.append('hourly')
            subdirs.append('{:4d}'.format(this_month.year))
            
            if self.time_res in ['1min', '5min',]:
                file_name_patterns = [
                    'omni', omni_product_dict[self.product], self.time_res, this_month.strftime("%Y%m")
                    ]
            elif self.time_res == '1h':
                file_name_patterns = [
                    'omni2', 'mrg1hr', this_month.strftime("%Y")
                    ]
            if str(self.version):
                file_name_patterns.append(self.version)
                
            paths = super().search_from_http(subdirs=subdirs, file_name_patterns=file_name_patterns)
            
            if len(paths) > 1 and self.time_res != '1h':
                mylog.StreamLogger.error("Find multiple matched files!")
                print(paths)

            file_paths_remote.extend(paths)
        
        return list(set(file_paths_remote))
    
    def save_files_from_http(self, file_paths_local=None, root_dir_remote=None):
        
        if file_paths_local is None:
            file_paths_local = []
            for fp_remote in self.file_paths_remote:
                if self.time_res in ['1min', '5min']:
                    subdir = f'{self.product}_high_res_{self.time_res}'
                elif self.time_res == '1h':
                    subdir = 'OMNI2_low_res_1h'
            
                fp_local = fp_remote.replace(self.base_url + '/' + self.root_dir_remote + '/', '')
                pattern_replaced = fp_local.split('/')[0]
                fp_local = fp_local.replace(
                    pattern_replaced, subdir, 1
                )
                file_paths_local.append(self.root_dir_local / fp_local)
        return super().save_files_from_http(file_paths_local, root_dir_remote)
        

omni_product_dict = {
    'OMNI': 'hro',
    'OMNI2': 'hro2',
}


def test():
    dt_fr = datetime.datetime(2020, 3, 4)
    dt_to = datetime.datetime(2020, 5, 3)
    download_obj = Downloader(dt_fr, dt_to, force_download=True, time_res='1h')
    pass


if __name__ == "__main__":
    test()
