# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

from turtle import rt

from geospacelab.config import prf

from geospacelab.datahub.sources.esa_eo import swarm
from geospacelab.datahub.sources.esa_eo.swarm.downloader import Downloader as DownloaderModel

from geospacelab.datahub.sources.esa_eo.swarm.downloader import DownloaderSwarm


class Downloader(DownloaderSwarm):
    
    def __init__(
        self,
        dt_fr, dt_to,
        sat_id=None,
        product='AEJ_LPS',
        product_version='latest',
        file_extension='.cdf',
        data_file_root_dir=None,
        force_indexing = False,
        query_mode='off',
        direct_download=True, force_download=False, dry_run=False,
    ):

        mission = 'SWARM'
        file_class = 'OPER'
        product_level = '2F'
        root_dir_remote = '/Level2daily'
        
        if data_file_root_dir is None:
            data_file_root_dir = prf.datahub_data_root_dir / "ESA" / "SWARM" / "Level2daily" / "AEJ_LPS"
        root_dir_local = data_file_root_dir
        
        self._default_sub_dirs_remote = {
            'latest': ['Latest_baselines', 'AEJ', 'LPS'],
            # 'Older_baselines': ['Older_baselines', 'AEJ', 'LPS'], # No older baselines for AEJ_LPS
            'Entire_mission': ['Entire_mission_data', 'AEJ', 'LPS']
        }
        
        
        super(Downloader, self).__init__(
            dt_fr=dt_fr, dt_to=dt_to,
            mission=mission,
            sat_id=sat_id,
            file_class=file_class,
            file_extension=file_extension,
            product=product,
            product_pattherns=None,
            product_level=product_level,
            product_version=product_version,
            root_dir_local=root_dir_local,
            root_dir_remote=root_dir_remote,
            sub_dirs_remote=None,
            sub_dirs_local=None,
            force_indexing = force_indexing,
            query_mode=query_mode,
            direct_download=direct_download, force_download=force_download, dry_run=dry_run,
        )


class Downloader2(DownloaderModel):

    _default_file_name_patterns = ['SW_OPER']

    def __init__(
            self, dt_fr, dt_to,
            sat_id=None,
            file_version=None,
            file_extension='.cdf',
            data_file_root_dir=None,
            ftp_data_dir=None,
            force=True, direct_download=True, **kwargs
    ):

        if ftp_data_dir is None:
            ftp_data_dir = f'Level2daily/Latest_baselines/AEJ/LPS/Sat_{sat_id.upper()}'

        if data_file_root_dir is None:
            data_file_root_dir = prf.datahub_data_root_dir / "ESA" / "SWARM" / "Level2daily" / "AEJ_LPS"
        file_name_patterns = list(self._default_file_name_patterns)

        file_name_patterns.extend(['AEJ' + sat_id.upper()])
        super(Downloader, self).__init__(
            dt_fr, dt_to,
            sat_id=sat_id,
            data_file_root_dir=data_file_root_dir,
            ftp_data_dir=ftp_data_dir,
            file_version=file_version,
            file_extension=file_extension,
            force=force, direct_download=direct_download,
            file_name_patterns=file_name_patterns, **kwargs
        )

    def download(self, **kwargs):

        done = super(Downloader, self).download(**kwargs)
        return done

    def search_files(self, **kwargs):

        file_list, versions = super(Downloader, self).search_files(**kwargs)

        return file_list, versions
        # version control