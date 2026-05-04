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

from geospacelab.datahub.sources.esa_eo.swarm.downloader import DownloaderSwarm


class Downloader(DownloaderSwarm):
    
    def __init__(
        self,
        dt_fr, dt_to,
        sat_id=None,
        product='MAG_LR',
        product_version='latest',
        file_extension='.cdf',
        data_file_root_dir=None,
        force_indexing = False,
        query_mode='off',
        direct_download=True, force_download=False, dry_run=False,
    ):

        mission = 'SWARM'
        file_class = 'OPER'
        product_level = '1B'
        root_dir_remote = '/Level1b'
        
        if data_file_root_dir is None:
            data_file_root_dir = prf.datahub_data_root_dir / "ESA" / "SWARM" / "Level1b" / "MAG_LR"
        root_dir_local = data_file_root_dir
        
        self._default_sub_dirs_remote = {
            'latest': ['Latest_baselines', 'MAGx_LR'],
            'Older_baselines': ['Older_baselines', 'MAGx_LR'], 
            'Entire_mission': ['Entire_mission_data', 'MAGx_LR']
        }
        
        super(Downloader, self).__init__(
            dt_fr=dt_fr, dt_to=dt_to,
            mission=mission,
            sat_id=sat_id,
            file_class=file_class,
            file_extension=file_extension,
            product=product,
            product_patterns=None,
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

    
    def _to_download(self, file_path, with_suffix=None):
        return super()._to_download(file_path, with_suffix)
