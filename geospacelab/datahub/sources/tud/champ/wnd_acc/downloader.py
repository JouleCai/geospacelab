# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

from geospacelab.config import prf

from geospacelab.datahub.sources.tud.downloader import Downloader as DownloaderModel


class Downloader(DownloaderModel):

    def __init__(
            self, dt_fr, dt_to,
            product='WND-ACC',
            version='v01',
            force=True, direct_download=True, **kwargs
    ):
        if version == 'v01':
            v_str = "version_01"
        elif version == 'v02':
            raise NotImplementedError
        else:
            raise NotImplementedError
        data_file_root_dir = prf.datahub_data_root_dir / "TUD" / "CHAMP" / product.upper() / version
        ftp_data_dir = f'{v_str}/CHAMP_data'
        file_name_patterns = ['CH', product.replace('-', '_')]
        super(Downloader, self).__init__(
            dt_fr, dt_to,
            data_file_root_dir=data_file_root_dir, ftp_data_dir=ftp_data_dir, force=force,
            direct_download=direct_download, file_name_patterns=file_name_patterns, **kwargs
        )

    def download(self, **kwargs):

        done = super(Downloader, self).download(**kwargs)
        return done

    def search_files(self, **kwargs):

        file_list = super(Downloader, self).search_files(**kwargs)

        return file_list
        # version control