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
            sat_id=None,
            product='DNS-ACC',
            version='v02',
            force=True, direct_download=True, **kwargs
    ):
        if version == 'v01':
            raise ValueError
        elif 'v02' in version:
            v_str = "version_02"
        else:
            raise NotImplementedError
        if sat_id == 'FO1':
            sat_id = 'C'
        else:
            raise NotImplementedError
        
        data_file_root_dir = prf.datahub_data_root_dir / "TUD" / "GRACE-FO" / product.upper() / version
        ftp_data_dir = f'{v_str}/GRACE-FO_data'
        file_name_patterns = [sat_id.upper(), product.replace('-', '_')]
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