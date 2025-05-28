__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import datetime
import h5py

from geospacelab.datahub.sources.madrigal.isr.risr_n.downloader import Downloader as DownloaderBase
from geospacelab.config import prf



INCLUDE_FILE_TYPE_PATTERNS = [['velocity']]


class Downloader(DownloaderBase):

    def __init__(
            self, dt_fr: datetime.datetime, dt_to: datetime,
            include_exp_name_patterns: list=None,
            exclude_exp_name_patterns: list=None,
            include_exp_ids: list = None,
            exclude_exp_ids: list= None,
            include_file_name_patterns: list = None,
            exclude_file_name_patterns: list = None,
            include_file_type_patterns=None,
            exclude_file_type_patterns=None,
            data_file_root_dir: str = None,
            direct_download = True,
            force_download = False,
            dry_run: bool=False,
            madrigal_url: str = "https://data.amisr.com/madrigal"):
        exclude_exp_ids = [100213840, ] if exclude_exp_ids is None else exclude_exp_ids
        include_file_type_patterns = INCLUDE_FILE_TYPE_PATTERNS if include_file_type_patterns is None else include_file_type_patterns
        data_product = 'vi'
        data_file_root_dir = prf.datahub_data_root_dir / 'Madrigal' / 'RISR-N' if data_file_root_dir is None else data_file_root_dir
        super().__init__(
            dt_fr=dt_fr, dt_to=dt_to,
            data_product=data_product,
            include_exp_name_patterns=include_exp_name_patterns,
            exclude_exp_name_patterns=exclude_exp_name_patterns,
            include_exp_ids=include_exp_ids,
            exclude_exp_ids=exclude_exp_ids,
            include_file_name_patterns=include_file_name_patterns,
            exclude_file_name_patterns=exclude_file_name_patterns,
            include_file_type_patterns=include_file_type_patterns,
            exclude_file_type_patterns=exclude_file_type_patterns,
            data_file_root_dir=data_file_root_dir,
            force_download=force_download, direct_download=direct_download, dry_run=dry_run,
            madrigal_url=madrigal_url,)
        
    def download(self, **kwargs):
        
        file_paths = super().download(**kwargs)
        
        return file_paths
            
            


