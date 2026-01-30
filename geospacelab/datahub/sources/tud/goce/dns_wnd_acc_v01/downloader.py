# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import datetime
from geospacelab.config import prf

from geospacelab.datahub.sources.tud.downloader import TUDownloader as DownloaderBase   


class Downloader(DownloaderBase):
    
    def __init__(self, 
        dt_fr, dt_to, 
        mission='GOCE', 
        sat_id='',
        version='v01',  
        product='DNS-WND-ACC',
        direct_download=True,
        force_download=False,
        dry_run=False,
        ):
        self.mission = mission
        self.version = version
        
        super().__init__(
            dt_fr, dt_to, 
            mission=mission,
            sat_id=sat_id, 
            version=version,
            product=product,
            direct_download=direct_download,
            force_download=force_download,
            dry_run=dry_run
        )
        