import datetime
import pathlib
import copy

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.config import prf
from geospacelab.datahub.sources.cdaweb.dmsp.ssusi.downloader import Downloader as DownloaderBase


class Downloader(DownloaderBase):
    
    def __init__(
            self,
            dt_fr=None, dt_to=None,
            sat_id=None,
            orbit_id=None,
            direct_download=True,
            force_download=False,
            dry_run=False,
            root_dir_local = None,
        ):
         
        super().__init__(
            dt_fr, dt_to,
            sat_id=sat_id,
            orbit_id=orbit_id,
            product='SDR_DISK',
            root_dir_local=root_dir_local,
            direct_download=direct_download,
            force_download=force_download,
            dry_run=dry_run,
        ) 


if __name__ == "__main__":
    downloader = Downloader(
        dt_fr = datetime.datetime(2011, 1, 6),
        dt_to = datetime.datetime(2011, 1, 6, 12),
        sat_id='F17',
        orbit_id='21523',
        force_download=True,
        dry_run=False,
    )