import datetime
import pathlib
import copy

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.config import prf
from geospacelab.datahub.sources.cdaweb.downloader import Downloader as DownloaderBase


class Downloader(DownloaderBase):
    
    def __init__(
            self,
            dt_fr=None, dt_to=None,
            sat_id=None,
            orbit_id=None,
            direct_download=True,
            force_download=False,
            data_file_root_dir = None,
            dry_run=False,
        ):
        product = 'EDR_AUR'
        if data_file_root_dir is None:
            data_file_root_dir = prf.datahub_data_root_dir / 'CDAWeb' / 'DMSP' / 'SSUSI' / product
        self.sat_id = sat_id
        self.orbit_id = orbit_id
        self.source_subdirs = ['dmsp', 'dmsp'+self.sat_id.lower(), 'ssusi', 'data', 'edr-aurora']

        super().__init__(
            dt_fr, dt_to,
            data_file_root_dir=data_file_root_dir,
            direct_download=direct_download,force_download=force_download,dry_run=dry_run
        )


    def search_from_http(self, file_name_patterns=None, allow_multiple_files=True):

        dt_fr_1 = self.dt_fr - datetime.timedelta(hours=3)
        dt_to_1 = self.dt_to + datetime.timedelta(hours=3)
        diff_days = dttool.get_diff_days(dt_fr_1, dt_to_1)
        dt0 = dttool.get_start_of_the_day(dt_fr_1)
        source_file_paths = []
        for nd in range(diff_days + 1):
            this_day = dt0 + datetime.timedelta(days=nd)
            doy = dttool.get_doy(this_day)
            sdoy = '{:03d}'.format(doy)
            subdirs = copy.deepcopy(self.source_subdirs)
            subdirs.extend(
                [str(this_day.year), sdoy]
            )

            if self.orbit_id is None:
                file_name_patterns = [
                    'dmsp' + self.sat_id.lower(),
                    'ssusi',
                    'edr-aurora',
                    this_day.strftime("%Y") + sdoy + 'T',
                    '.nc'
                ]
            else:
                file_name_patterns = [
                    'dmsp' + self.sat_id.lower(),
                    'ssusi',
                    'edr-aurora',
                    'REV',
                    self.orbit_id,
                    '.nc'
                ]
            paths = super().search_from_http(subdirs=subdirs, file_name_patterns=file_name_patterns)
            source_file_paths.extend(paths)
        return source_file_paths

    def save_file_from_http(self, url, file_dir=None, file_name=None):

        sy = url.split('/')[-3]
        sdoy = url.split('/')[-2]
        year = int(sy)
        this_day = dttool.convert_doy_to_datetime(year, int(sdoy))
        if file_dir is None:
            file_dir = self.data_file_root_dir /  self.sat_id.upper() / sy / this_day.strftime("%Y%m%d")
        super().save_file_from_http(url, file_dir=file_dir)



if __name__ == "__main__":
    downloader = Downloader(
        dt_fr = datetime.datetime(2011, 1, 6),
        dt_to = datetime.datetime(2011, 1, 6, 12),
        sat_id='F17',
        orbit_id='21523',
        dry_run=False,
    )