import datetime
import pathlib
import copy

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.config import prf
from geospacelab.datahub.sources.cdaweb.downloader import CDAWebHTTPDownloader as DownloaderBase


class Downloader(DownloaderBase):
    
    def __init__(
            self,
            dt_fr=None, dt_to=None,
            sat_id=None,
            orbit_id=None,
            product='EDR_AUR',
            direct_download=True,
            force_download=False,
            dry_run=False,
            root_dir_local = None,
        ):
         
        self.sat_id = sat_id
        self.orbit_id = orbit_id
        self.product = product
        if root_dir_local is None:
            root_dir_local = prf.datahub_data_root_dir / 'CDAWeb' / 'DMSP' / 'SSUSI' / self.product.upper
        else:
            root_dir_local = pathlib.Path(root_dir_local)
        root_dir_remote = '/'.join([
            self.root_dir_remote, 'dmsp', 'dmsp'+self.sat_id.lower(), 'ssusi', 'data', product_dict[self.product]
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

        dt_fr_1 = self.dt_fr - datetime.timedelta(hours=3)
        dt_to_1 = self.dt_to + datetime.timedelta(hours=3)
        diff_days = dttool.get_diff_days(dt_fr_1, dt_to_1)
        dt0 = dttool.get_start_of_the_day(dt_fr_1)
        file_paths_remote = []
        for nd in range(diff_days + 1):
            this_day = dt0 + datetime.timedelta(days=nd)
            doy = dttool.get_doy(this_day)
            sdoy = '{:03d}'.format(doy)
            subdirs = [str(this_day.year), sdoy]

            if self.orbit_id is None:
                file_name_patterns = [
                    'dmsp' + self.sat_id.lower(),
                    'ssusi',
                    product_dict[self.product],
                    this_day.strftime("%Y") + sdoy + 'T',
                    '.nc'
                ]
            else:
                file_name_patterns = [
                    'dmsp' + self.sat_id.lower(),
                    'ssusi',
                    product_dict[self.product],
                    'REV',
                    self.orbit_id,
                    '.nc'
                ]
            paths = super().search_from_http(subdirs=subdirs, file_name_patterns=file_name_patterns)
            file_paths_remote.extend(paths)
        return file_paths_remote
    
    def save_files_from_http(self, file_paths_local=None, root_dir_remote=None):
        if file_paths_local is None:
            file_paths_local = []
            for fp_remote in self.file_paths_remote:
                sy = fp_remote.split('/')[-3]
                sdoy = fp_remote.split('/')[-2]
                year = int(sy)
                this_day = dttool.convert_doy_to_datetime(year, int(sdoy))
                file_dir_local = self.root_dir_local /  self.sat_id.upper() / sy / this_day.strftime("%Y%m%d")
                file_paths_local.append(file_dir_local / fp_remote.split('/')[-1])
        return super().save_files_from_http(file_paths_local, root_dir_remote)
                

product_dict = {
    'EDR_AUR': 'edr-aurora',
    'EDR_DAY_LIMB': 'edr-day-limb',
    'EDR_IONO': 'edr-iono',
    'EDR_NIGHT_LIMB': 'edr-night-limb',
    'L1B': 'l1b',
    'SDR_DISK': 'sdr-disk',
    'SDR_LIMB': 'sdr-limb',
    'SDR2_DISK': 'sdr2-disk',
    'SPECT_L1B': 'spect-l1b',
}

