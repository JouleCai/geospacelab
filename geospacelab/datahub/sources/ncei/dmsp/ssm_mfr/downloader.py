import pathlib
import re
import datetime

import numpy as np

from geospacelab.config import prf
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool
from geospacelab.datahub.sources.ncei.dmsp.downloader import Downloader as DownloaderBase


class Downloader(DownloaderBase):

    def __init__(
            self,
            dt_fr=None,
            dt_to=None,
            sat_id=None, file_dir_root=None,
            force_download=False,
            direct_download=True,
    ):
        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.sat_id = sat_id

        if file_dir_root is None:
            file_dir_root = prf.datahub_data_root_dir / 'NCEI' / 'DMSP' / 'SSM_MFR'

        self.file_dir_root = pathlib.Path(file_dir_root)

        super().__init__(force_download=force_download, direct_download=direct_download)

    def download(self, **kwargs):

        diff_months = dttool.get_diff_months(self.dt_fr, self.dt_to)

        files = []
        for nm in range(diff_months+1):
            this_month = dttool.get_next_n_months(self.dt_fr, nm)
            sub_url = '/'.join([
                self.sat_id.lower(),
                'ssm', this_month.strftime("%Y/%m"),
            ])
            url = self._url_base + sub_url
            files_seg = self.search_files(url, file_name_patterns=['SSM', 'MFR'])
            if files_seg is None:
                continue
            files.extend(files_seg)

        for furl in files:
            rc = re.compile(r"DD.(\d{8})")
            dstr = rc.search(furl).groups()[0]
            this_day = datetime.datetime.strptime(dstr, '%Y%m%d')
            if this_day < dttool.get_start_of_the_day(self.dt_fr) or this_day > dttool.get_end_of_the_day(self.dt_to):
                continue

            file_dir = self.file_dir_root / self.sat_id.upper() / this_day.strftime("%Y/%m")
            file_dir.mkdir(parents=True, exist_ok=True)
            done = super().download(furl, file_dir=file_dir)
            self.done = done


if __name__ == "__main__":
    download_obj = Downloader(
        sat_id='F18',
        dt_fr=datetime.datetime(2024, 5, 10),
        dt_to=datetime.datetime(2024, 5, 13),
        force_download=True
    )