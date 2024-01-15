import datetime
import numpy as np
import pathlib
import re
import requests
import tqdm

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.config import prf


class Downloader(object):

    def __init__(
            self, dt_fr, dt_to,
            data_file_root_dir=None,
            data_product='grd', pole='N', user_name=None, direct_download=True, force_download=False):
        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.data_product = data_product
        self.user_name = user_name
        self.pole = pole
        self.force_download = force_download
        self.base_url = "https://ampere.jhuapl.edu/services"
        self.done = False
        self.data_file_root_dir = data_file_root_dir
        self.data_file_paths = []
        if self.data_file_root_dir is None:
            self.data_file_root_dir = prf.datahub_data_root_dir / 'JHUAPL' / 'AMPERE' / data_product.upper()

        if direct_download:
            self.download()

    def download(self):
        diff_days = dttool.get_diff_days(self.dt_fr, self.dt_to)
        dt_0 = dttool.get_start_of_the_day(self.dt_fr)
        for nd in range(diff_days+1):
            this_day = dt_0 + datetime.timedelta(days=nd)
            
            for hh in range(24):
             
                time_start = this_day + datetime.timedelta(hours=hh)
                time_extent = 3600 - 1
                time_end = time_start + datetime.timedelta(seconds=time_extent)
                if (time_start < self.dt_fr) or (time_end > self.dt_to):
                    continue
                params = {
                    'logon': self.user_name,
                    'start': time_start.strftime('%Y-%m-%dT%H:%M'),
                    'end': time_end.strftime('%Y-%m-%dT%H:%M'),
                    'pole': self.pole_str
                }
                url = self.base_url + '/data-' + self.data_product.lower() + '.php'
                file_name = '_'.join([
                    'AMPERE',
                    self.data_product.upper(),
                    time_start.strftime('%Y%m%dT%H%M'),
                    time_end.strftime('%Y%m%dT%H%M'),
                    self.pole,
                ]) + '.nc'
                file_dir = self.data_file_root_dir / dt_0.strftime("%Y%m%d")
                file_dir.mkdir(parents=True, exist_ok=True)
                fp = file_dir / file_name
                if fp.is_file():
                    if not self.force_download:
                        self.done = True
                        mylog.simpleinfo.info("The file {} exists in {}.".format(file_name, file_dir))
                        self.data_file_paths.append(fp)
                        continue
                mylog.simpleinfo.info("Downloading {} from the online database ...".format(file_name))
                status = self._download_by_requests_get(url, data_file_path=fp, params=params)
                if status == -1:
                    mylog.StreamLogger.warning(
                        f'Failed to download the AMPERE {self.data_product.upper()} data between' 
                        + f'{time_start.strftime("%Y%m%dT%H:%M")} and {time_end.strftime("%Y%m%dT%H:%M")}!')
                mylog.simpleinfo.info("Done. The file has been saved to {}".format(file_dir))
                self.data_file_paths.append(fp)
                self.done = True

    def _download_by_requests_get(self, url, data_file_path=None, params=None, stream=True, **kwargs):
        r = requests.get(url, params=params, stream=stream, **kwargs)
        if r.status_code != 200:
            return -1

        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        with open(data_file_path, 'wb') as file:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            mylog.StreamLogger.error("Something wrong during the download!")
            data_file_path.unlink(missing_ok=True)
            return -1

        return 1

    @property
    def pole(self):
        return self._pole
    
    @pole.setter
    def pole(self, value):
        if value.upper() in ['N', 'NORTH']:
            self.pole_str = 'north'
        elif value.upper() in ['S', 'SOUTH']:
            self.pole_str = 'south'
        else:
            raise ValueError
        self._pole = value  


if __name__ == "__main__":
    dt_fr = datetime.datetime(2016, 3, 14, 1)
    dt_to = datetime.datetime(2016, 3, 15, 1)
    data_file_root_dir = pathlib.Path(__file__).parent.resolve()
    download = Downloader(dt_fr=dt_fr, dt_to=dt_to, data_file_root_dir=data_file_root_dir,
                          pole='N', user_name='leicai') 
