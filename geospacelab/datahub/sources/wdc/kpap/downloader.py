
import datetime
import numpy as np
import requests
import bs4
import re

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab import preferences as prf


class Downloader(object):

    def __init__(self, dt_fr,  dt_to, data_file_root_dir=None):

        self.dt_fr = dt_fr
        self.dt_to = dt_to

        self.done = False
        if data_file_root_dir is None:
            self.data_file_root_dir = prf.datahub_data_root_dir / 'WDC' / 'KpAp'
        else:
            self.data_file_root_dir = data_file_root_dir

        self.url_base = "http://wdc.kugi.kyoto-u.ac.jp"

        self.download()

    def download(self):
        diff_year = self.dt_to.year - self.dt_fr.year
        dt0 = datetime.datetime(self.dt_fr.year, 1, 1)
        r = requests.get(self.url_base + '/kp/')
        soup = bs4.BeautifulSoup(r.text, 'html.parser')
        form_tag = soup.find_all('form')
        r_method = form_tag[0].attrs['method']
        r_action_url = self.url_base + form_tag[0].attrs['action']
        for i in range(diff_year):
            dt1 = datetime.datetime(dt0.year + i, 1, 1)
            dt2 = datetime.datetime(dt0.year + i, 12, 31)
            data_form = {
                'SCent': int(dt1.year/100),
                'STens': int((dt1.year - np.floor(dt1.year/100)*100) / 10),
                'SYear': int((dt1.year - np.floor(dt1.year/10)*10)),
                'From': 1,
                'ECent': int(dt2.year/100),
                'ETens': int((dt2.year - np.floor(dt2.year/100)*100) / 10),
                'EYear': int((dt2.year - np.floor(dt2.year/10)*10)),
                'To': 12,
                'Email': 'lei.cai@oulu.fi'
            }
            if r_method.lower() == 'post':
                r_file = requests.post(r_action_url, data=data_form)

            if 'YYYYMMDD' not in r_file.text:
                return

            file_name = 'WDC_KpAp_' + dt1.strftime('%Y%m.dat')
            file_path = self.data_file_root_dir / '{:4d}'.format(dt1.year) / file_name
            if file_path.is_file():
                mylog.simpleinfo.info(
                    "The file {} exists in the directory {}.".format(file_path.name, file_path.parent.resolve()))
            else:
                file_path.parent.resolve().mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as omni:
                    mylog.simpleinfo.info(
                        "Downloading {} to the directory {} ...".format(file_path.name, file_path.parent.resolve())
                    )
                    omni.write(r_file.content)
                    mylog.simpleinfo.info("Done")