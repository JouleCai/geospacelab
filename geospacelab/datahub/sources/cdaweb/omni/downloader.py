# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import datetime
import requests
import bs4
import re

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.config import prf


class Downloader(object):
    def __init__(self, dt_fr,  dt_to, res='1min', new_omni=True, data_file_root_dir=None, version=None):
        if res == '1h':
            new_omni = True
        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.res = res
        self.new_omni = new_omni
        self.version = version
        self.done = False
        if data_file_root_dir is None:
            self.data_file_root_dir = prf.datahub_data_root_dir / "CDAWeb" / 'OMNI'
        else:
            self.data_file_root_dir = data_file_root_dir

        self.url_base = "https://cdaweb.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/"

        self.download()

    def download(self):
        if self.res in ['1min', '5min']:
            self.download_high_res_omni()
        elif self.res == '1h':
            self.download_low_res_omni()

    def download_high_res_omni(self):
        if self.new_omni:
            omni_type = "hro2"
            omni_dir_name = 'OMNI2'
        else:
            omni_type = "hro"
            omni_dir_name = "OMNI"

        num_month = (self.dt_to.year - self.dt_fr.year) * 12 + self.dt_to.month - self.dt_fr.month + 1
        dt0 = datetime.datetime(self.dt_fr.year, self.dt_fr.month, 1)
        for nm in range(num_month):
            dt1 = dttool.get_next_n_months(dt0, nm)
            url = self.url_base + omni_type + '_' + self.res + '/' + '{:4d}'.format(dt1.year) + '/'

            r = requests.get(url)

            soup = bs4.BeautifulSoup(r.text, 'html.parser')
            a_tags = soup.find_all('a', href=True)
            hrefs = []
            for a_tag in a_tags:
                href = a_tag['href']
                pattern = omni_type + '_' + self.res + '_' + dt1.strftime("%Y%m%d")
                if pattern in href:
                    hrefs.append(href)
            if len(hrefs) == 0:
                mylog.StreamLogger.info("Cannot find the queried data file!")
                return
            if len(hrefs) > 1:
                mylog.StreamLogger.warning("Find multiple matched files!")
                print(hrefs)
                return

            href = hrefs[0]
            ma = re.search('v[0-5][0-9]', href)
            version = ma.group(0)
            if self.version is None:
                self.version = version
            elif self.version != version:
                mylog.StreamLogger.info("Cannot find the queried data file! Version={}.".format(version))

            r_file = requests.get(url + href, allow_redirects=True)
            file_name = href
            file_path = self.data_file_root_dir / (omni_dir_name + '_high_res_' + self.res)
            file_path = file_path / '{:4d}'.format(dt1.year) / file_name
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
            self.done = True

    def download_low_res_omni(self):
        raise NotImplemented


def test():
    dt_fr = datetime.datetime(2020, 3, 4)
    dt_to = datetime.datetime(2020, 5, 3)
    download_obj = Downloader(dt_fr, dt_to)
    pass


if __name__ == "__main__":
    test()
