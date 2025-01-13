# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import datetime
import requests
import bs4
import certifi
import os
import zlib

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.config import prf


class Downloader(object):
    """
    A class to Download SSUSI data
    :param file_type:  "l1b", "edr-aur", or "sdr"
    """
    def __init__(self, dt_fr, dt_to, sat_id=None, orbit_id=None, data_file_root_dir=None, file_type='edr-aur'):

        dt_fr = dttool.get_start_of_the_day(dt_fr)
        dt_to = dttool.get_start_of_the_day(dt_to)

        self.file_type = file_type.lower()
        if dt_fr == dt_to:
            dt_to = dt_to + datetime.timedelta(hours=23, minutes=59)
        self.dt_fr = dt_fr  # datetime from
        self.dt_to = dt_to  # datetime to
        self.sat_id = sat_id.lower()
        self.orbit_id = orbit_id

        if data_file_root_dir is None:
            self.data_file_root_dir = prf.datahub_data_root_dir / 'JHUAPL' / 'DMSP' / 'SSUSI'
        else:
            self.data_file_root_dir = data_file_root_dir
        self.done = False

        self.url_base = "https://ssusi.jhuapl.edu/"

        self.download_files()

    def download_files(self):
        """
        Get a list of the urls from input date
        and datatype and download the files
        and also move them to the corresponding
        folders.!!!
        """
        diff_days = dttool.get_diff_days(self.dt_fr, self.dt_to)
        dt0 = dttool.get_start_of_the_day(self.dt_fr)

        for iday in range(diff_days + 1):
            thisday = dt0 + datetime.timedelta(days=iday)

            # construct day of year from date
            doy = thisday.timetuple().tm_yday
            doy_str = "{:03d}".format(doy)

            if self.file_type in ['l1b', 'edr-aur', 'edr-iono']:
                payload_type = self.file_type
            elif self.file_type in ['sdr-limb', 'sdr-disk', 'sdr2-disk']:
                payload_type = 'sdr'
            elif self.file_type in ['edr-night-disk', 'edr-day-disk']:
                payload_type = 'edr-disk'
            elif self.file_type in ['edr-night-limb', 'edr-day-limb']:
                payload_type = 'edr-limb'
            elif self.file_type in ['edr-gaim-disk', 'edr-gaim-lim']:
                payload_type = 'edr-gaim'
            else:
                raise NotImplementedError

            payload = {"spc": self.sat_id, "type": payload_type,
                       "Doy": doy_str, "year": "{:d}".format(thisday.year)}

            # get a list of the files from dmsp ssusi website
            # based on the data type and date
            try:
                r = requests.get(self.url_base + "data_retriver/",
                                params=payload, verify=True)
            except:
                r = requests.get(self.url_base + "data_retriver/",
                                params=payload, verify=False)
            soup = bs4.BeautifulSoup(r.text, 'html.parser')
            div_filelist = soup.find("div", {"id": "filelist"})
            href_list = div_filelist.find_all(href=True)
            url_list = [self.url_base + href["href"] for href in href_list]

            import natsort
            url_list = natsort.natsorted(url_list, reverse=False)

            for f_url in url_list:

                # we only need data files which have .NC
                if ".NC" not in f_url:
                    continue
                # If working with sdr data use only
                # sdr-disk files
                if self.file_type.upper() not in f_url:
                    continue

                if self.orbit_id is not None:
                    if len(self.orbit_id) != 5:
                        raise ValueError
                    if self.orbit_id not in f_url:
                        continue

                file_dir = self.data_file_root_dir / self.sat_id.lower() / thisday.strftime("%Y%m%d")
                file_dir.mkdir(parents=True, exist_ok=True)
                file_name = f_url.split('/')[-1]
                file_path = file_dir / file_name
                if file_path.is_file():
                    self.done = True
                    mylog.simpleinfo.info("The file {} exists.".format(file_name))
                    continue
                mylog.simpleinfo.info("Downloading {} from the online database ...".format(file_name))
                try:
                    rf = requests.get(f_url, verify=True)
                except:
                    rf = requests.get(f_url, verify=False)
                file_name = rf.url.split("/")[-1]

                with open(file_path, "wb") as ssusi_data:
                    ssusi_data.write(rf.content)
                mylog.simpleinfo.info("Done. The file has been saved to {}".format(file_dir))
                self.done = True
                if self.orbit_id is not None:
                    return
                # self.file_dir = file_dir
                # self.file_name = file_name
            if self.orbit_id is None and self.done:
                fp_log = file_dir / (self.file_type.upper() + '.full.log')
                fp_log.touch()
            if not self.done:
                mylog.StreamLogger.warning(
                    "Cannot find the requested data on {} from the online database!".format(thisday.strftime("%Y-%m-%d")))


if __name__ == "__main__":
    dt_fr = datetime.datetime(2015, 12, 5)
    dt_to = datetime.datetime(2015, 12, 6)
    file_type = 'edr-aur'
    sat_id = 'f17'
    orbit_id = '46871'
    orbit_id = None

    Downloader(dt_fr, dt_to, sat_id=sat_id, orbit_id=orbit_id, file_type=file_type)