# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import datetime
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as dtparse
import numpy as np
import re
import requests
import bs4
import os
import pathlib

from geospacelab.config import prf
import geospacelab.datahub.sources.madrigal as madrigal
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool


def test():
    dt_fr = datetime.datetime(2014, 3, 10)
    dt_to = datetime.datetime(2014, 3, 10)
    download_obj = Downloader(dt_fr, dt_to, sat_id='f16', file_type='s1')


data_type_dict = {
    's1': 'ion drift',
    's4': 'plasma temp',
    'e': 'flux/energy',
    'ssies2': 'SSIES-2',
    'hp':     'Hemispherical power',
    'ssies3':   'SSIES-3'
}


class Downloader(object):
    """Download the GNSS TEC data
    """

    def __init__(self, dt_fr, dt_to, sat_id=None, file_type=None, data_file_root_dir=None, force=False,
                 user_fullname=madrigal.default_user_fullname,
                 user_email=madrigal.default_user_email,
                 user_affiliation=madrigal.default_user_affiliation):
        self.user_fullname = user_fullname
        self.user_email = user_email
        self.user_affiliation = user_affiliation

        self.instrument_code = 8100

        dt_fr = dttool.get_start_of_the_day(dt_fr)
        dt_to = dttool.get_end_of_the_day(dt_to)

        if sat_id is None:
            sat_id = input("input the sat_id (e.g., f16)")
        self.sat_id = sat_id

        if file_type is None:
            file_type = input("Input the file type (s1, s4, or e):")
        self.file_type = file_type
        if dt_fr == dt_to:
            dt_to = dt_to + datetime.timedelta(hours=23, minutes=59)
        self.dt_fr = dt_fr  # datetime from
        self.dt_to = dt_to  # datetime to
        self.force = force

        if data_file_root_dir is None:
            self.data_file_root_dir = prf.datahub_data_root_dir / 'Madrigal' / 'DMSP'
        else:
            self.data_file_root_dir = data_file_root_dir
        self.done = False

        self.madrigal_url = "http://cedar.openmadrigal.org/"
        self.download_madrigal_files()

    def download_madrigal_files(self, download_pp=False):

        mylog.simpleinfo.info("Searching data from the Madrigal database ...")
        exp_list, _, database = madrigal.utilities.list_experiments(
            self.instrument_code, self.dt_fr, self.dt_to, madrigal_url=self.madrigal_url
        )
        for exp in exp_list:
            files = database.getExperimentFiles(exp.id)
            for file in files:
                if self.file_type == 'hp':
                    rpattern = data_type_dict[self.file_type]
                else:
                    rpattern = self.sat_id.upper() + '.*' + data_type_dict[self.file_type]

                m = re.search(rpattern, file.kindatdesc)
                if m is None:
                    if '_' + self.sat_id[1:] + self.file_type not in file.name:
                        continue

                file_path_remote = pathlib.Path(file.name)
                file_name = file_path_remote.name
                m = re.search(r'([\d]{8})', file_name)
                dtstr = m.group()
                thisday = datetime.datetime.strptime(dtstr, "%Y%m%d")
                if thisday < datetime.datetime(self.dt_fr.year, self.dt_fr.month, self.dt_fr.day):
                    continue
                if thisday > datetime.datetime(self.dt_to.year, self.dt_to.month, self.dt_to.day):
                    continue

                data_file_dir = self.data_file_root_dir / thisday.strftime("%Y%m") / thisday.strftime('%Y%m%d')
                data_file_dir.mkdir(parents=True, exist_ok=True)

                data_file_path = data_file_dir / file_name
                if data_file_path.is_file():
                    mylog.simpleinfo.info("The file {} has been downloaded.".format(data_file_path.name))
                    if not self.force:
                        continue
                    else:
                        print('Force downloading ...')

                mylog.simpleinfo.info("Downloading  {} from the Madrigal database ...".format(file_name))
                database.downloadFile(
                    file.name, data_file_path,
                    self.user_fullname, self.user_email, self.user_affiliation,
                    "hdf5"
                )
                self.done = True
                mylog.simpleinfo.info("Save the file as {} in the directory {}".format(file_name, data_file_dir))


if __name__ == "__main__":
    test()
