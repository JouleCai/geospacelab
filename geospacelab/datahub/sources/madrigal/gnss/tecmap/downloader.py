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
    dt_fr = datetime.datetime(2021, 3, 10)
    dt_to = datetime.datetime(2021, 3, 10)
    download_obj = Downloader(dt_fr, dt_to)


data_type_dict = {
    'TEC-MAP': 'binned',
    'TEC-LOS': 'Line of sight',
    'TEC-Sites': 'List of sites',
}


class Downloader(object):
    """Download the GNSS TEC data
    """

    def __init__(self, dt_fr, dt_to, data_file_root_dir=None, file_type='TEC-MAP',
                 user_fullname=madrigal.default_user_fullname,
                 user_email=madrigal.default_user_email,
                 user_affiliation=madrigal.default_user_affiliation):
        self.user_fullname = user_fullname
        self.user_email = user_email
        self.user_affiliation = user_affiliation

        dt_fr = dttool.get_start_of_the_day(dt_fr)
        dt_to = dttool.get_end_of_the_day(dt_to)
        self.file_type = file_type
        if dt_fr == dt_to:
            dt_to = dt_to + datetime.timedelta(hours=23, minutes=59)
        self.dt_fr = dt_fr  # datetime from
        self.dt_to = dt_to  # datetime to

        if data_file_root_dir is None:
            self.data_file_root_dir = prf.datahub_data_root_dir / 'Madrigal' / 'GNSS' / 'TEC'
        else:
            self.data_file_root_dir = data_file_root_dir
        self.done = False

        self.madrigal_url = "http://cedar.openmadrigal.org/"
        self.download_madrigal_files()

    def download_madrigal_files(self, download_pp=False):
        icodes = [8000, ]
        for icode in icodes:
            mylog.simpleinfo.info("Searching data from the Madrigal database ...")
            exp_list, _, database = madrigal.utilities.list_experiments(
                icode, self.dt_fr, self.dt_to, madrigal_url=self.madrigal_url
            )
            for exp in exp_list:
                files = database.getExperimentFiles(exp.id)
                for file in files:
                    if data_type_dict[self.file_type] not in file.kindatdesc:
                        continue
                    file_path_remote = pathlib.Path(file.name)
                    file_name = file_path_remote.name
                    res = re.search(r'/([\d]+[a-z]+[\d]+)', file.doi)
                    dtstr = res.groups()[0]
                    dtstr = dtstr[0:2] + dtstr[2].upper() + dtstr[3:]
                    thisday = datetime.datetime.strptime(dtstr, "%d%b%y")

                    data_file_dir = self.data_file_root_dir / thisday.strftime("%Y") / thisday.strftime('%Y%m%d')
                    data_file_dir.mkdir(parents=True, exist_ok=True)
                    file_name_new = 'GNSS_' + self.file_type.replace('-', '_') + '_' + thisday.strftime('%Y%m%d') + \
                                    '.' + file_name.split('.')[-2] + '.' + file_name.split('.')[-1]
                    data_file_path = data_file_dir / file_name_new
                    if data_file_path.is_file():
                        mylog.simpleinfo.info("The file {} has been downloaded.".format(data_file_path.name))
                        continue

                    mylog.simpleinfo.info("Downloading  {} from the Madrigal database ...".format(file_name))
                    database.downloadFile(
                        file.name, data_file_path,
                        self.user_fullname, self.user_email, self.user_affiliation,
                        "hdf5"
                    )
                    self.done = True
                    mylog.simpleinfo.info("Save the file as {} in the directory {}".format(file_name_new, data_file_dir))



if __name__ == "__main__":
    test()
