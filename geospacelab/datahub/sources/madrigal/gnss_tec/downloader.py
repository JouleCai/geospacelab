# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


"""
To download the EISCAT quickplots and analyzed data archived in http://portal.eiscat.se/schedule/schedule.cgi
By Lei Cai on 2021.04.01
"""
import datetime
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as dtparse
import numpy as np
import re
import requests
import bs4
import os
import pathlib

from geospacelab import preferences as pfr
import geospacelab.datahub.sources.madrigal as madrigal
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool


def test():
    sites = ['UHF', 'ESR']
    dt_fr = datetime.datetime(2021, 3, 10)
    dt_to = datetime.datetime(2021, 3, 10)
    download_obj = Downloader(dt_fr, dt_to)


class Downloader(object):
    """Download the quickplots and archieved analyzed results from EISCAT schedule webpage
    """

    def __init__(self, dt_fr, dt_to, data_file_root_dir=None,
                 user_fullname=madrigal.default_user_fullname,
                 user_email=madrigal.default_user_email,
                 user_affiliation=madrigal.default_user_affiliation):
        self.user_fullname = user_fullname
        self.user_email = user_email
        self.user_affiliation = user_affiliation

        dt_fr = dttool.get_start_of_the_day(dt_fr)
        dt_to = dttool.get_start_of_the_day(dt_to)
        if dt_fr == dt_to:
            dt_to = dt_to + datetime.timedelta(hours=23, minutes=59)
        self.dt_fr = dt_fr  # datetime from
        self.dt_to = dt_to  # datetime to

        if data_file_root_dir is None:
            self.data_file_root_dir = pfr.datahub_data_root_dir / 'Madrigal' / 'EISCAT' / 'analyzed'
        else:
            self.data_file_root_dir = data_file_root_dir
        self.done = False

        self.madrigal_url = "https://madrigal.eiscat.se/madrigal/"
        self.download_madrigal_files()

    def download_madrigal_files(self, download_pp=False):
        icodes = []
        for site in self.sites:
            icodes.extend(instrument_codes[site])
        for icode in icodes:
            exp_list, _, database = madrigal.list_experiments(icode, self.dt_fr, self.dt_to,
                                                              madrigal_url=self.madrigal_url)
            for exp in exp_list:
                files = database.getExperimentFiles(exp.id)
                for file in files:
                    if not download_pp and 'GUISDAP pp' in file.kindatdesc:
                        continue
                    file_path = pathlib.Path(file.name)
                    site = file_path.name.split("@")[1][0:3].upper()
                    if '32' in site or '42' in site:
                        site = 'ESR'

                    match = re.search('\d{4}-\d{2}-\d{2}', file_path.name)
                    dt_str = match.group(0)
                    thisday = datetime.datetime.strptime(dt_str, "%Y-%m-%d")
                    if thisday < self.dt_fr or thisday > self.dt_to:
                        continue

                    # sub_dir = file_path.name.split('_', maxsplit=1)[1]
                    search_pattern = re.search("\d{4}-\d{2}-\d{2}_[a-zA-Z0-9]*", file_path.name).group(0)
                    sub_dir = search_pattern + '@' + site
                    data_file_dir = self.data_file_root_dir / site / dt_str[0:4] / sub_dir
                    data_file_dir.mkdir(parents=True, exist_ok=True)
                    data_file_path = data_file_dir / file_path.name
                    if data_file_path.is_file():
                        mylog.simpleinfo.info("The file {} has been downloaded.".format(data_file_path.name))
                        continue

                    mylog.simpleinfo.info("Downloading  {} from the Madrigal database ...".format(file_path.name))
                    database.downloadFile(
                        file_path, data_file_path,
                        self.user_fullname, self.user_email, self.user_affiliation,
                        "hdf5"
                    )
                    self.done = True
                    mylog.simpleinfo.info("Done!")


        # fhdf5 = h5py.File(outDir + fn, 'r')