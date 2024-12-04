# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


from cProfile import label
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
    dt_fr = datetime.datetime(2016, 3, 14)
    dt_to = datetime.datetime(2016, 3, 14)
    download_obj = Downloader(dt_fr, dt_to, dry_run=False)


file_type_dict = {
    'combined': 'combined',
    'zenith single pulse': 'zenith single-pulse',
    'zenith alternating': 'zenith alternating',
    'misa single-pulse': 'misa (steerable) single-pulse',
    'misa alternating': 'misa (steerable) alternating',
    'gridded': 'gridded',
    'ion velocity': 'ion velocities',
}


class Downloader(object):
    """Download the Millstone ISR data
    """

    def __init__(self, dt_fr, dt_to, data_file_root_dir=None, file_type='combined', exp_name_pattern='',
                 dry_run=False,
                 user_fullname=madrigal.default_user_fullname,
                 user_email=madrigal.default_user_email,
                 user_affiliation=madrigal.default_user_affiliation):
        """

        :param dt_fr: Starting time.
        :type dt_fr: datetime.datetime
        :param dt_to: Stopping time.
        :type dt_to: datetime.datetime
        :param data_file_root_dir: The root directory for the data.
        :type data_file_root_dir: str  or pathlib.Path
        :param file_type: The file type associated with the data product.
        :type file_type: str, {'basic', 'zenith single-pulse', 'zenith alternating',
            'misa (steerable) single-pulse', 'misa (steerable) alternating', 'gridded', 'ion-velocities'
        :param experiment_key:
        :param user_fullname:
        :param user_email:
        :param user_affiliation:
        """
        self.user_fullname = user_fullname
        self.user_email = user_email
        self.user_affiliation = user_affiliation

        dt_fr = dttool.get_start_of_the_day(dt_fr)
        dt_to = dttool.get_end_of_the_day(dt_to)
        self.file_type = file_type
        self.exp_name_pattern = exp_name_pattern
        self.dry_run = dry_run
        if dt_fr == dt_to:
            dt_to = dt_to + datetime.timedelta(hours=23, minutes=59)
        self.dt_fr = dt_fr  # datetime from
        self.dt_to = dt_to  # datetime to

        if self.file_type not in file_type_dict.keys():
            raise TypeError("The file type cannot be identified!")

        if data_file_root_dir is None:
            self.data_file_root_dir = prf.datahub_data_root_dir / 'Madrigal' / 'Millstone_ISR'
        else:
            self.data_file_root_dir = pathlib.Path(data_file_root_dir)
        self.done = False

        self.madrigal_url = "http://millstonehill.haystack.mit.edu/"
        self.download_madrigal_files()

    def download_madrigal_files(self, download_pp=False):
        icodes = [30, ]
        all_files=[]
        for icode in icodes:
            mylog.simpleinfo.info("Searching data from the Madrigal database ...")
            exp_list, _, database = madrigal.utilities.list_experiments(
                icode, self.dt_fr, self.dt_to, madrigal_url=self.madrigal_url
            )
                    
            if self.dry_run:
                mylog.simpleinfo.info("Dry run mode: check the online file information...")
                mylog.simpleinfo.info("Listing all of the online files:")
                mylog.simpleinfo.info("{:<40s}\t{:<20s}{:<30s}{:<80s}".format(
                    'Experiment Name', 'Experiment Label', 'File Name', 'File Type'
                    )
                )
                for exp in exp_list:
                    files = database.getExperimentFiles(exp.id)
                    for file in files:
                        all_files.append([exp, file])
                for exp, file in all_files:
                    exp_url = exp.url
                    res = re.search(r'mlh/([\d]{2}[a-z]{3}[\d]{2}[\w]*)', exp_url)
                    exp_label = res.groups()[0]
                    file_path_remote = pathlib.Path(file.name)
                    file_name = file_path_remote.name 
                    mylog.simpleinfo.info(
                        "{:<40.40s}\t{:<20.20s}{:<30.30s}{:<80.80s}".format(
                            exp.name,
                            exp_label,
                            file_name,
                            file.kindatdesc, 
                        ) 
                    )
                return 
            
            matches = ['usrp receiver', 'calibration', 'f2 peak density']
            num_exp = 0
            
            for exp in exp_list:
                exp_url = exp.url
                res = re.search(r'mlh/([\d]{2}[a-z]{3}[\d]{2}[\w]*)', exp_url)
                exp_label = res.groups()[0]
                if not str(self.exp_name_pattern):
                    if any(x in exp.name.lower() for x in matches):
                        continue
                elif self.exp_name_pattern not in exp.name.lower():
                    continue
                
                mylog.simpleinfo.info(f"Searching files from the experiment {exp.name} ...")
                
                num_exp = num_exp + 1
                files = database.getExperimentFiles(exp.id)
                num_file = 0
                for file in files:
                    if file_type_dict[self.file_type] not in file.kindatdesc.lower():
                        continue
                    
                    num_file = num_file + 1

                    file_path_remote = pathlib.Path(file.name)
                    file_name = file_path_remote.name
                    res = re.search(r'/([\d]{2}[a-z]{3}[\d]{2})', file.doi)
                    dtstr = res.groups()[0]
                    dtstr = dtstr[0:2] + dtstr[2].upper() + dtstr[3:]
                    thisday = datetime.datetime.strptime(dtstr, "%d%b%y")

                    exp_name_patterns = re.findall('\w+', exp.name)
                    if len(exp_name_patterns) > 5:
                        exp_name_patterns = exp_name_patterns[:5]
                    exp_name = '-'.join(exp_name_patterns).lower()
                    data_file_dir = self.data_file_root_dir / thisday.strftime("%Y") / thisday.strftime('%Y%m%d') / exp_name
                    data_file_dir.mkdir(parents=True, exist_ok=True)
                    file_name_new = 'MillstoneHill_ISR_' + self.file_type.replace(' ', '_') + '_' + thisday.strftime('%Y%m%d') + \
                                    '.' + file_name.split('.')[-2] + '.' + file_name.split('.')[-1]
                    data_file_path = data_file_dir / file_name_new
                    if data_file_path.is_file():
                        mylog.simpleinfo.info("The file {} has been downloaded.".format(data_file_path.name))
                        self.done = True
                        continue

                    mylog.simpleinfo.info("Downloading  {} from the Madrigal database ...".format(file_name))
                    database.downloadFile(
                        file.name, data_file_path,
                        self.user_fullname, self.user_email, self.user_affiliation,
                        "hdf5"
                    )
                    self.done = True
                    mylog.simpleinfo.info("Save the file as {} in the directory {}".format(file_name_new, data_file_dir))
                if num_file > 1:
                    mylog.StreamLogger.warning("Multiple files matched and downloaded! Check the file information, please!")
            if num_exp > 1:
                mylog.StreamLogger.warning("Multiple experiments matched! Check the experiment information, please!") 

if __name__ == "__main__":
    test()
