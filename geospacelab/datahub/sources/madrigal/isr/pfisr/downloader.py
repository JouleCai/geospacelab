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
from geospacelab.datahub.sources.madrigal.downloader import Downloader as DownloaderBase


def test():
    dt_fr = datetime.datetime(2016, 3, 14)
    dt_to = datetime.datetime(2016, 3, 14, 23)
    eid = 100213152
    download_obj = Downloader(
        dt_fr, dt_to, include_exp_ids=[100213152, 100213190],
        data_product='vi',
        include_file_name_patterns= [['pfa']],
        include_file_type_patterns=[['velocity']],
        exclude_file_type_patterns=['.*uncorrected.*', '.*power.*'],
        force_download=True,
        dry_run=False)


class Downloader(DownloaderBase):

    def __init__(
            self, dt_fr: datetime.datetime, dt_to: datetime,
            data_product=None,
            include_exp_name_patterns: list=None,
            exclude_exp_name_patterns: list=None,
            include_exp_ids: list = None,
            exclude_exp_ids: list=[100213840, ],
            include_file_name_patterns: list = None,
            exclude_file_name_patterns: list = None,
            include_file_type_patterns=None,
            exclude_file_type_patterns=None,
            data_file_root_dir: str = None,
            direct_download = True,
            force_download = False,
            dry_run: bool=False,
            madrigal_url: str = "http://cedar.openmadrigal.org/",
            user_fullname: str=madrigal.default_user_fullname,
            user_email: str=madrigal.default_user_email,
            user_affiliation: str=madrigal.default_user_affiliation):

        icodes = [61,]
        self.data_product = data_product if isinstance(data_product, str) else ''
        
        dt_fr = dttool.get_start_of_the_day(dt_fr)
        dt_to = dttool.get_end_of_the_day(dt_to)

        super().__init__(
            dt_fr=dt_fr, dt_to=dt_to, icodes=icodes,
            include_exp_name_patterns=include_exp_name_patterns, 
            exclude_exp_name_patterns=exclude_exp_name_patterns, 
            include_exp_ids=include_exp_ids,
            exclude_exp_ids=exclude_exp_ids,
            include_file_name_patterns=include_file_name_patterns, 
            exclude_file_name_patterns=exclude_file_name_patterns,
            include_file_type_patterns=include_file_type_patterns, 
            exclude_file_type_patterns=exclude_file_type_patterns,  
            data_file_root_dir=data_file_root_dir,
            force_download=force_download, direct_download=direct_download, dry_run=dry_run,
            madrigal_url=madrigal_url,
            user_fullname=user_fullname, user_email=user_email, user_affiliation=user_affiliation)

    def download(self, **kwargs):

        exps, database = self.get_exp_list(
            dt_fr=self.dt_fr,
            dt_to=self.dt_to,
            include_exp_name_patterns=self.include_exp_name_patterns,
            exclude_exp_name_patterns=self.exclude_exp_name_patterns,
            include_exp_ids=self.include_exp_ids,
            exclude_exp_ids=self.exclude_exp_ids,
            icodes=self.icodes,
            madrigal_url=self.madrigal_url,
            display=True)
        self.exp_list = list(exps)
        self.database = database

        exps, exps_error = self.get_online_file_list(
            exp_list=self.exp_list, database=database,
            include_file_name_patterns=self.include_file_name_patterns,
            exclude_file_name_patterns=self.exclude_file_name_patterns,
            include_file_type_patterns=self.include_file_type_patterns,
            exclude_file_type_patterns=self.exclude_file_type_patterns,
            display=True
        )
        self.exp_list_error = list(exps_error)

        file_paths = [] 
        for exp in exps:
            for file in list(exp.files):

                file_path_remote = pathlib.Path(file.name)
                file_name_remote = file_path_remote.name

                res = re.search(r'/([\d]{2}[a-z]{3}[\d]{2})', file.doi)
                dtstr = res.groups()[0]
                dtstr = dtstr[0:2] + dtstr[2].upper() + dtstr[3:]
                thisday = datetime.datetime.strptime(dtstr, "%d%b%y")

                exp_name_patterns = re.findall('\w+', exp.name)
                if len(exp_name_patterns) > 5:
                    exp_name_patterns = exp_name_patterns[:5]
                exp_name = '_'.join(exp_name_patterns).lower()
                file_dir_local = self.data_file_root_dir / thisday.strftime("%Y") / thisday.strftime('%Y%m%d') / \
                                ('eid-' + str(exp.id) + '_' + exp_name)
                file_dir_local.mkdir(parents=True, exist_ok=True)
                file_name_local = 'PFISR_' + self.data_product.replace(' ', '_') + '_' + thisday.strftime(
                    '%Y%m%d') + '.' + '.'.join(file_name_remote.split('.')[1:])
                file_path_local = file_dir_local / file_name_local
                if file_path_local.is_file():
                    mylog.simpleinfo.info("The file has been downloaded: {}.".format(file_path_local))
                    self.done = True
                    continue
                super().download(
                    file_path_remote=file.name, file_path_local=file_path_local,
                    file_format='hdf5')
                file_paths.append(file_path_local)
        return file_paths


if __name__ == "__main__":
    test()
