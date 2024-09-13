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


class Downloader(object):

    def __init__(self, dt_fr, dt_to, direct_download=True, force_download=True):

        self.url_base = "https://cdaweb.gsfc.nasa.gov/pub/data/"
        self.force_download = force_download

        self.dt_fr = dt_fr
        self.dt_to = dt_to
        self.source_file_paths = []

        if direct_download:
            self.download()

    def download(self):
        raise NotImplementedError

    def search_file(self, subdirs, file_name_patterns, logging=True):
        url = self.url_base + '/'.join(subdirs)

        r = requests.get(url)

        soup = bs4.BeautifulSoup(r.text, 'html.parser')
        a_tags = soup.find_all('a', href=True)
        hrefs = [a_tag['href'] for a_tag in a_tags]

        search_pattern = '.*' + '.*'.join(file_name_patterns) + '.*'
        fn_regex = re.compile(search_pattern)
        hrefs = list(filter(fn_regex.match, hrefs))

        if logging:
            if len(hrefs) == 0:
                mylog.StreamLogger.warning("Cannot find any files matching the file patterns!")
            elif len(hrefs) > 1:
                mylog.StreamLogger.warning("Multiple files matching the file patterns are found!")

        for href in hrefs:
            self.source_file_paths.append(url + href)
        return

    def save_file(self, file_dir, file_name, r_source_file):
        file_path = file_dir / file_name

        if file_path.is_file():
            mylog.simpleinfo.info(
                "The file {} exists in the directory {}.".format(file_path.name, file_path.parent.resolve()))

        if self.force_download:
            file_path.parent.resolve().mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                mylog.simpleinfo.info(
                    "Downloading {} to the directory {} ...".format(file_path.name, file_path.parent.resolve())
                )
                f.write(r_source_file.content)
                mylog.simpleinfo.info("Done")

