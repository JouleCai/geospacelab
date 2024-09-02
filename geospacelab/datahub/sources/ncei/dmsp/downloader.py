import pathlib
import bs4
import numpy as np
import re
import gzip
import requests

import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog


class Downloader(object):

    def __init__(
            self,
            direct_download=True,
            force_download=False,
    ):

        self._url_base = 'https://www.ncei.noaa.gov/data/dmsp-space-weather-sensors/access/'
        self.force_download = force_download
        self.done = False
        if direct_download:
            self.download()

    def download(self, url, file_dir=None, file_name=None):

        file_path_remote = pathlib.Path(url)
        file_name_remote = file_path_remote.name

        if file_name is None:
            if file_path_remote.suffix == '.gz':
                file_name = file_path_remote.with_suffix('').name
            else:
                file_name = file_name_remote

        file_path_local = file_dir / file_name
        if file_path_local.is_file() and not self.force_download:
            mylog.simpleinfo.info(
                "The file {} exists in {}!".format(file_path_local.name, file_dir)
            )
            return file_path_local

        mylog.simpleinfo.info("Contacting NCEI ...")
        content = self._get_content(url)

        if file_path_remote.suffix == '.gz':
            content = gzip.decompress(content)

        with open(file_path_local, 'wb') as f:
            f.write(content)
        mylog.simpleinfo.info("The file {} has been downloaded to {}.".format(
            file_path_local.name, file_dir
        ))
        return file_path_local

    @staticmethod
    def search_files(url, file_name_patterns=None):
        if file_name_patterns is None:
            file_name_patterns = []
        r = requests.get(url)
        r.raise_for_status()
        soup = bs4.BeautifulSoup(r.text, 'html.parser')
        links = soup.find_all('a', href=True)
        r.close()

        files = []
        search_pattern = '.*' + '.*'.join(file_name_patterns) + '.*'
        rc = re.compile(search_pattern)
        for link in links:
            href = link['href']
            if not any(href.endswith(s) for s in ['.gz', '.cdf', '.dat', '.txt', '.pdf', '.png']):
                continue

            file_name = href.split('/')[-1]
            res = rc.match(file_name)
            if res is None:
                continue
            file_url = url + '/' + href
            files.append(file_url)

        if not list(files):
            mylog.StreamLogger.warning("No files matching the reqeust from {}!".format(url))
            return None

        return files


    @staticmethod
    def _get_content(url):
        r = requests.get(url)
        r.raise_for_status()
        content = r.content
        r.close()
        return content

