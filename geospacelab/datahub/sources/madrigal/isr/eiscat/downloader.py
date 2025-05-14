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

from geospacelab.config import pref as pfr
import geospacelab.datahub.sources.madrigal as madrigal
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool

from geospacelab.datahub.sources.madrigal.downloader import Downloader as DownloaderBase


class Downloader(DownloaderBase):

    def __init__(
            self, 
            dt_fr: datetime.datetime, dt_to: datetime,
            antennas=None, 
            data_file_root_dir=None, 
            kind_data="madrigal",
            user_fullname=madrigal.default_user_fullname,
            user_email=madrigal.default_user_email,
            user_affiliation=madrigal.default_user_affiliation,
            include_exp_name_patterns: list=None,
            exclude_exp_name_patterns: list=None,
            include_exp_ids: list=None,
            exclude_exp_ids: list=None,
            include_file_name_patterns: list = None,
            exclude_file_name_patterns: list = None,
            include_file_type_patterns=None,
            exclude_file_type_patterns=None,
            direct_download = True,
            force_download = False,
            dry_run: bool=False,
            madrigal_url: str = "http://madrigal.eiscat.se/",
        ):

        # dt_fr = dttool.get_start_of_the_day(dt_fr)
        # dt_to = dttool.get_end_of_the_day(dt_to)
        
        icodes = []
        for a in antennas:
            icodes.extend(instrument_codes[a])  
        icodes = [int(i) for i in np.unique(icodes)]
        
        self.kind_data=kind_data

        if data_file_root_dir is None:
            self.data_file_root_dir = pfr.datahub_data_root_dir / 'Madrigal' / 'EISCAT' / 'Analyzed'
        else:
            self.data_file_root_dir = data_file_root_dir

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
        if self.kind_data.lower() == 'eiscat':
            self.download_EISCAT()
        elif self.kind_data.lower() == 'madrigal':
            self.download_MAD()
        else:
            raise NotImplementedError    
        return 
    
    def download_EISCAT(self):
        exps, database = self.get_exp_list(
            dt_fr=dttool.get_start_of_the_day(self.dt_fr),
            dt_to=dttool.get_end_of_the_day(self.dt_to),
            include_exp_name_patterns=self.include_exp_name_patterns,
            exclude_exp_name_patterns=self.exclude_exp_name_patterns,
            include_exp_ids=self.include_exp_ids,
            exclude_exp_ids=self.exclude_exp_ids,
            icodes=self.icodes,
            madrigal_url=self.madrigal_url,
            display=True)
        self.exp_list = list(exps)
        self.database = database
        
        cookies = {
                'user_email': self.user_email,
                'user_fullname': self.user_fullname,
                'user_affiliation': self.user_affiliation
            }
        for exp in exps:
            dt_fr_exp = datetime.datetime(
                exp.startyear, exp.startmonth, exp.startday, exp.starthour, exp.startmin, exp.startsec
            )
            dt_to_exp = datetime.datetime(
                exp.endyear, exp.endmonth, exp.endday, exp.endhour, exp.endmin, exp.endsec
            )
            if (dt_fr_exp >= self.dt_to) or (dt_to_exp <= self.dt_fr):
                continue
            try:
                res = re.search(r'([\d]{4}\-[\d]{2}\-[\d]{2})_(\w+)@(\w+)', exp.name)
                thisday = datetime.datetime.strptime(res.groups()[0], "%Y-%m-%d")
                pulse_code = res.groups()[1]
                antenna_ = res.groups()[2]
                if 'uhf' in antenna_.lower():
                    antenna = 'UHF'
                elif 'vhf' in antenna_.lower():
                    antenna = 'VHF'
                elif any([a in antenna_.lower() for a in ['32m', '42m', 'esr']]):
                    antenna = 'ESR'
                elif 'sod' in antenna_.lower():
                    antenna = 'SOD'
                elif 'kir' in antenna_.lower():
                    antenna = 'KIR'
                else:
                    raise NotImplementedError
            except Exception as e:
                print(e)
                mylog.StreamLogger.warning("Parsing the experiment name was failed! EXP: {}".format(exp.name))
            
            url = "https://madrigal.eiscat.se/madrigal/showExperiment?experiment_list=" + str(exp.id) + "&show_plots="
            
            r = requests.get(url, cookies=cookies)
            soup = bs4.BeautifulSoup(r.text, 'html.parser')
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                if any(href.endswith(s) for s in ['.png', '.tar.gz', '.hdf5']):
                    filename = href.split('/')[-1]
                    file_dir_local = self.data_file_root_dir / antenna / thisday.strftime("%Y") / \
                                (exp.name + '_EID-' + str(exp.id) + '_'
                                 + dt_fr_exp.strftime("%Y%m%dT%H%M%S") + '_'
                                 + dt_to_exp.strftime("%Y%m%dT%H%M%S"))
                    file_dir_local.mkdir(parents=True, exist_ok=True)

                    remote_file = requests.get(href)
                    file_path = file_dir_local / filename
                    if file_path.is_file() and not self.force_download:
                        print("The file {} has been downloaded.".format(filename))
                        self.done=True
                        continue
                    mylog.simpleinfo.info(
                        'Downloading "{} ..."'.format(filename)
                    )
                    with open(file_path, "wb") as eiscat:
                        eiscat.write(remote_file.content)
                    mylog.simpleinfo.info('Saved to {}.'.format(file_dir_local))
                    self.done = True
        return 
    
    def download_MAD(self, **kwargs):

        exps, database = self.get_exp_list(
            dt_fr=dttool.get_start_of_the_day(self.dt_fr),
            dt_to=dttool.get_end_of_the_day(self.dt_to),
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
            dt_fr_exp = datetime.datetime(
                exp.startyear, exp.startmonth, exp.startday, exp.starthour, exp.startmin, exp.startsec
            )
            dt_to_exp = datetime.datetime(
                exp.endyear, exp.endmonth, exp.endday, exp.endhour, exp.endmin, exp.endsec
            )
            # if (dt_fr_exp >= self.dt_to) or (dt_to_exp <= self.dt_fr):
            #     continue
            for file in list(exp.files):

                file_path_remote = pathlib.Path(file.name)
                file_name_remote = file_path_remote.name

                try:
                    res = re.search(r'([\d]{4}\-[\d]{2}\-[\d]{2})_(\w+)@(\w+)', exp.name)
                    thisday = datetime.datetime.strptime(res.groups()[0], "%Y-%m-%d")
                    pulse_code = res.groups()[1]
                    antenna_ = res.groups()[2]
                    if 'uhf' in antenna_.lower():
                        antenna = 'UHF'
                    elif 'vhf' in antenna_.lower():
                        antenna = 'VHF'
                    elif any([a in antenna_.lower() for a in ['32m', '42m', 'esr']]):
                        antenna = 'ESR'
                    elif 'sod' in antenna_.lower():
                        antenna = 'SOD'
                    elif 'kir' in antenna_.lower():
                        antenna = 'KIR'
                    else:
                        raise NotImplementedError
                except Exception as e:
                    print(e)
                    mylog.StreamLogger.warning("Parsing the experiment name was failed! EXP: {}".format(exp.name))

                file_dir_local = self.data_file_root_dir / antenna / thisday.strftime("%Y") / \
                                (exp.name + '_EID-' + str(exp.id) + '_'
                                 + dt_fr_exp.strftime("%Y%m%dT%H%M%S") + '_'
                                 + dt_to_exp.strftime("%Y%m%dT%H%M%S"))
                file_dir_local.mkdir(parents=True, exist_ok=True)

                file_name_local = file_name_remote
                file_path_local = file_dir_local / file_name_local

                super().download(
                    file_path_remote=file.name, file_path_local=file_path_local,
                    file_format='hdf5')
                file_paths.append(file_path_local)
        return file_paths


def test():
    dt_0 = datetime.datetime(2016, 3, 9, 19)
    dt_1 = datetime.datetime(2016, 3, 9, 23, 59)
    downloader = DownloaderNew(
        dt_fr=dt_0, dt_to=dt_1, 
        kind_data='EISCAT',
        antennas=['UHF', 'VHF'],
        force_download=True,
        exclude_file_type_patterns=['pp']
        )
    pass

class DownloaderOld(object):
    """Download the quickplots and archieved analyzed results from EISCAT schedule webpage
    """

    def __init__(self, dt_fr, dt_to, sites=None, data_file_root_dir=None, kind_data="EISCAT", download_pp=False,
                 user_fullname=madrigal.default_user_fullname,
                 user_email=madrigal.default_user_email,
                 user_affiliation=madrigal.default_user_affiliation):
        self.user_fullname = user_fullname
        self.user_email = user_email
        self.user_affiliation = user_affiliation

        dt_fr = dttool.get_start_of_the_day(dt_fr)
        dt_to = dttool.get_end_of_the_day(dt_to)
        if dt_fr == dt_to:
            dt_to = dt_to + datetime.timedelta(hours=23, minutes=59)
        self.dt_fr = dt_fr  # datetime from
        self.dt_to = dt_to  # datetime to
        if sites is None:
            sites = ['UHF', 'VHF', 'KIR', 'SOD',
                     'ESR']  # site list: default is the list of all the sites, use e.g., ['UHF'] for single site
        self.sites = sites

        if data_file_root_dir is None:
            self.data_file_root_dir = pfr.datahub_data_root_dir / 'Madrigal' / 'EISCAT' / 'analyzed'
        else:
            self.data_file_root_dir = data_file_root_dir
        self.done = False

        if kind_data.lower() == "madrigal":
            self.madrigal_url = "https://madrigal.eiscat.se/madrigal/"
            self.download_madrigal_files(download_pp=download_pp)
        elif kind_data.lower() == "eiscat":
            self.url_base = 'http://portal.eiscat.se'  # EISCAT schedule webpage
            self.urls = []
            self.search_scheduled_url()
            self.download_eiscat_files()

    def search_scheduled_url(self):
        """
        Search available urls
        :return: None
        """
        diff_month = (self.dt_to.year - self.dt_fr.year) * 12 + self.dt_to.month - self.dt_fr.month
        urls = set()
        for nm in range(diff_month + 1):
            dt_now = datetime.datetime(self.dt_fr.year, self.dt_fr.month, 1) + relativedelta(months=nm)

            payload_options = []
            for site in self.sites:
                payload_options.append(site)
                if site == 'UHF':
                    payload_options.append('TRO')  # full list: VHF=on&UHF=on&TRO=on&KIR=on&SOD=on&ESR=on&HEA=on
            payload = {
                'year': "{:04d}".format(dt_now.year),
                'month': '{:d}'.format(dt_now.month),
                'A': 'on'
            }
            for opt in payload_options:
                payload[opt] = 'on'

            r = requests.get(self.url_base + '/schedule/schedule.cgi', params=payload)
            soup = bs4.BeautifulSoup(r.text, 'html.parser')

            strong_tags = soup.find_all('strong')

            for stag in strong_tags:
                yymm = dt_now.strftime("%Y:%m")
                if yymm not in stag.string:
                    continue
                tagline = stag.sourceline
                # print(stag.string)
                content = r.text.split('\n')[tagline - 1]
                # print(content)
                soup_line = bs4.BeautifulSoup(content, 'html.parser')
                link = soup_line.find('a', href=True)['href']
                if 'experiment_list' not in link:
                    print('No link for the existing experiment ({})'.format(link))
                    continue
                urls.add(link)
            self.urls = urls
        return None

    def download_eiscat_files(self):
        """
        Download the files
        """
        if not list(self.urls):
            mylog.StreamLogger.info("No experiments available!")
            return None

        for url in self.urls:
            cookies = {
                'user_email': self.user_email,
                'user_fullname': self.user_fullname,
                'user_affiliation': self.user_affiliation
            }
            r = requests.get(url, cookies=cookies)
            soup = bs4.BeautifulSoup(r.text, 'html.parser')
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                if any(href.endswith(s) for s in ['.png', '.tar.gz', '.hdf5']):
                    filename = href.split('/')[-1]

                    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
                    thisday = datetime.datetime.strptime(match.group(), '%Y-%m-%d')
                    if thisday < self.dt_fr or thisday > self.dt_to:
                        continue
                    for key, value in locs.items():
                        if '/' + value + '/' in href:
                            site = key

                    #site1 = re.search("[@|_][a-zA-Z0-9]*[.]", filename).group(0)[1:-1]
                    search_pattern = re.search(r"\d{4}-\d{2}-\d{2}_[a-zA-Z0-9]*", filename).group(0)
                    sub_dir = search_pattern + '@' + site
                    data_file_dir = self.data_file_root_dir / site / thisday.strftime('%Y') / sub_dir
                    data_file_dir.mkdir(parents=True, exist_ok=True)

                    remote_file = requests.get(href)
                    file_path = data_file_dir / filename
                    if file_path.is_file():
                        print("The file {} has been downloaded.".format(filename))
                        continue
                    mylog.simpleinfo.info(
                        'Downloading "{} from portal.eiscat.se to {} ..."'.format(filename, data_file_dir)
                    )
                    with open(file_path, "wb") as eiscat:
                        eiscat.write(remote_file.content)
                    mylog.simpleinfo.info('Done!')
                    self.done = True
        return

    def download_madrigal_files(self, download_pp=False):
        icodes = []
        for site in self.sites:
            icodes.extend(instrument_codes[site])
        for icode in icodes:
            exp_list, _, database = madrigal.utilities.list_experiments(
                icode, self.dt_fr, self.dt_to, madrigal_url=self.madrigal_url)
            for exp in exp_list:
                files = database.getExperimentFiles(exp.id)
                for file in files:
                    if not download_pp and 'GUISDAP pp' in file.kindatdesc:
                        continue
                    file_path = pathlib.Path(file.name)
                    site = file_path.name.split("@")[1][0:3].upper()
                    if '32' in site or '42' in site:
                        site = 'ESR'

                    match = re.search(r'\d{4}-\d{2}-\d{2}', file_path.name)
                    dt_str = match.group(0)
                    thisday = datetime.datetime.strptime(dt_str, "%Y-%m-%d")
                    if thisday < self.dt_fr or thisday > self.dt_to:
                        continue

                    # sub_dir = file_path.name.split('_', maxsplit=1)[1]
                    search_pattern = re.search(r"\d{4}-\d{2}-\d{2}_[a-zA-Z0-9]*", file_path.name).group(0)
                    sub_dir = search_pattern + '@' + site
                    data_file_dir = self.data_file_root_dir / site / dt_str[0:4] / sub_dir
                    data_file_dir.mkdir(parents=True, exist_ok=True)
                    data_file_path = data_file_dir / file_path.name
                    if data_file_path.is_file():
                        mylog.simpleinfo.info("The file {} has been downloaded.".format(data_file_path.name))
                        continue

                    mylog.simpleinfo.info("Downloading  {} from the Madrigal database ...".format(file_path.name))
                    database.downloadFile(
                        file.name, data_file_path,
                        self.user_fullname, self.user_email, self.user_affiliation,
                        "hdf5"
                    )
                    self.done = True
                    mylog.simpleinfo.info("Done!")


        # fhdf5 = h5py.File(outDir + fn, 'r')


instrument_codes = {
    'UHF': [72],
    'VHF': [74],
    '42m': [95],
    '32m': [95],
    'SOD': [73, 76],
    'KIR': [71, 75]
}

locs = {
    'UHF': 'tro',
    'VHF': 'eis',
    'ESR': 'lyr',
    'SOD': 'sod',
    'KIR': 'kir'
}


class EISCATSchedule(object):
    """ Analyze EISCAT schedule
    """

    def __init__(self, dt_fr, dt_to, sites=None, root_filepath=None, monthly=False):
        self.url_base = 'http://portal.eiscat.se'  # EISCAT schedule webpage
        self.dt_fr = dt_fr
        self.dt_to = dt_to  # datetime
        if sites is None:
            sites = ['UHF', 'VHF', 'KIR', 'SOD',
                     'ESR']  # site list: default is the list of all the sites, use e.g., ['UHF'] for single site
        self.sites = sites
        self.rs = []
        self.experiments = []

        self.urls = []
        if root_filepath is None:
            root_filepath = pfr.datahub_data_root_dir  # root path for the download files. Default: the current work directory + /results/
        self.root_filepath = root_filepath
        self.monthly = monthly  # True: download the whole month data. False: only for the specific date
        self.search_archives()
        self.analyze_archives()
        self.to_txt()

    def search_archives(self):
        """
        Search available urls
        :return: None
        """

        diff_month = (self.dt_to.year - self.dt_fr.year) * 12 + self.dt_to.month - self.dt_fr.month
        for nm in range(diff_month + 1):
            dt_now = datetime.datetime(self.dt_fr.year, self.dt_fr.month, 1) + relativedelta(months=nm)

            payload_options = []
            for site in self.sites:
                payload_options.append(site)
                if site == 'UHF':
                    payload_options.append('TRO')  # full list: VHF=on&UHF=on&TRO=on&KIR=on&SOD=on&ESR=on&HEA=on
            payload = {
                'year': "{:04d}".format(dt_now.year),
                'month': '{:d}'.format(dt_now.month),
                'A': 'on'
            }
            for opt in payload_options:
                payload[opt] = 'on'

            r = requests.get(self.url_base + '/schedule/schedule.cgi', params=payload)

            self.rs.append(r)

    def analyze_archives(self):
        nexp = 0
        for r in self.rs:
            soup = bs4.BeautifulSoup(r.text, 'html.parser')
            schedule = soup.pre.text.split('\n')
            for item_text in schedule:
                if not is_date(item_text[0:10].replace(':', '-')):
                    continue
                dt_now = datetime.datetime.strptime(item_text[0:10], '%Y:%m:%d')
                if not self.monthly:
                    if dt_now < self.dt_fr or dt_now > self.dt_to:
                        continue

                time_rec = item_text[15:63]

                inds = [x for x, v in enumerate(time_rec) if v == 'A']
                if not list(inds):
                    continue

                info = item_text[65:-1].split(' ')
                # print(dt_fr)
                # print(info)
                antenna = info[0]
                any_pattern = False
                for site in self.sites:
                    if site.lower() in antenna.lower():
                        any_pattern = True
                    if site == 'ESR':
                        if ('32' in antenna) or ('42' in antenna):
                            any_pattern = True
                if not any_pattern:
                    continue

                exp = info[1]
                info = item_text[65:-1].split(')')[-1].strip()

                info_splits = info.split('_')
                code = info_splits[0]
                mode = ''
                version = ''
                if len(info_splits) > 1:
                    mode = info_splits[1]
                if len(info_splits) > 2:
                    version = info_splits[2]

                # if not (('ip2' in mode.lower()) or 'cp2' in mode.lower()):
                #    continue
                # try:
                #     code = info.split('_')[0]
                #     mode = info.split('_')[1]
                #     version = info.split('_')[2]
                # except IndexError:
                #     print('Not a standard experiment {}'.format(info))
                #     continue

                ind_diff = np.diff(inds)
                ind_gaps = np.where(ind_diff > 3)[0]  # time difference greater than 1.5 h
                ind_ranges = []
                for ind_, ind_gap in enumerate(ind_gaps):
                    if ind_ == 0:
                        ind_fr = inds[0]
                        ind_to = inds[ind_gap]
                    else:
                        ind_fr = inds[ind_gaps[ind_ - 1] + 1]
                        ind_to = inds[ind_gap]
                    ind_ranges.append([ind_fr, ind_to])

                if list(ind_gaps):
                    ind_ranges.append([inds[ind_gaps[-1] + 1], inds[-1]])
                else:
                    ind_ranges.append([inds[0], inds[-1]])

                for ind_range in ind_ranges:
                    ind_1 = ind_range[0]
                    ind_2 = ind_range[1]
                    dt_fr = dt_now + datetime.timedelta(hours=ind_1 / 2)
                    HH = (ind_2 + 1) / 2
                    if HH >= 24:
                        HH = 23
                        MM = 59
                    else:
                        MM = 0
                    dt_to = dt_now + datetime.timedelta(hours=HH, minutes=MM)
                    suffix = 'OK'
                    if time_rec.count('A') < (ind_2 - ind_1 + 1):
                        suffix = '?'

                    self.experiments.append([dt_fr, dt_to, antenna, code, mode, version, exp, suffix, time_rec])
                    nexp = nexp + 1

        if nexp == 0:
            print('No available archived experiments!')
            return False

        return True

    def to_txt(self, file_name=None, file_path=None):
        if file_name is None:
            file_name = "EISCAT_archives.txt"
        if file_path is None:
            file_path = self.root_filepath

        with open(os.path.join(file_path, file_name), 'w') as out_file:
            for line in self.experiments:
                line_str = "{0:<17s}{1:<7s}{2:<5s}{3:<8s}{4:<30s}{5:<40s}{6:<40s}{7:<80s}{8:<65s}".format(
                    line[0].strftime('%Y-%m-%d %H:%M'), line[1].strftime('%H:%M'),
                    line[7], line[2], line[3], line[4], line[5], line[6], line[8]) + '\n'
                out_file.write(line_str)


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        dtparse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


if __name__ == "__main__":
    test()
