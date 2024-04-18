# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"


import time
import datetime
import re
import pathlib
import madrigalWeb.madrigalWeb as madrigalweb
import numpy as np
import copy


from geospacelab.config import prf
import geospacelab.datahub.sources.madrigal as madrigal
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool


DEFAULT_MADRIGAL_URL = "http://cedar.openmadrigal.org/"


class Downloader(object):

    def __init__(
            self, dt_fr: datetime.datetime, dt_to: datetime,
            icodes: list,
            include_exp_name_patterns: list=None,
            exclude_exp_name_patterns: list=None,
            include_exp_ids: list = None,
            exclude_exp_ids: list= None,
            include_file_name_patterns: list = None,
            exclude_file_name_patterns: list = None,
            include_file_type_patterns=None,
            exclude_file_type_patterns=None,
            data_file_root_dir: str = None,
            direct_download = False,
            force_download = False,
            dry_run: bool=False,
            madrigal_url: str = DEFAULT_MADRIGAL_URL,
            user_fullname: str=madrigal.default_user_fullname,
            user_email: str=madrigal.default_user_email,
            user_affiliation: str=madrigal.default_user_affiliation):
        """

        :param dt_fr: Starting time.
        :type dt_fr: datetime.datetime
        :param dt_to: Stopping time.
        :type dt_to: datetime.datetime
        :param data_file_root_dir: The root directory for the data.
        :type data_file_root_dir: str  or pathlib.Path
        :param user_fullname:
        :param user_email:
        :param user_affiliation:
        """

        self.user_fullname = user_fullname
        self.user_email = user_email
        self.user_affiliation = user_affiliation

        self.dt_fr = dt_fr
        self.dt_to = dt_to

        self.icodes = icodes
        self.madrigal_url = madrigal_url

        self.dry_run = dry_run
        self.include_exp_ids = include_exp_ids if include_exp_ids is not None else []
        self.exclude_exp_ids = exclude_exp_ids if exclude_exp_ids is not None else []
        self.include_exp_name_patterns = include_exp_name_patterns \
            if include_exp_name_patterns is not None else []
        self.exclude_exp_name_patterns = exclude_exp_name_patterns \
            if exclude_exp_name_patterns is not None else []
        self.include_file_name_patterns = include_file_name_patterns if include_file_name_patterns is not None else []
        self.exclude_file_name_patterns = exclude_file_name_patterns if exclude_file_name_patterns is not None else []
        self.include_file_type_patterns = include_file_type_patterns if include_file_type_patterns is not None else []
        self.exclude_file_type_patterns = exclude_file_type_patterns if exclude_file_type_patterns is not None else []

        self.force_download = force_download
        
        self.data_file_root_dir = pathlib.Path(data_file_root_dir) if isinstance(data_file_root_dir, str) else data_file_root_dir
        if self.data_file_root_dir is None:
            self.data_file_root_dir = pathlib.Path().absolute()
        
        self.exp_list = []
        self.exp_list_error = []
        self.database = None
        
        self.data_file_paths = []

        self.done = False
        
        if dry_run:
            self.show_all_experiment_and_files()
            return
        if direct_download:
            self.download()

    def show_all_experiment_and_files(self):
        exps, database = self.get_exp_list(
            exp_ids=[], exp_name_patterns=[], display=False)
        exps = self.get_online_file_list(exp_list=exps, display=True)
        return

    def download(self, database=None, file_path_remote=None, file_path_local=None, file_format='hdf5'):
        database = self.database if database is None else database
        self.data_file_paths.append(file_path_local)
        
        if file_path_local.is_file():
            mylog.simpleinfo.info("The file {} has been downloaded.".format(file_path_local.name))
            if not self.force_download:
                return
        files_error = []
        mylog.simpleinfo.info("Downloading {} ...".format(file_path_remote))
        try:
            database.downloadFile(
                file_path_remote, file_path_local,
                self.user_fullname, self.user_email, self.user_affiliation,
                file_format
            )

            mylog.simpleinfo.info("--> Saved as {}.".format(file_path_local))
            self.done = True
        except Exception as e:
            try:
                time.sleep(3)
                database.downloadFile(
                    file_path_remote, file_path_local,
                    self.user_fullname, self.user_email, self.user_affiliation,
                    file_format
                )

                mylog.simpleinfo.info("--> Saved as {}.".format(file_path_local))
                self.done = True 
            except:
                print(e)
                mylog.StreamLogger.warning(f"Failed to download the file: {file_path_remote}")
        return
    
    @staticmethod
    def get_online_file_list(
            exp_list=None,
            include_file_name_patterns=None,
            exclude_file_name_patterns=None,
            include_file_type_patterns=None,
            exclude_file_type_patterns=None,
            database=None, display=False):

        def try_to_get_experiment_files(max=3, interval=10):
            for m in range(max):
                try: 
                    files = database.getExperimentFiles(exp.id)
                    return files
                except Exception as e:
                    if m < max - 1:
                        time.sleep(interval)
                    continue
            print(e)
            mylog.StreamLogger.warning(f"Failed to get experiment files with {max} connection(s)!")
            return -1
        
        include_file_name_patterns = [] if include_file_name_patterns is None else include_file_name_patterns
        exclude_file_name_patterns = [] if exclude_file_name_patterns is None else exclude_file_name_patterns
        include_file_type_patterns = [] if include_file_type_patterns is None else include_file_type_patterns
        exclude_file_type_patterns = [] if exclude_file_type_patterns is None else exclude_file_type_patterns

        exps_new = []
        exps_error = []
        mylog.simpleinfo.info("Searching files ...")
        for exp in exp_list:
            time.sleep(1)
            mylog.simpleinfo.info(f"Checking the experiment: {exp.name} (ID: {exp.id})")
            
            files = try_to_get_experiment_files(max=3, interval=10)
            
            if files == -1:
                mylog.StreamLogger.warning(
                    "Error when querying files: {} (ID: {}) in {}".format(exp.name, exp.id, exp.url)
                )
                exps_error.append(exp)
                continue

            if list(include_file_name_patterns):
                files_new = []
                for file in files:
                    matching = 0
                    for fnp in include_file_name_patterns:
                        if isinstance(fnp, list):
                            fnp = '.*' + '.*'.join(fnp) + '.*'
                        rc = re.compile(fnp)
                        file_name = pathlib.Path(file.name).name
                        rm = rc.match(file_name.lower())
                        if rm is not None:
                            matching = 1
                    if matching == 1:
                        files_new.append(file)
                if not list(files_new):
                    mylog.StreamLogger.warning(
                        f"No files matching the file name patterns! Experiment: {exp.name} (ID: {exp.id}).")
                    continue
                files = np.array(files_new)

            if list(exclude_file_name_patterns):
                files_new = []
                for file in files:
                    matching = 1
                    for fnp in exclude_file_name_patterns:
                        if isinstance(fnp, list):
                            fnp = '.*' + '.*'.join(fnp) + '.*'
                        rc = re.compile(fnp)
                        file_name = pathlib.Path(file.name).name
                        rm = rc.match(file_name.lower())
                        if rm is not None:
                            matching = 0
                    if matching == 1:
                        files_new.append(file)
                if not list(files_new):
                    mylog.StreamLogger.warning(
                        f"All files are excluded with the file name patterns! Experiment: {exp.name} (ID: {exp.id}).")
                    continue
                files = np.array(files_new)

            if list(include_file_type_patterns):
                files_new = []
                for file in files:
                    matching = 0
                    for fnp in include_file_type_patterns:
                        if isinstance(fnp, list):
                            fnp = '.*' + '.*'.join(fnp) + '.*'
                        rc = re.compile(fnp)
                        rm = rc.match(file.kindatdesc.lower())
                        if rm is not None:
                            matching = 1
                    if matching == 1:
                        files_new.append(file)
                if not list(files_new):
                    mylog.StreamLogger.warning(
                        f"No files matching the file type patterns! Experiment: {exp.name} (ID: {exp.id}).")
                    continue
                files = np.array(files_new)

            if list(exclude_file_type_patterns):
                files_new = []
                for file in files:
                    matching = 1
                    for fnp in exclude_file_type_patterns:
                        if isinstance(fnp, list):
                            fnp = '.*' + '.*'.join(fnp) + '.*'
                        rc = re.compile(fnp)
                        rm = rc.match(file.kindatdesc.lower())
                        if rm is not None:
                            matching = 0
                    if matching == 1:
                        files_new.append(file)
                if not list(files_new):
                    mylog.StreamLogger.warning(
                        f"All files are excluded with the file type patterns! Experiment: {exp.name} (ID: {exp.id}).")
                    continue
                files = np.array(files_new)

            exp.files = files
            exps_new.append(exp)

            mylog.simpleinfo.info('Listing matched files ...')
            for file in files:
                mylog.simpleinfo.info(file.name)

        if not list(exps_new):
            mylog.StreamLogger.warning(f"No experiments have the matched files!")
            return
        exps = np.array(exps_new)


        if display:
            mylog.simpleinfo.info("Listing matched experiments and files ...")
            exp_info = Downloader.get_exp_info(exps, include_file_info=True)
            mylog.simpleinfo.info("{:>10s}\t{:<24s}\t{:<24s}\t{:<16s}\t{:<15s}\t{:<40.40s}\t{:<30.30s}\t{:<80.80s}".format(
                'EXP NUM', 'START TIME', 'END TIME', 'DURATION (hour)', 'EXP ID', 'EXP Name', 'File Name', 'File Type'
            )
            )
            for ind, (exp_id, name, dt_fr, dt_to, duration) in enumerate(
                    zip(
                        exp_info['ID'], exp_info['NAME'], exp_info['DATETIME_0'], exp_info['DATETIME_1'],
                        exp_info['DURATION']
                    )
            ):
                
                for file in exp_info['FILES'][ind]:
                    line_str = "{:>10d}\t{:<24s}\t{:<24s}\t{:<16.1f}\t{:<15d}\t{:<40.40s}\t{:<30.30s}\t{:<80.80s}".format(
                        ind + 1,
                        dt_fr.strftime("%Y-%m-%d %H:%M:%S"),
                        dt_to.strftime("%Y-%m-%d %H:%M:%S"),
                        duration,
                        exp_id,
                        name,
                        file['NAME'],
                        file['DESC']
                    )
                    mylog.simpleinfo.info(line_str)
            mylog.simpleinfo.info("")
        return exps, exps_error

    @staticmethod
    def get_exp_list(
            dt_fr=None, dt_to=None,
            include_exp_name_patterns=None,
            exclude_exp_name_patterns=None,
            include_exp_ids=None,
            exclude_exp_ids=None,
            icodes=None, madrigal_url=None, display=True,
    ): 
        
        def try_to_get_database(max=3, interval=30):
            for m in range(max):
                try: 
                    database = madrigalweb.MadrigalData(madrigal_url)
                    return database
                except Exception as e:
                    if m < max - 1:
                        time.sleep(interval)
                    continue
            
            print(e)
            raise ImportError(f"Failed to connect the Madrigal database with {max} connection(s)!")
        
        def try_to_get_experiments(max=3, interval=30):
            for m in range(max):
                try: 
                    exps_o = database.getExperiments(
                        icode,
                        dt_fr.year, dt_fr.month, dt_fr.day, dt_fr.hour, dt_fr.minute, dt_fr.second,
                        dt_to.year, dt_to.month, dt_to.day, dt_to.hour, dt_to.minute, dt_to.second,
                        local=0
                    )
                    return exps_o
                except Exception as e:
                    if m < max - 1:
                        time.sleep(interval)
                    continue
            
            print(e)
            raise ImportError(f"Failed to get the experiments from the database with {max} connection(s)!") 
        
        include_exp_name_patterns = [] if include_exp_name_patterns is None else include_exp_name_patterns
        include_exp_ids = [] if include_exp_ids is None else include_exp_ids
        icodes = [] if icodes is None else icodes
        madrigal_url = DEFAULT_MADRIGAL_URL if madrigal_url is None else madrigal_url
        exclude_exp_ids = [] if exclude_exp_ids is None else exclude_exp_ids
        
        exps = []
        mylog.simpleinfo.info(f"Contacting the Madrigal database (URL: {madrigal_url}) ...")
        database = try_to_get_database(max=3, interval=30) 
            
        for icode in icodes:
            mylog.simpleinfo.info("Searching experiments ...")
            exps_o = try_to_get_experiments(max=3, interval=30)
            exps.extend(exps_o)
        exps = np.array(exps, dtype=object)

        exps_new = []
        another_madrigal_url = ''
        for exp in exps:
            if exp.id == -1:
                if another_madrigal_url != exp.madrigalUrl:
                    mylog.StreamLogger.warning(
                        f'Another Madrigal site detected: {exp.madrigalUrl}!'
                    )
                    another_madrigal_url = exp.madrigalUrl
                continue
            if exp.id in exclude_exp_ids:
                mylog.StreamLogger.warning(
                    f'The following experiment is excluded: {exp.name} (ID: {exp.id})!'
                )
                continue
            exps_new.append(exp)
  
        exps = np.array(exps_new, dtype=object)

        if not list(exps):
            raise ValueError('Cannot find available experiments from the current database! Check the input values!')
        elif str(another_madrigal_url):
            mylog.StreamLogger.warning(
                'Some data are located in another Madrigal site and will not be processed!'
            )
        else:
            pass
        
        dts_fr = np.array([
                datetime.datetime(
                    exp.startyear, exp.startmonth, exp.startday, exp.starthour, exp.startmin, exp.startsec)
                for exp in exps
            ])
        dts_to = np.array([
            datetime.datetime(
                exp.endyear, exp.endmonth, exp.endday, exp.endhour, exp.endmin, exp.endsec)
            for exp in exps
        ])

        eids = np.array([exp.id for exp in exps])

        if list(include_exp_ids):
            exps_new = []
            for exp_id in include_exp_ids:
                ind = np.where(eids == exp_id)[0]

                if list(ind):
                    exps_new.append(exps[ind[0]])
                else:
                    mylog.StreamLogger.warning(
                        f'The requested experiment (ID: {exp_id}) cannot be found!'
                    )
            exps = np.array(exps_new, dtype=object)
            # inds_o = np.array(eids).argsort()
            # inds = inds_o[np.searchsorted(eids[inds_o], include_exp_ids)]
            # if not list(inds):
            #     mylog.StreamLogger.error("Cannot find available experiments for the input experiment IDs!")
            #     raise AttributeError
            # exps = exps[inds]

        else:
            ind_dt_no = np.where(
                (dt_fr > dts_to) | (dt_to < dts_fr))[0]
            exps = exps[[i not in ind_dt_no for i in range(eids.size)]]
            if not list(exps):
                mylog.StreamLogger.error("No experiments matching the time range!")
                raise AttributeError

            if list(include_exp_name_patterns):
                exps_new = []
                for exp in exps:
                    matching = 0
                    for enp in include_exp_name_patterns:
                        if isinstance(enp, list):
                            enp = '.*' + '.*'.join(enp) + '.*'
                        rc = re.compile(enp)
                        rm = rc.match(exp.name.lower())
                        if rm is not None:
                            matching = 1
                    if matching == 1:
                        exps_new.append(exp)
                if not list(exps_new):
                    mylog.StreamLogger.warning(
                        f"No experiments matching the exp name patterns!")
                    raise AttributeError
                exps = np.array(exps_new, dtype=object)

            if list(exclude_exp_name_patterns):
                exps_new = []
                for exp in exps:
                    matching = 1
                    for enp in exclude_exp_name_patterns:
                        if isinstance(enp, list):
                            enp = '.*' + '.*'.join(enp) + '.*'
                        rc = re.compile(enp)
                        rm = rc.match(exp.name.lower())
                        if rm is not None:
                            matching = 0
                    if matching == 1:
                        exps_new.append(exp)
                if not list(exps_new):
                    mylog.StreamLogger.warning(
                        f"All experiments are excluded with the exp name patterns!")
                    raise AttributeError
                exps = np.array(exps_new, dtype=object)

            if not list(exps):
                mylog.StreamLogger.error("Cannot find available experiments for the input experiment name patterns!")
                raise AttributeError

        if display:
            mylog.simpleinfo.info("Listing matched experiments ...")
            exp_info = Downloader.get_exp_info(exps)
            mylog.simpleinfo.info(
                "{:>10s}\t{:<24s}\t{:<24s}\t{:<16s}\t{:<15s}\t{:<40.40s}\t{:<s}".format(
                'EXP NUM', 'START TIME', 'END TIME', 'DURATION (hour)', 'EXP ID', 'EXP Name', 'EXP_URL'
                )
            )
            for ind, (exp_id, name, dt_fr, dt_to, duration, url) in enumerate(
                    zip(
                        exp_info['ID'], exp_info['NAME'], exp_info['DATETIME_0'], exp_info['DATETIME_1'],
                        exp_info['DURATION'], exp_info['URL']
                    )
            ):
                line_str = "{:>10d}\t{:<24s}\t{:<24s}\t{:<16.1f}\t{:<15d}\t{:<40.40s}\t{:<s}".format(
                    ind+1,
                    dt_fr.strftime("%Y-%m-%d %H:%M:%S"),
                    dt_to.strftime("%Y-%m-%d %H:%M:%S"),
                    duration,
                    exp_id,
                    name,
                    url
                )

                mylog.simpleinfo.info(line_str)
            mylog.simpleinfo.info("")

        return exps, database

    @staticmethod
    def get_exp_info(exps, include_file_info=False):

        exp_info = {
                'ID': [],
                'URL': [],
                'NAME': [],
                'DURATION': [],
                'DATETIME_0': [],
                'DATETIME_1': [],
            }
        if include_file_info:
            exp_info['FILES'] = []
        file_info = {
            'NAME': '',
            'DESC': '',
            'STATUS': '',
        }

        for exp in exps:
            dt_fr = datetime.datetime(
                exp.startyear, exp.startmonth, exp.startday, exp.starthour, exp.startmin, exp.startsec
            )
            dt_to = datetime.datetime(
                exp.endyear, exp.endmonth, exp.endday, exp.endhour, exp.endmin, exp.endsec
            )
            duration = (dt_to - dt_fr).total_seconds() / 3600

            exp_info['ID'].append(exp.id)
            exp_info['URL'].append(exp.url)
            exp_info['NAME'].append(exp.name)
            exp_info['DATETIME_0'].append(dt_fr)
            exp_info['DATETIME_1'].append(dt_to)
            exp_info['DURATION'].append(duration)

            if include_file_info:
                exp_info['FILES'].append([])
                for file in exp.files:
                    file_path_remote = pathlib.Path(file.name)
                    file_name = file_path_remote.name
                    exp_info['FILES'][-1].append(copy.deepcopy(file_info))
                    exp_info['FILES'][-1][-1]['NAME'] = file_name
                    exp_info['FILES'][-1][-1]['DESC'] = file.kindatdesc
                    exp_info['FILES'][-1][-1]['STATUS'] = file.status

        return exp_info