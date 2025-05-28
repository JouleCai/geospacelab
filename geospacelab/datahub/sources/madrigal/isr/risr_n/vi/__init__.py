# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

__author__ = "Lei Cai"
__copyright__ = "Copyright 2021, GeospaceLab"
__license__ = "BSD-3-Clause License"
__email__ = "lei.cai@oulu.fi"
__docformat__ = "reStructureText"

import datetime
import numpy as np
import h5py
import re

from geospacelab.datahub.sources.madrigal.isr.risr_n.vi.loader import Loader as default_Loader
from geospacelab.datahub.sources.madrigal.isr.risr_n.vi.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.madrigal.isr.risr_n.vi.variable_config as var_config
from geospacelab.datahub.sources.madrigal import madrigal_database
from geospacelab.config import prf
from geospacelab import datahub
from geospacelab.datahub import SiteModel, DatabaseModel, FacilityModel
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pylogging as mylog

default_dataset_attrs = {
    'kind': 'sourced',
    'database': madrigal_database,
    'facility': 'RISR-N',
    'exp_name_pattern': [],
    'exp_check': False,
    'data_file_type': 'vi',
    'data_file_ext': ['h5', 'hdf5'],
    'data_root_dir': prf.datahub_data_root_dir / 'Madrigal' / 'RISR-N',
    'allow_download': True,
    'status_control': False,
    'residual_control': False,
    'beam_location': True,
    'data_search_recursive': True,
    'label_fields': ['database', 'facility', 'site', 'data_file_type'],
}

default_variable_names = [
    'DATETIME', 'CGM_LAT', 'v_i_N', 'v_i_N_err', 'v_i_E', 'v_i_E_err', 'v_i_Z', 'v_i_Z_err', 'E_N', 'E_N_err', 'E_E',
    'E_E_err', 'GEO_ALT_MAX', 'GEO_ALT_MIN', 'INT_TIME', 
    # 'EF_MAG', 'EF_MAG_err', 'EF_ANGLE',  'EF_ANGLE_err',
    # 'v_i_MAG', 'v_i_MAG_err', 'v_i_ANGLE', 'v_i_ANGLE_err'
]

# default_data_search_recursive = True

default_attrs_required = []


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', '')
        self.facility = kwargs.pop('facility', '')
        self.site = kwargs.pop('site', RISR('RISR-N'))
        self.experiment = kwargs.pop('experiment', '')
        self.exp_ids = kwargs.pop('exp_ids', [])
        self.exp_name_pattern = kwargs.pop('exp_name_pattern', [])
        self.integration_time = kwargs.pop('integration_time', None)
        self.exp_check = kwargs.pop('exp_check', False)
        self.data_file_type = kwargs.pop('data_file_type', 'vi')
        self.affiliation = kwargs.pop('affiliation', '')
        self.allow_download = kwargs.pop('allow_download', True)
        self.metadata = None

        allow_load = kwargs.pop('allow_load', True)
        self.status_control = kwargs.pop('status_control', False)
        self.residual_control = kwargs.pop('residual_control', False)

        # self.config(**kwargs)

        if self.loader is None:
            self.loader = default_Loader
        if self.downloader is None:
            self.downloader = default_Downloader

        self._set_default_variables(
            default_variable_names,
            configured_variables=var_config.configured_variables
        )

        self._validate_attrs()

        if allow_load:
            self.load_data()

    def _validate_attrs(self):
        for attr_name in default_attrs_required:
            attr = getattr(self, attr_name)
            if not str(attr):
                mylog.StreamLogger.warning("The parameter {} is required before loading data!".format(attr_name))

        if self.exp_check:
            self.download_data(dry_run=True)

    def label(self, **kwargs):
        label = super().label()
        return label

    def load_data(self, **kwargs):
        self.check_data_files(**kwargs)

        for file_path in self.data_file_paths:
            load_obj = self.loader(file_path, )

            for var_name in self._variables.keys():
                self._variables[var_name].join(load_obj.variables[var_name])
            self.metadata = load_obj.metadata

        if self.time_clip:
            self.time_filter_by_range()

    def search_data_files(self, recursive=True, **kwargs):
        dt_fr = self.dt_fr
        dt_to = self.dt_to
        done = False
        if list(self.exp_ids):
            initial_file_dir = self.data_root_dir

            for exp_id in self.exp_ids:
                search_pattern = f"*EID-{exp_id}*/*{self.data_file_type}*"
                done = super().search_data_files(
                    initial_file_dir=initial_file_dir,
                    search_pattern=search_pattern, recursive=True, allow_multiple_files=True
                )

                if not done and self.allow_download:
                    done = self.download_data()
                    if done:
                        done = super().search_data_files(
                            initial_file_dir=initial_file_dir,
                            search_pattern=search_pattern, recursive=True, allow_multiple_files=True
                        )
                    else:
                        print('The requested experiment does not exist in the online database!')
                if len(done) > 1:
                    if isinstance(self.exp_name_pattern, (str, list)):
                        mylog.StreamLogger.warning(
                            "Multiple data files detected! Check the files:")
                    else:
                        mylog.StreamLogger.warning(
                            "Multiple data files detected!" +
                            "Specify the experiment name by the keyword 'exp_name_pattern' if possible.")
                    for fp in done:
                        mylog.simpleinfo.info(fp)

        else:
            diff_years = dt_to.year - dt_fr.year
            for ny in range(diff_years + 1):
                initial_file_dir = self.data_root_dir / str(dt_to.year + ny)
                search_pattern = "*EID-*/"
                exp_dirs = list(initial_file_dir.glob(search_pattern))

                if not list(exp_dirs) and self.allow_download:
                    self.download_data() 

                def dir_parser(dirs):
                    dirs_out = []
                    for d in dirs:
                        rc = re.compile(r'_([\d]{14})')
                        res = rc.findall(str(d))
                        dt_0 = datetime.datetime.strptime(res[0], "%Y%m%d%H%M%S")
                        dt_1 = datetime.datetime.strptime(res[1], "%Y%m%d%H%M%S")

                        if dt_1 < dt_fr:
                            continue
                        if dt_0 > dt_to:
                            continue
                        dirs_out.append(d)
                    return dirs_out

                file_dirs = dir_parser(exp_dirs)

                if not list(file_dirs) and self.allow_download:
                    self.download_data() 

                for fd in file_dirs:
                    if isinstance(self.exp_name_pattern, list):
                        p = '.*'.join(self.exp_name_pattern)
                    elif isinstance(self.exp_name_pattern, str):
                        p = self.exp_name_pattern
                    else:
                        p = '.*'
                    rc = re.compile(p)
                    res = rc.search(str(fd))
                    if res is None:
                        continue

                    file_patterns = ['RISR-N', self.data_file_type]

                    # remove empty str
                    file_patterns = [pattern for pattern in file_patterns if str(pattern)]

                    search_pattern = '*' + '*'.join(file_patterns) + '*'

                    recursive = False

                    done = super().search_data_files(
                        initial_file_dir=fd, search_pattern=search_pattern,
                        allow_multiple_files=True, recursive=recursive)

                    # Validate file paths

                    if not done and self.allow_download:
                        done = self.download_data()
                        if done:
                            done = super().search_data_files(
                                initial_file_dir=initial_file_dir,
                                search_pattern=search_pattern,
                                allow_multiple_files=True, recursive=recursive
                            )
                        else:
                            print('The requested experiment does not exist in the online database!')

                    if len(done) > 1:
                        if isinstance(self.exp_name_pattern, (str, list)):
                            mylog.StreamLogger.warning(
                                "Multiple data files detected! Check the files:")
                        else:
                            mylog.StreamLogger.warning(
                                "Multiple data files detected!" +
                                "Specify the experiment name by the keyword 'exp_name_pattern' if possible.")
                        for fp in done:
                            mylog.simpleinfo.info(fp)

        self._select_file_by_time_resolution()
        import natsort
        self.data_file_paths = natsort.natsorted(self.data_file_paths, reverse=False)
        return done

    def _select_file_by_time_resolution(self):
        def show_option():
            mylog.simpleinfo.info("Options for the integration time [s]: ")
            ts = np.unique(itg_times)
            for t in ts:
                mylog.simpleinfo.info(f"{t}")
                
        file_paths = self.data_file_paths
        itg_times = []
        for file_path in file_paths:
            with h5py.File(file_path, 'r') as fh5:
                itg_time = np.array(fh5['Data']['Array Layout']['1D Parameters']['inttms']).astype(np.float32)
                itg_times.append(np.median(itg_time))

        if self.integration_time is None:
            mylog.StreamLogger.error(f"The integration time was not set. Reset is required!")
            show_option()
            raise AttributeError
            
        if self.integration_time not in itg_times:
            mylog.StreamLogger.error(f"The integration time is not found. Reset is required!")
            show_option()
            raise AttributeError 
        
        ind = np.where(np.array(itg_times, dtype=np.float32) == self.integration_time)[0]

        file_paths = [file_paths[i] for i in ind]
        mylog.simpleinfo.info(f'The following files are selected with the integration time {self.integration_time} [s]:')
        for file_path in file_paths:
            mylog.simpleinfo.info(file_path)

        self.data_file_paths = [self.data_file_paths[i] for i in ind]

    def download_data(self, dry_run=False):
        include_exp_name_patterns = [self.exp_name_pattern] if isinstance(self.exp_name_pattern, list) else []
        download_obj = self.downloader(
            dt_fr=self.dt_fr, dt_to=self.dt_to,
            dry_run=dry_run,
            data_file_root_dir=self.data_root_dir,
            include_exp_ids=self.exp_ids,
            include_exp_name_patterns=include_exp_name_patterns)
        return download_obj.data_file_paths


    @property
    def database(self):
        return self._database

    @database.setter
    def database(self, value):
        if isinstance(value, str):
            self._database = DatabaseModel(value)
        elif issubclass(value.__class__, DatabaseModel):
            self._database = value
        else:
            raise TypeError

    @property
    def facility(self):
        return self._facility

    @facility.setter
    def facility(self, value):
        if isinstance(value, str):
            self._facility = FacilityModel(value)
        elif issubclass(value.__class__, FacilityModel):
            self._facility = value
        else:
            raise TypeError

    @property
    def site(self):
        return self._site

    @site.setter
    def site(self, value):
        if isinstance(value, str):
            self._site = RISR(value)
        elif issubclass(value.__class__, SiteModel):
            self._site = value
        else:
            raise TypeError



class RISR(SiteModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj

    def __init__(self, str_in, **kwargs):
        self.name = 'Resolute Bay North IS Radar'
        self.location = {
            'GEO_LAT': 74.72955,
            'GEO_LON': 256.09424,
            'GEO_ALT': 0.145
        }







