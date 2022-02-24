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

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel
from geospacelab.datahub.sources.cdaweb import cdaweb_database
from geospacelab import preferences as prf
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
from geospacelab.datahub.sources.cdaweb.omni.loader import Loader as default_Loader
from geospacelab.datahub.sources.cdaweb.omni.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.cdaweb.omni.variable_config as var_config


default_dataset_attrs = {
    'facility': 'OMNI',
    'omni_type': 'omni2',
    'omni_res': '1min',
    'data_file_type': 'hres-cdf',
    'data_file_ext': 'cdf',
    'data_root_dir': prf.datahub_data_root_dir / 'CDAWeb' / 'OMNI',
    'allow_load': False,
    'allow_download': True,
    'data_search_recursive': False,
    'label_fields': ['database', 'facility', 'omni_type', 'omni_res'],
    'time_clip': True,
}

default_variable_names = ['DATETIME', 'SC_ID_IMF', 'SC_ID_PLS', 'IMF_PTS', 'PLS_PTS', 'PCT_INTERP',
                          'Timeshift', 'Timeshift_RMS',
                          'B_x_GSE', 'B_y_GSE', 'B_z_GSE',
                          'B_x_GSM', 'B_y_GSM', 'B_z_GSM',
                          'v_sw', 'v_x', 'v_y', 'v_z',
                          'n_p', 'T', 'p_dyn', 'E', 'beta', 'Ma_A', 'Ma_MSP',
                          'BSN_x', 'BSN_y', 'BSN_z']

# default_data_search_recursive = True

default_attrs_required = ['omni_type', 'omni_res']


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = cdaweb_database
        self.facility = kwargs.pop('facility', '')
        self.omni_type = kwargs.pop('omni_type', 'omni2')
        self.omni_res = kwargs.pop('omni_res', '1min')
        self.data_file_type = kwargs.pop('data_file_type','')
        self.allow_download = kwargs.pop('allow_download', True)

        self.metadata = None

        allow_load = kwargs.pop('allow_load', False)

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

        if str(self.data_file_type):
            self.data_file_ext = self.data_file_type.split('-')[1]

    def label(self, **kwargs):
        label = super().label()
        return label

    def load_data(self, **kwargs):
        self.check_data_files(**kwargs)

        for file_path in self.data_file_paths:
            load_obj = self.loader(file_path, file_type=self.data_file_type)

            for var_name in self._variables.keys():
                self._variables[var_name].join(load_obj.variables[var_name])

            self.metadata = load_obj.metadata
            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range()

    def search_data_files(self, **kwargs):
        dt_fr = self.dt_fr
        dt_to = self.dt_to
        diff_months = dttool.get_diff_months(dt_fr, dt_to)
        dt0 = dttool.get_first_day_of_month(dt_fr)
        for i in range(diff_months + 1):
            thismonth = dttool.get_next_n_months(dt0, i)
            if self.omni_res in ['1min', '5min']:
                initial_file_dir = kwargs.pop('initial_file_dir', None)
                if initial_file_dir is None:
                    initial_file_dir = self.data_root_dir / \
                                       (self.omni_type.upper() + '_high_res_' + self.omni_res) / \
                                       '{:4d}'.format(thismonth.year)
                file_patterns = [
                    self.omni_res,
                    thismonth.strftime('%Y%m%d')
                ]
            else:
                raise NotImplementedError
            # remove empty str
            search_pattern = kwargs.pop('search_pattern', None)
            if search_pattern is None:
                file_patterns = [pattern for pattern in file_patterns if str(pattern)]

                search_pattern = '*' + '*'.join(file_patterns) + '*'

            done = super().search_data_files(
                initial_file_dir=initial_file_dir, search_pattern=search_pattern)

            # Validate file paths

            if not done and self.allow_download:
                done = self.download_data()
                if done:
                    done = super().search_data_files(
                        initial_file_dir=initial_file_dir, search_pattern=search_pattern)
                else:
                    print('Cannot find files from the online database!')

        return done

    def download_data(self):
        if self.data_file_type == 'hres-cdf':
            if self.omni_type.upper() == 'OMNI':
                new_omni = False
            elif self.omni_type.upper() == 'OMNI2':
                new_omni = True
            download_obj = self.downloader(dt_fr=self.dt_fr, dt_to=self.dt_to,
                                           res=self.omni_res, new_omni=new_omni,
                                           data_file_root_dir=self.data_root_dir)
        else:
            raise NotImplementedError
        return download_obj.done

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













