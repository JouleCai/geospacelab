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

from geospacelab.datahub.sources.madrigal.isr.millstonehill.vi.loader import Loader as default_Loader
from geospacelab.datahub.sources.madrigal.isr.millstonehill.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.madrigal.isr.millstonehill.vi.variable_config as var_config
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
    'facility': 'MillstoneHillISR',
    'exp_name_pattern': '',
    'exp_check': False,
    'data_file_type': 'ion velocity',
    'data_file_ext': 'hdf5',
    'data_root_dir': prf.datahub_data_root_dir / 'Madrigal' / 'MillstoneHill_ISR',
    'allow_download': True,
    'status_control': False,
    'residual_control': False,
    'beam_location': True,
    'data_search_recursive': True,
    'label_fields': ['database', 'facility', 'site', 'data_file_type'],
}

default_variable_names = [
    'DATETIME', 'HEIGHT', 'v_i_N', 'v_i_N_err', 'v_i_E', 'v_i_E_err', 'v_i_Z', 'v_i_Z_err', 'E_N', 'E_N_err', 'E_E',
    'E_E_err', 'E_Z', 'E_Z_err', 'GEO_LAT', 'GEO_LON', 'TEC', 'n_e_max', 'h_max', 'GEO_ALT',
]

# default_data_search_recursive = True

default_attrs_required = []


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', '')
        self.facility = kwargs.pop('facility', '')
        self.site = kwargs.pop('site', MillstoneHillSite('MillstoneHill'))
        self.experiment = kwargs.pop('experiment', '')
        self.exp_name_pattern = kwargs.pop('exp_name_pattern', '')
        self.exp_check = kwargs.pop('exp_check', False)
        self.data_file_type = kwargs.pop('data_file_type', '')
        self.affiliation = kwargs.pop('affiliation', '')
        self.allow_download = kwargs.pop('allow_download', True)
        self.beam_location = kwargs.pop('beam_location', True)
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
            self.download_data()

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
        diff_days = dttool.get_diff_days(dt_fr, dt_to)
        day0 = dttool.get_start_of_the_day(dt_fr)
        for i in range(diff_days + 1):
            thisday = day0 + datetime.timedelta(days=i)
            initial_file_dir = self.data_root_dir / thisday.strftime('%Y') / thisday.strftime('%Y%m%d')

            file_patterns = ['MillstoneHill', self.data_file_type.replace(' ', '_'), thisday.strftime('%Y%m%d')]

            # remove empty str
            file_patterns = [pattern for pattern in file_patterns if str(pattern)]

            search_pattern = '*' + '*'.join(file_patterns) + '*'

            if str(self.exp_name_pattern):
                search_pattern = '*' + self.exp_name_pattern.replace(' ', '-') + '*/' + search_pattern
                recursive = False

            done = super().search_data_files(
                initial_file_dir=initial_file_dir, search_pattern=search_pattern, recursive=recursive)

            # Validate file paths

            if not done and self.allow_download:
                done = self.download_data()
                if done:
                    done = super().search_data_files(
                        initial_file_dir=initial_file_dir, search_pattern=search_pattern)
                else:
                    print('The requested experiment does not exist in the online database!')

            if len(done) > 1:
                if str(self.exp_name_pattern):
                    mylog.StreamLogger.error(
                        "Multiple data files detected! Check the files:")
                else:
                    mylog.StreamLogger.error(
                        "Multiple data files detected!" +
                        "Specify the experiment name by the keyword 'exp_name_pattern' if possible.")
                for fp in done:
                    mylog.simpleinfo.info(fp)
                raise KeyError

        return done

    def download_data(self):
        download_obj = self.downloader(
            dt_fr=self.dt_fr, dt_to=self.dt_to,
            data_file_root_dir=self.data_root_dir,
            file_type=self.data_file_type,
            exp_name_pattern=self.exp_name_pattern)
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

    @property
    def site(self):
        return self._site

    @site.setter
    def site(self, value):
        if isinstance(value, str):
            self._site = MillstoneHillSite(value)
        elif issubclass(value.__class__, SiteModel):
            self._site = value
        else:
            raise TypeError


class MillstoneHillSite(SiteModel):
    def __new__(cls, str_in, **kwargs):
        obj = super().__new__(cls, str_in, **kwargs)
        return obj

    def __init__(self, str_in, **kwargs):
        self.name = 'MillstoneHill'
        self.location = {
            'GEO_LAT': 42.619,
            'GEO_LON': 288.51,
            'GEO_ALT': 0.146
        }












