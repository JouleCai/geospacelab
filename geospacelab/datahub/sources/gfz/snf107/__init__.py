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
from geospacelab.datahub import DatabaseModel, ProductModel

from geospacelab.config import prf
import geospacelab.toolbox.utilities.pydatetime as dttool
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog

from geospacelab.datahub.sources.gfz import gfz_database
from geospacelab.datahub.sources.gfz.snf107.loader import Loader as default_Loader
from geospacelab.datahub.sources.gfz.kpap.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.gfz.snf107.variable_config as var_config


default_dataset_attrs = {
    'database': gfz_database,
    'product': 'KpAp',
    'data_file_ext': 'nc',
    'data_root_dir': prf.datahub_data_root_dir / 'GFZ' / 'Indices',
    'allow_load': True,
    'allow_download': True,
    'force_download': False,
    'data_search_recursive': False,
    'label_fields': ['database', 'product', 'data_file_ext'],
    'time_clip': True,
}

default_variable_names = ['DATETIME', 'SN', 'BSRN', 'BSRN_Days', 'F107_ADJ', 'F107_OBS']

# default_data_search_recursive = True

default_attrs_required = []


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', gfz_database)
        self.product = kwargs.pop('product', 'SNF107')
        self.allow_download = kwargs.pop('allow_download', True)
        self.force_download = kwargs.pop('force_download', True)

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

    def label(self, **kwargs):
        label = super().label()
        return label

    def load_data(self, **kwargs):
        self.check_data_files(**kwargs)

        for file_path in self.data_file_paths:
            load_obj = self.loader(file_path, file_type=self.data_file_ext)

            for var_name in self._variables.keys():
                self._variables[var_name].join(load_obj.variables[var_name])

            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range()

    def search_data_files(self, **kwargs):
        dt_fr = self.dt_fr
        dt_to = self.dt_to
        diff_years = dt_to.year - dt_fr.year
        dt0 = datetime.datetime(dt_fr.year, 1, 1)
        for i in range(diff_years + 1):
            thisyear = datetime.datetime(dt0.year + i, 1, 1)
            if datetime.date.today().year == thisyear.year:
                self.force_download = True

            initial_file_dir = kwargs.pop('initial_file_dir', None)
            if initial_file_dir is None:
                initial_file_dir = self.data_root_dir / 'SN_F107'
            file_patterns = [thisyear.strftime("%Y")]
            # remove empty str
            file_patterns = [pattern for pattern in file_patterns if str(pattern)]

            search_pattern = '*' + '*'.join(file_patterns) + '*'

            if not self.force_download:
                done = super().search_data_files(
                    initial_file_dir=initial_file_dir, search_pattern=search_pattern
                )
            else:
                done = False

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
        if self.data_file_ext == 'nc':
            download_obj = self.downloader(self.dt_fr, self.dt_to, data_file_root_dir=self.data_root_dir, force=self.force_download)
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
    def product(self):
        return self._product

    @product.setter
    def product(self, value):
        if isinstance(value, str):
            self._product = ProductModel(value)
        elif issubclass(value.__class__, ProductModel):
            self._product = value
        else:
            raise TypeError














