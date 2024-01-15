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

from geospacelab.datahub.sources.fmi import fmi_database
from geospacelab.datahub.sources.fmi.image.ie.loader import Loader as default_Loader
# from geospacelab.datahub.sources.gfz.kpap.downloader import Downloader as default_Downloader
import geospacelab.datahub.sources.fmi.image.ie.variable_config as var_config


default_dataset_attrs = {
    'database': fmi_database,
    'product': 'IL',
    'data_file_ext': 'dat',
    'data_root_dir': prf.datahub_data_root_dir / 'FMI' / 'IMAGE' / 'IE',
    'allow_load': True,
    'allow_download': True,
    'force_download': False,
    'data_search_recursive': False,
    'label_fields': ['database', 'product', 'data_file_ext'],
    'time_clip': True,
}

default_variable_names = ['DATETIME', 'IL', 'IU', 'IE', 'GEO_LAT_IL', 'GEO_LON_IL', 'GEO_LAT_IU', 'GEO_LON_IU']

# default_data_search_recursive = True

default_attrs_required = []


class Dataset(datahub.DatasetSourced):
    def __init__(self, **kwargs):
        kwargs = basic.dict_set_default(kwargs, **default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', fmi_database)
        self.product = kwargs.pop('product', 'IE')
        self.load_mode = kwargs.pop('load_mode', 'AUTO')
        self.allow_download = kwargs.pop('allow_download', False)
        self.force_download = kwargs.pop('force_download', False)

        self.metadata = None

        allow_load = kwargs.pop('allow_load', True)

        # self.config(**kwargs)

        if self.loader is None:
            self.loader = default_Loader

        if self.downloader is None:
            self.downloader = None

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

            self.metadata = load_obj.metadata

            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range()

    def search_data_files(self, **kwargs):
        dt_fr = self.dt_fr
        dt_to = self.dt_to
        diff_days = dttool.get_diff_days(dt_fr, dt_to)
        dt0 = dttool.get_start_of_the_day(dt_fr)
        for i in range(diff_days + 1):
            thisday = dt0 + datetime.timedelta(days=i)
    
            initial_file_dir = kwargs.pop('initial_file_dir', None)
            if initial_file_dir is None:
                initial_file_dir = self.data_root_dir / thisday.strftime('%Y')
            file_patterns = [thisday.strftime("%Y%m%d")]
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
        if self.data_file_ext == 'dat':
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














